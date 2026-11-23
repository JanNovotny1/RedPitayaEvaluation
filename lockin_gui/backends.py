"""
Lock-in back ends for the GUI.

PyrplLockin  - real hardware: PyRPL IQ module iq2 does excitation + demodulation in
               the FPGA; its two quadratures (DSP signals 'iq2', 'iq2_2') are routed
               to the scope, which runs in rolling mode (circular buffer written
               continuously). read_new() fetches only the samples written since the
               previous call -> gapless I/Q stream as long as it is called more
               often than once per buffer length (1.07 s at decimation 8192).
SimLockin    - synthetic data with the same interface, for testing without a board.
"""
import time
from dataclasses import asdict, dataclass

import numpy as np

FPGA_CLOCK = 125e6
SCOPE_LENGTH = 2**14            # scope buffer length [samples]
CH1_BASE, CH2_BASE = 0x10000, 0x20000


@dataclass
class LockinSettings:
    frequency: float = 1e6       # excitation / demodulation frequency [Hz]
    amplitude: float = 0.25      # excitation amplitude [V], PyRPL value (into 50 ohm):
                                 # a 1 MOhm input sees ~2x this -> keep <= 0.35 V on LV jumpers
    output: str = "out1"         # 'out1', 'out2' or 'off'
    input: str = "in1"           # 'in1' or 'in2'
    phase: float = 0.0           # demodulation phase [deg]
    bandwidth: float = 500.0     # I/Q low-pass corner [Hz] (rounded to a valid value)
    filter_order: int = 2        # 1 or 2 low-pass stages
    acbandwidth: float = 10e3    # input AC-coupling high-pass [Hz], 0 = DC coupled
    quadrature_factor: float = 1.0  # digital gain on I/Q before readout


def choose_decimation(out_rate, oversample=8, min_buffer_s=0.4):
    """Largest power-of-two decimation giving fs >= oversample*out_rate while
    keeping the circular buffer >= min_buffer_s long when possible."""
    best = 1
    for k in range(17):
        dec = 2**k
        fs = FPGA_CLOCK / dec
        if fs >= oversample * out_rate:
            best = dec
    # make sure buffer isn't too short for a 50 ms polling loop
    while SCOPE_LENGTH * best / FPGA_CLOCK < min_buffer_s and best < 2**16:
        if FPGA_CLOCK / (best * 2) < out_rate:
            break
        best *= 2
    return best


IQ_GAIN_DEFAULT = 0.41   # |I+jQ| per volt of input amplitude at quadrature_factor 1,
                         # measured on STEMlab 125-14 / OS 3.00 / RedPitaya PyRPL fork
                         # (theory for an ideal mixer: 0.5). Recalibrate with tools/pyrpl_diag.py.


def iq_to_volts(z, quadrature_factor, iq_gain=IQ_GAIN_DEFAULT):
    """Input amplitude [V pk] from raw I/Q: |I+jQ| = iq_gain * quadrature_factor * A."""
    return np.abs(z) / max(iq_gain * quadrature_factor, 1e-12)


class PyrplLockin:
    name = "PyRPL (FPGA)"

    def __init__(self, host, password="root", config="rp_lockin_gui"):
        self.host, self.password, self.config = host, password, config
        self.p = self.r = self.iq = self.scope = None
        self.fs = None
        self._wp_last = None
        self._t_last = None
        self.overruns = 0

    # --- connection -------------------------------------------------------
    def connect(self):
        from pyrpl import Pyrpl  # imported late: pyrpl creates the QApplication if none exists
        self.p = Pyrpl(config=self.config, hostname=self.host, password=self.password, gui=False)
        self.r = self.p.rp
        self.iq = self.r.iq2
        self.scope = self.r.scope
        try:
            self.scope.stop()          # a previous session may have left it running
        except Exception:
            pass
        for name in ("asg0", "asg1", "iq0", "iq1"):   # nothing else drives the outputs
            try:
                getattr(self.r, name).output_direct = "off"
            except Exception:
                pass
        # PyRPL caches the filter constants (stages, shift bits, min bandwidth) on first
        # read. Read right after the FPGA image was (re)loaded they can be wrong, which
        # limits the low-pass list to ~1 Hz. Wait, drop the cache and read them again.
        time.sleep(1.0)
        self._refresh_filter_cache()

    def _refresh_filter_cache(self):
        for attr in ("bandwidth", "acbandwidth"):
            for key in ("filterstages", "shiftbits", "minbw"):
                self.iq.__dict__.pop(f"_{attr}_{key}", None)
            self.iq.__dict__.pop(f"{attr}_options", None)

    def bandwidths(self):
        try:
            bws = sorted({float(b) for b in self.iq.bandwidths if b > 0})
            if bws and max(bws) < 1000:        # stale cache -> try once more
                time.sleep(1.0)
                self._refresh_filter_cache()
                bws = sorted({float(b) for b in self.iq.bandwidths if b > 0})
            return bws
        except Exception:
            return [10, 30, 100, 300, 500, 1000, 3000, 10000]

    # --- settings ---------------------------------------------------------
    def apply(self, s: LockinSettings):
        self.iq.setup(frequency=s.frequency,
                      bandwidth=[s.bandwidth] * int(s.filter_order),
                      gain=0.0,
                      phase=s.phase,
                      acbandwidth=s.acbandwidth,
                      amplitude=s.amplitude,
                      input=s.input,
                      output_direct=s.output,
                      output_signal="quadrature",
                      quadrature_factor=s.quadrature_factor)
        return self.actual()

    def actual(self):
        iq = self.iq
        return dict(frequency=iq.frequency, amplitude=iq.amplitude, phase=iq.phase,
                    bandwidth=list(iq.bandwidth), acbandwidth=iq.acbandwidth,
                    quadrature_factor=iq.quadrature_factor, input=iq.input,
                    output=iq.output_direct)

    # --- streaming --------------------------------------------------------
    def start_stream(self, decimation):
        sc = self.scope
        sc.input1 = "iq2"
        sc.input2 = "iq2_2"
        sc.decimation = decimation
        sc.average = True            # average within each decimation interval
        sc.trigger_source = "immediately"
        sc._start_acquisition_rolling_mode()
        self.fs = FPGA_CLOCK / decimation
        self._wp_last = int(sc._write_pointer_current)
        self._t_last = time.perf_counter()
        self.overruns = 0
        return self.fs

    def _read_range(self, base, start, n):
        vals = self.scope._reads(base + 4 * start, n)
        x = np.asarray(vals, dtype=np.int64) & 0x3FFF
        x[x >= 2**13] -= 2**14
        return x / 2.0**13

    def _read_both(self, start, n):
        return (self._read_range(CH1_BASE, start, n),
                self._read_range(CH2_BASE, start, n))

    def read_new(self):
        """Return complex raw I+jQ samples written since the last call."""
        now = time.perf_counter()
        if (now - self._t_last) * self.fs > 0.95 * SCOPE_LENGTH:
            self.overruns += 1        # buffer may have wrapped -> data lost
        self._t_last = now

        wp = int(self.scope._write_pointer_current)   # index of last written sample
        n = (wp - self._wp_last) % SCOPE_LENGTH
        if n == 0:
            return np.empty(0, dtype=complex)
        start = (self._wp_last + 1) % SCOPE_LENGTH
        if start + n <= SCOPE_LENGTH:
            i, q = self._read_both(start, n)
        else:
            n1 = SCOPE_LENGTH - start
            i1, q1 = self._read_both(start, n1)
            i2, q2 = self._read_both(0, n - n1)
            i, q = np.concatenate([i1, i2]), np.concatenate([q1, q2])
        self._wp_last = wp
        return i + 1j * q

    def stop_stream(self):
        pass

    def close(self):
        try:
            self.iq.output_direct = "off"
            self.iq.amplitude = 0
        except Exception:
            pass


class SimLockin:
    """Synthetic lock-in: DUT = gain 0.98, phase 30 deg, slow drift, white noise."""
    name = "Simulator"

    def __init__(self, *args, **kwargs):
        self.s = LockinSettings()
        self.fs = None
        self._t_last = None
        self._phi_noise = 0.0
        self.overruns = 0
        self.rng = np.random.default_rng()

    def connect(self):
        time.sleep(0.2)

    def bandwidths(self):
        return [float(x) for x in (10, 30, 100, 300, 500, 1000, 3000, 10000)]

    def apply(self, s):
        self.s = LockinSettings(**asdict(s))
        return self.actual()

    def actual(self):
        s = self.s
        return dict(frequency=s.frequency, amplitude=s.amplitude, phase=s.phase,
                    bandwidth=[s.bandwidth] * s.filter_order, acbandwidth=s.acbandwidth,
                    quadrature_factor=s.quadrature_factor, input=s.input, output=s.output)

    def start_stream(self, decimation):
        self.fs = FPGA_CLOCK / decimation
        self._t_last = time.perf_counter()
        self._n = 0
        return self.fs

    def read_new(self):
        now = time.perf_counter()
        n = min(int((now - self._t_last) * self.fs), SCOPE_LENGTH)
        if n == 0:
            return np.empty(0, dtype=complex)
        self._t_last += n / self.fs
        s = self.s
        t = (self._n + np.arange(n)) / self.fs
        self._n += n
        amp_in = 0.0 if s.output == "off" else 0.98 * s.amplitude * (1 + 2e-4 * np.sin(2 * np.pi * 0.2 * t))
        phi = np.radians(30.0 - s.phase) + 2e-3 * np.sin(2 * np.pi * 0.05 * t)
        z = s.quadrature_factor * IQ_GAIN_DEFAULT * 2 * amp_in * np.exp(1j * phi)   # 2x: 1 MOhm input
        noise_sd = 2e-4 * np.sqrt(s.bandwidth / 500.0) * s.quadrature_factor
        z = z + noise_sd * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))
        # 14-bit quantisation and +-1 V clipping like the real scope
        z = np.clip(np.round(z.real * 8192), -8192, 8191) / 8192 + 1j * np.clip(np.round(z.imag * 8192), -8192, 8191) / 8192
        return z

    def stop_stream(self):
        pass

    def close(self):
        pass
