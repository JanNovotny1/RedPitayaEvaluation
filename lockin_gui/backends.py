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


IQ_GAIN_DEFAULT = 0.5    # |I+jQ| per volt of input amplitude at quadrature_factor 1.
                         # Measured 0.5004 on STEMlab 125-14 / OS 3.00 / RedPitaya PyRPL fork
                         # (0.2 V excitation, no clipping) = ideal mixer value.
                         # Recalibrate with tools/pyrpl_diag.py.


HIRES_DEFAULT_SCALE = 2.0**-22   # 24-bit LSB -> stream units at qf = 1 (PyRPL's na rescale);
                                 # the GUI replaces it by a calibration against the 14-bit stream


def _signed(v, bits):
    return v - (1 << bits) if v >> (bits - 1) else v


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

    # --- high-resolution readout --------------------------------------------
    # The IQ block sums the 24-bit low-pass-filtered quadratures (quadrature1/2, before
    # the quadrature factor and the cut to 14 bits) in 62-bit accumulators over
    # na_averages clock cycles (registers 0x130, 0x140-0x14C). Writing the frequency
    # register (0x108) restarts the sum. One value per averaging time, plus the network
    # round trip as dead time between values.
    REG_FREQ, REG_NA_AVG, REG_NA_SLEEP, REG_NA_DATA = 0x108, 0x130, 0x134, 0x140

    def start_hires(self, t_avg):
        self.hires_n = max(int(round(t_avg * FPGA_CLOCK)), 1)
        self.iq._write(self.REG_NA_SLEEP, 0)
        self.iq._write(self.REG_NA_AVG, self.hires_n)
        self._freq_word = int(self.iq._read(self.REG_FREQ))
        self._hires_restart()

    def _hires_restart(self):
        self.iq._write(self.REG_FREQ, self._freq_word)   # same frequency: restarts the sum
        self._hires_t0 = time.perf_counter()

    def poll_hires(self):
        """None while averaging; else (t_start, t_end, mean_raw) with mean_raw the complex
        mean of the 24-bit quadratures in LSB units. Restarts the next average at once."""
        if time.perf_counter() - self._hires_t0 < self.hires_n / FPGA_CLOCK:
            return None
        a, b, c, d = (int(v) for v in self.iq._reads(self.REG_NA_DATA, 4))
        if (a | b | c | d) >> 31:
            return None                                   # still averaging
        t_start, t_end = self._hires_t0, time.perf_counter()
        self._hires_restart()
        i_sum = _signed((a & 0x7FFFFFFF) | ((b & 0x7FFFFFFF) << 31), 62)
        q_sum = _signed((c & 0x7FFFFFFF) | ((d & 0x7FFFFFFF) << 31), 62)
        return t_start, t_end, complex(i_sum, q_sum) / self.hires_n

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

    def _sim_z(self, t, n_avg_samples=1, quantise=True):
        s = self.s
        amp_in = 0.0 if s.output == "off" else 0.98 * s.amplitude * (1 + 2e-4 * np.sin(2 * np.pi * 0.2 * t))
        phi = np.radians(30.0 - s.phase) + 2e-3 * np.sin(2 * np.pi * 0.05 * t)
        return IQ_GAIN_DEFAULT * 2 * amp_in * np.exp(1j * phi)

    def start_hires(self, t_avg):
        self.hires_t = t_avg
        self._hires_t0 = time.perf_counter()
        self._sim_t0 = getattr(self, "_sim_t0", self._hires_t0)

    def poll_hires(self):
        now = time.perf_counter()
        if now - self._hires_t0 < self.hires_t + 0.002:   # + simulated network dead time
            return None
        t_start = self._hires_t0
        self._hires_t0 = now
        z = self._sim_z(t_start - self._sim_t0)
        noise_sd = 2e-4 * np.sqrt(1e-3 / max(self.hires_t, 1e-6))   # white noise averages down
        z = z + noise_sd * (self.rng.standard_normal() + 1j * self.rng.standard_normal())
        # raw 24-bit LSB units, same convention as the hardware (qf not applied)
        return t_start, now, z / HIRES_DEFAULT_SCALE

    def close(self):
        pass
