"""
PyRPL diagnostics for the lock-in GUI.

Prints what the FPGA reports for the IQ module filters, which modules drive the
outputs, the real input amplitude at IN1 (raw scope, 125 MS/s), the matching
|I+jQ| from iq2, and the residual 1 MHz signal on IN1 with the excitation off.

Usage (SCPI server and web apps stopped, OUT1 -> IN1 cable connected):
    python tools\\pyrpl_diag.py 169.254.101.38 [--amp 0.2]
Keep --amp <= 0.35 V with LV jumpers: a 1 MOhm input sees about twice the PyRPL amplitude.
"""
import argparse
import sys
import time

import numpy as np
from qtpy import QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
from pyrpl import Pyrpl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("host", nargs="?", default="169.254.101.38")
ap.add_argument("--amp", type=float, default=0.2, help="PyRPL excitation amplitude [V]")
ap.add_argument("--freq", type=float, default=1e6)
args = ap.parse_args()
HOST, F, AMP = args.host, args.freq, args.amp


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def tone(x, fs, f=F):
    """amplitude [V pk] and phase of tone f in x (integer number of cycles)"""
    spc = fs / f
    n = int(len(x) // spc * spc) if spc >= 1 else len(x)
    t = np.arange(n) / fs
    z = 2 * np.mean((x[:n] - x[:n].mean()) * np.exp(-2j * np.pi * f * t))
    return abs(z), np.degrees(np.angle(z))


def scope_trace(r, in1, in2, decimation):
    s = r.scope
    s.stop()
    s.input1, s.input2 = in1, in2
    s.decimation = decimation
    s.average = True
    s.trigger_source = "immediately"
    s.rolling_mode = False
    s.trace_average = 1
    data = np.asarray(s.single())
    return data[0], data[1], 125e6 / decimation


p = Pyrpl(config="rp_diag", hostname=HOST, gui=False)
r = p.rp
iq = r.iq2

section("1. IQ filter configuration reported by the FPGA (iq2)")
try:
    print("filterstages reg 0x230 =", iq._read(0x230))
    print("shiftbits    reg 0x234 =", iq._read(0x234))
    print("minbw        reg 0x238 =", iq._read(0x238))
except Exception as e:
    print("register read failed:", e)
try:
    bws = sorted({float(b) for b in iq.bandwidths if b > 0})
    print(f"valid low-pass bandwidths ({len(bws)}): min {min(bws):.4g} Hz, max {max(bws):.4g} Hz")
    print("  ", ", ".join(f"{b:.4g}" for b in bws))
except Exception as e:
    print("bandwidths failed:", e)

section("2. Which modules drive OUT1 / OUT2")
for name in ("asg0", "asg1", "iq0", "iq1", "iq2", "pid0", "pid1", "pid2"):
    try:
        m = getattr(r, name)
        od = getattr(m, "output_direct", "?")
        extra = ""
        if hasattr(m, "frequency"):
            extra += f" f={m.frequency:.3f} Hz"
        if hasattr(m, "amplitude"):
            extra += f" amp={m.amplitude:.4f} V"
        print(f"{name:5s} output_direct={od}{extra}")
    except Exception as e:
        print(f"{name:5s} ?: {e}")

section(f"3. Set iq2 like the GUI: {F/1e6:g} MHz, {AMP} V, out1/in1, LPF 500 Hz x2, AC 10 kHz")
for name in ("asg0", "asg1", "iq0", "iq1"):
    try:
        getattr(r, name).output_direct = "off"
    except Exception:
        pass
iq.setup(frequency=F, bandwidth=[500, 500], gain=0.0, phase=0, acbandwidth=10e3,
         amplitude=AMP, input="in1", output_direct="out1",
         output_signal="quadrature", quadrature_factor=1.0)
time.sleep(0.5)
print("actual: f =", iq.frequency, " amp =", iq.amplitude, " bandwidth =", iq.bandwidth,
      " acbandwidth =", iq.acbandwidth, " qf =", iq.quadrature_factor)

section("4. Raw IN1 at 125 MS/s vs demodulated |I+jQ|")
x1, x2, fs = scope_trace(r, "in1", "out1", 1)
a_in, ph_in = tone(x1, fs)
a_out, _ = tone(x2, fs)
print(f"OUT1 (internal) tone : {a_out:.4f} V pk")
print(f"IN1 tone             : {a_in:.4f} V pk   (IN1 min/max {x1.min():+.3f}/{x1.max():+.3f} V)")
i_tr, q_tr, fs2 = scope_trace(r, "iq2", "iq2_2", 8192)
zi = i_tr + 1j * q_tr
print(f"iq2 at 15.26 kS/s    : |I+jQ| = {np.abs(zi).mean():.5f}, I = {i_tr.mean():+.5f}, Q = {q_tr.mean():+.5f}")
clipped = max(abs(x1.min()), abs(x1.max())) > 0.97
if clipped:
    print("WARNING: IN1 reaches the +-1 V limit -> clipping, lower --amp")
if a_in > 0:
    print(f"IQ gain = |I+jQ| / A_in = {np.abs(zi).mean() / a_in:.4f}   "
          f"(enter this as 'IQ gain (cal.)' in the GUI)")
    print(f"IN1 / OUT1 amplitude    = {a_in / max(a_out, 1e-12):.3f}   (2.0 expected for a 1 MOhm input)")

section("4b. Ripple of |I+jQ| and phase over 8.6 s (decimation 65536, 1.9 kS/s)")
i_tr, q_tr, fs3 = scope_trace(r, "iq2", "iq2_2", 65536)
zi = i_tr + 1j * q_tr
mag = np.abs(zi)
ph = np.unwrap(np.angle(zi))
print(f"|I+jQ| mean {mag.mean():.5f}, sd {mag.std()/mag.mean()*1e6:.0f} ppm, "
      f"peak-peak {np.ptp(mag)/mag.mean()*100:.3f} %")
print(f"phase  sd {np.degrees(ph.std())*1e3:.1f} mdeg, peak-peak {np.degrees(np.ptp(ph)):.3f} deg")
win = np.hanning(len(mag))
spec = np.abs(np.fft.rfft((mag - mag.mean()) * win)) * 4 / len(mag)
fr = np.fft.rfftfreq(len(mag), 1 / fs3)
sel = fr > 0.3
top = np.argsort(np.where(sel, spec, 0))[-5:][::-1]
print("strongest ripple components in |I+jQ|:")
for kk in top:
    print(f"   {fr[kk]:8.3f} Hz   {spec[kk]/mag.mean()*1e6:8.0f} ppm of |I+jQ|")

section("5. Excitation OFF: anything still at 1 MHz on IN1? (pickup / other sources)")
iq.amplitude = 0
iq.output_direct = "off"
time.sleep(0.3)
x1, _, fs = scope_trace(r, "in1", "in2", 1)
a_res, _ = tone(x1, fs)
print(f"IN1 residual at 1 MHz: {a_res*1e3:.3f} mV pk   (IN1 rms {x1.std()*1e3:.3f} mV)")
x1, _, fs = scope_trace(r, "in1", "in2", 8)    # 15.6 MS/s, 1.05 ms window
spec = np.abs(np.fft.rfft((x1 - x1.mean()) * np.hanning(len(x1)))) * 4 / len(x1)
fr = np.fft.rfftfreq(len(x1), 1 / fs)
top = np.argsort(spec)[-5:][::-1]
print("strongest IN1 components with excitation off:")
for kk in top:
    print(f"   {fr[kk]/1e3:9.2f} kHz   {spec[kk]*1e3:8.3f} mV pk")

print("\nDone. Excitation left OFF.")
