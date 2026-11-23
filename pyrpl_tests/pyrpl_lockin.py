"""
PyRPL FPGA lock-in at 1 MHz on Red Pitaya (loopback test).

Wiring: OUT1 -> IN1 (direct cable, or through your DUT). Inputs on LV jumpers.

The IQ module runs entirely in the FPGA at 125 MS/s:
  - its NCO generates the 1 MHz excitation on OUT1 (amplitude below)
  - it demodulates IN1 against the same NCO and low-pass filters I and Q
Python only reads the filtered I/Q values.

Install on the PC (development version, recommended for Red Pitaya OS 2/3):
  pip install "git+https://github.com/pyrpl-fpga/pyrpl.git#egg=pyrpl[qt-pyqt5]"

Notes:
  - PyRPL connects over SSH and loads its OWN FPGA bitstream. Close the
    Red Pitaya web apps first; reboot the board to get the stock apps back.
  - iq2 is used because it is the IQ module with both quadratures
    available as signals ('iq2' = I, 'iq2_2' = Q).
"""
import sys
import time

import numpy as np
from pyrpl import Pyrpl

HOST = sys.argv[1] if len(sys.argv) > 1 else "rp-f109f7.local"
F_EX = 1e6          # excitation / demod frequency, Hz
AMP = 0.25          # excitation amplitude on OUT1, V
LPF = 500           # demod low-pass bandwidth, Hz (PyRPL rounds to nearest allowed)
N = 200             # number of I/Q readings

p = Pyrpl(config="nanosensor_lockin", hostname=HOST, gui=False)
r = p.rp
iq = r.iq2

iq.setup(frequency=F_EX,
         bandwidth=[LPF, LPF],      # 2nd-order low-pass on I and Q
         gain=0.0,                  # no feedback path
         phase=0,                   # demod phase, deg (rotate to put signal in I)
         acbandwidth=10e3,          # AC-coupling high-pass on the input, Hz
         amplitude=AMP,
         input="in1",
         output_direct="out1",      # excitation goes out on OUT1
         output_signal="quadrature",
         quadrature_factor=1)

print(f"IQ set: f = {iq.frequency:.3f} Hz, LPF = {iq.bandwidth} Hz, "
      f"AC = {iq.acbandwidth} Hz, amp = {iq.amplitude} V")
time.sleep(0.2)                     # let the filters settle

I = np.empty(N)
Q = np.empty(N)
for k in range(N):
    I[k] = r.sampler.iq2
    Q[k] = r.sampler.iq2_2
    time.sleep(0.005)

z = I + 1j * Q
amp = np.abs(z)
ph = np.degrees(np.angle(z))
print(f"I = {I.mean():+.6f}  Q = {Q.mean():+.6f}  (raw IQ units)")
print(f"|IQ| = {amp.mean():.6f}  sd {amp.std():.2e}  ({amp.std()/amp.mean()*1e6:.1f} ppm)")
print(f"phase = {ph.mean():+.3f} deg  sd {ph.std()*1e3:.2f} mdeg")
