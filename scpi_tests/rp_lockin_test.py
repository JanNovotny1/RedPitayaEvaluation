r"""
Red Pitaya STEMlab 125-14 - generator + software lock-in loopback test.

Wiring:  OUT1 --T--> IN1   (reference)
               \--> [DUT or plain cable] --> IN2
Inputs on LV (+/-1 V) jumpers.

Prerequisite: start the SCPI server on the board
(web UI -> Development -> SCPI server -> Run), and close other apps.

Usage:  python rp_lockin_test.py [host]      (default host rp-f109f7.local)
"""
import socket
import sys
import time

import numpy as np

HOST = sys.argv[1] if len(sys.argv) > 1 else "rp-f109f7.local"
PORT = 5000
F_EX = 1_000_000      # excitation frequency, Hz
AMP = 0.5             # generator amplitude, V peak (into high-Z)
FS = 125e6            # ADC rate at decimation 1
N_CAPTURES = 20


class SCPI:
    def __init__(self, host, port=PORT, timeout=5.0):
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.buf = b""

    def tx(self, cmd):
        self.s.sendall((cmd + "\r\n").encode())

    def rx(self):
        while b"\r\n" not in self.buf:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("SCPI connection closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line.decode().strip()

    def q(self, cmd):
        self.tx(cmd)
        return self.rx()


def parse_data(resp):
    return np.array([float(v) for v in resp.strip("{}").split(",")])


def lockin(x, f=F_EX, fs=FS):
    """Return complex phasor (peak amplitude, phase) using an integer number of cycles."""
    spc = fs / f                          # samples per cycle (125 at 1 MHz)
    n = int(len(x) // spc * spc)          # integer number of cycles -> no leakage
    x = x[:n] - np.mean(x[:n])
    t = np.arange(n) / fs
    return 2 * np.mean(x * np.exp(-2j * np.pi * f * t))


def acquire(rp):
    rp.tx("ACQ:RST")
    rp.tx("ACQ:DEC 1")
    rp.tx("ACQ:DATA:UNITS VOLTS")
    rp.tx("ACQ:TRig:LEV 0")
    rp.tx("ACQ:START")
    time.sleep(0.01)                      # let the buffer fill pre-trigger samples
    rp.tx("ACQ:TRig CH1_PE")
    t0 = time.time()
    while rp.q("ACQ:TRig:STAT?") != "TD":
        if time.time() - t0 > 2:
            raise TimeoutError("No trigger - check OUT1 -> IN1 cable and generator")
    while rp.q("ACQ:TRig:FILL?") != "1":
        pass
    a = parse_data(rp.q("ACQ:SOUR1:DATA?"))
    b = parse_data(rp.q("ACQ:SOUR2:DATA?"))
    return a, b


def main():
    rp = SCPI(HOST)
    print("Connected:", rp.q("*IDN?"))

    # Generator: continuous 1 MHz sine on OUT1
    rp.tx("GEN:RST")
    rp.tx("SOUR1:FUNC SINE")
    rp.tx(f"SOUR1:FREQ:FIX {F_EX}")
    rp.tx(f"SOUR1:VOLT {AMP}")
    rp.tx("OUTPUT1:STATE ON")
    rp.tx("SOUR1:TRig:INT")
    time.sleep(0.2)

    zA, zB = [], []
    for i in range(N_CAPTURES):
        a, b = acquire(rp)
        zA.append(lockin(a))
        zB.append(lockin(b))
    zA, zB = np.array(zA), np.array(zB)
    ratio = zB / zA

    def report(name, z):
        amp, ph = np.abs(z), np.degrees(np.angle(z))
        print(f"{name:10s} amp = {amp.mean()*1e3:9.3f} mV pk  (sd {amp.std()*1e6:7.2f} uV)   "
              f"phase = {ph.mean():8.3f} deg (sd {ph.std()*1e3:6.2f} mdeg)")

    print(f"\nf = {F_EX/1e6:.3f} MHz, {len(a)} samples/capture, {N_CAPTURES} captures")
    report("IN1 (A)", zA)
    report("IN2 (B)", zB)
    print(f"{'B/A':10s} |ratio| = {np.abs(ratio).mean():.5f} (sd {np.abs(ratio).std()*1e6:.1f} ppm)   "
          f"dphase = {np.degrees(np.angle(ratio)).mean():.4f} deg "
          f"(sd {np.degrees(np.angle(ratio)).std()*1e3:.2f} mdeg)")

    rp.tx("OUTPUT1:STATE OFF")
    rp.s.close()


if __name__ == "__main__":
    main()
