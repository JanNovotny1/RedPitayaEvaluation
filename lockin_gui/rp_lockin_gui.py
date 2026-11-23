"""
Red Pitaya lock-in GUI: set excitation and demodulation parameters, stream I/Q at a
fixed output rate (default 1 kHz), plot live and record to CSV.

Signal chain (hardware mode):
  FPGA  : PyRPL IQ module iq2 -> excitation sine on OUT1/OUT2, demodulation of IN1/IN2,
          I/Q low-pass filters -> scope ch1 = I ('iq2'), ch2 = Q ('iq2_2'),
          scope in rolling mode at decimation chosen for the output rate
          (8192 -> 15.26 kS/s for 1 kHz output)
  PC    : every 50 ms read only the new scope samples (gapless), average into exact
          1/rate bins, optional software phase rotation, plot, record

Usage:
  python lockin_gui\\rp_lockin_gui.py                 # hardware, host field in GUI
  python lockin_gui\\rp_lockin_gui.py --simulate      # no board needed
"""
import argparse
import csv
import datetime as dt
import os
import sys
import time
from dataclasses import asdict

import numpy as np

# Create the Qt application before pyrpl is imported (pyrpl reuses an existing one)
from qtpy import QtCore, QtGui, QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

import pyqtgraph as pg  # noqa: E402

from backends import (IQ_GAIN_DEFAULT, LockinSettings, PyrplLockin,  # noqa: E402
                      SimLockin, choose_decimation, iq_to_volts)
from binning import Binner  # noqa: E402

POLL_MS = 50
SCATTER_POINTS = 2000
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


class RingBuffer:
    def __init__(self, size):
        self.size = int(size)
        self.t = np.empty(0)
        self.z = np.empty(0, dtype=complex)

    def extend(self, t, z):
        self.t = np.concatenate([self.t, t])[-self.size:]
        self.z = np.concatenate([self.z, z])[-self.size:]

    def resize(self, size):
        self.size = int(size)
        self.t, self.z = self.t[-self.size:], self.z[-self.size:]

    def clear(self):
        self.t = np.empty(0)
        self.z = np.empty(0, dtype=complex)


class LockinWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Red Pitaya lock-in")
        self.resize(1400, 860)
        self.backend = None
        self.streaming = False
        self.binner = None
        self.rotation = 0.0                 # software phase rotation [rad]
        self.rec_file = None
        self.rec_writer = None
        self.rec_rows = 0
        self.samples_in = 0
        self.t_stream0 = None
        self.buf = RingBuffer(10 * 1000)

        self._build_ui(args)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self, args):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QHBoxLayout(central)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(340)
        pl = QtWidgets.QVBoxLayout(panel)
        lay.addWidget(panel)

        # connection
        g = QtWidgets.QGroupBox("Connection")
        f = QtWidgets.QFormLayout(g)
        self.host = QtWidgets.QLineEdit(args.host)
        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password.setPlaceholderText("root (PyRPL default)")
        self.sim = QtWidgets.QCheckBox("Simulate (no board)")
        self.sim.setChecked(args.simulate)
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self._connect)
        f.addRow("Host", self.host)
        f.addRow("SSH password", self.password)
        f.addRow(self.sim)
        f.addRow(self.btn_connect)
        pl.addWidget(g)

        # excitation
        g = QtWidgets.QGroupBox("Excitation")
        f = QtWidgets.QFormLayout(g)
        self.freq = self._dspin(0, 62.5e6, 1e6, 3, " Hz", 1000)
        self.amp = self._dspin(0, 1.0, 0.25, 4, " V", 0.01)
        self.amp.setToolTip("PyRPL amplitude (into 50 ohm). A direct cable into IN1 (1 MOhm)\n"
                            "sees about twice this: keep <= 0.35 V with LV jumpers (+-1 V range).")
        self.out = QtWidgets.QComboBox()
        self.out.addItems(["out1", "out2", "off"])
        f.addRow("Frequency", self.freq)
        f.addRow("Amplitude", self.amp)
        f.addRow("Output", self.out)
        pl.addWidget(g)

        # demodulation
        g = QtWidgets.QGroupBox("Demodulation")
        f = QtWidgets.QFormLayout(g)
        self.inp = QtWidgets.QComboBox()
        self.inp.addItems(["in1", "in2"])
        self.phase = self._dspin(-360, 360, 0, 3, " deg", 1)
        self.bw = QtWidgets.QComboBox()
        self.order = QtWidgets.QComboBox()
        self.order.addItems(["1", "2"])
        self.order.setCurrentText("2")
        self.acbw = self._dspin(0, 1e6, 10e3, 1, " Hz", 1000)
        self.qf = self._dspin(1, 1000, 1, 2, "", 1)
        self.qf.setToolTip("Digital gain on I/Q before readout. Raise it for small signals,\n"
                           "but keep |I|,|Q| below ~0.9 (readout full scale is 1).")
        f.addRow("Input", self.inp)
        f.addRow("Phase (HW)", self.phase)
        f.addRow("Low-pass BW", self.bw)
        f.addRow("Filter order", self.order)
        f.addRow("AC coupling", self.acbw)
        f.addRow("Quadrature factor", self.qf)
        self.iqgain = self._dspin(0.01, 10, IQ_GAIN_DEFAULT, 4, "", 0.01)
        self.iqgain.setToolTip("Calibration: |I+jQ| per volt of input amplitude at quadrature factor 1.\n"
                               "Measured 0.5004 on this board (ideal 0.5); tools/pyrpl_diag.py prints it.")
        f.addRow("IQ gain (cal.)", self.iqgain)
        pl.addWidget(g)

        # readout
        g = QtWidgets.QGroupBox("Readout")
        f = QtWidgets.QFormLayout(g)
        self.rate = self._dspin(1, 10000, 1000, 1, " Hz", 100)
        self.window = self._dspin(1, 600, 10, 1, " s", 1)
        self.window.valueChanged.connect(self._resize_buffer)
        self.dec_label = QtWidgets.QLabel("-")
        self.dec_label.setWordWrap(True)
        self.zoom_cluster = QtWidgets.QCheckBox("Phasor: zoom to cloud")
        self.zoom_cluster.setChecked(True)
        self.zoom_cluster.toggled.connect(self._reset_phasor_range)
        self.units = QtWidgets.QComboBox()
        self.units.addItems(["V rms", "V pk"])
        self.units.currentTextChanged.connect(self._units_changed)
        self.follow = QtWidgets.QCheckBox("Follow newest data")
        self.follow.setChecked(True)
        self.follow.toggled.connect(lambda on: on and self._reset_view())
        f.addRow("Output rate", self.rate)
        f.addRow("Display window", self.window)
        f.addRow("Scope sampling", self.dec_label)
        f.addRow("Amplitude units", self.units)
        f.addRow(self.zoom_cluster)
        f.addRow(self.follow)
        pl.addWidget(g)

        # buttons
        bl = QtWidgets.QGridLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply settings")
        self.btn_apply.clicked.connect(self._apply)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self._toggle_stream)
        self.btn_autophase = QtWidgets.QPushButton("Auto-phase")
        self.btn_autophase.setToolTip("Rotate I/Q in software so the signal lies on +I (Q = 0)")
        self.btn_autophase.clicked.connect(self._autophase)
        self.btn_rot0 = QtWidgets.QPushButton("Reset rotation")
        self.btn_rot0.clicked.connect(self._reset_rotation)
        self.btn_clear = QtWidgets.QPushButton("Clear plots")
        self.btn_clear.clicked.connect(self._clear_plots)
        self.btn_view = QtWidgets.QPushButton("Reset view")
        self.btn_view.setToolTip("Scroll with new data again and auto-scale all axes")
        self.btn_view.clicked.connect(self._reset_view)
        self.btn_rec = QtWidgets.QPushButton("Record CSV")
        self.btn_rec.setCheckable(True)
        self.btn_rec.toggled.connect(self._toggle_record)
        for i, b in enumerate([self.btn_apply, self.btn_start, self.btn_autophase,
                               self.btn_rot0, self.btn_clear, self.btn_rec, self.btn_view]):
            bl.addWidget(b, i // 2, i % 2)
        pl.addLayout(bl)
        for w in (self.btn_apply, self.btn_start, self.btn_autophase, self.btn_rot0, self.btn_rec):
            w.setEnabled(False)

        self.stats = QtWidgets.QLabel("")
        self.stats.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.stats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        pl.addWidget(self.stats)
        pl.addStretch(1)

        # plots
        pg.setConfigOptions(antialias=False)
        self.gl = pg.GraphicsLayoutWidget()
        lay.addWidget(self.gl, 1)
        self.p_iq = self.gl.addPlot(row=0, col=0, title="I and Q")
        self.p_iq.addLegend(offset=(10, 5))
        self.c_i = self.p_iq.plot(pen=pg.mkPen("#1f77b4", width=1), name="I")
        self.c_q = self.p_iq.plot(pen=pg.mkPen("#ff7f0e", width=1), name="Q")
        self.p_r = self.gl.addPlot(row=1, col=0, title="Amplitude R at input (V rms)")
        self.c_r = self.p_r.plot(pen=pg.mkPen("#2ca02c", width=1))
        self.p_ph = self.gl.addPlot(row=2, col=0, title="Phase (deg)")
        self.c_ph = self.p_ph.plot(pen=pg.mkPen("#d62728", width=1))
        for p in (self.p_iq, self.p_r, self.p_ph):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("bottom", "time", units="s")
            p.setDownsampling(auto=True, mode="peak")
            p.disableAutoRange()                # ranges are set explicitly in _update_plots
            p.getViewBox().setFlag(p.getViewBox().GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self.p_r.setXLink(self.p_iq)
        self.p_ph.setXLink(self.p_iq)
        self.p_sc = self.gl.addPlot(row=0, col=1, rowspan=3, title="Phasor I/Q (last 2000 points)")
        self.p_sc.setAspectLocked(True)
        self.p_sc.showGrid(x=True, y=True, alpha=0.3)
        self.p_sc.setLabel("bottom", "I")
        self.p_sc.setLabel("left", "Q")
        self.p_sc.addLine(x=0, pen=pg.mkPen("w", width=0.5))
        self.p_sc.addLine(y=0, pen=pg.mkPen("w", width=0.5))
        self.sc = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(31, 119, 180, 160))
        self.p_sc.addItem(self.sc)
        self.gl.ci.layout.setColumnStretchFactor(0, 3)
        self.gl.ci.layout.setColumnStretchFactor(1, 2)

        self._sc_c = None                       # phasor view centre
        self._sc_half = None                    # phasor view half-width
        self.status = self.statusBar()
        self.status.showMessage("Not connected")

    @staticmethod
    def _dspin(lo, hi, val, dec, suffix, step):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setValue(val)
        w.setSuffix(suffix)
        w.setSingleStep(step)
        w.setKeyboardTracking(False)
        return w

    # ------------------------------------------------------------ settings
    def settings(self):
        return LockinSettings(
            frequency=self.freq.value(), amplitude=self.amp.value(),
            output=self.out.currentText(), input=self.inp.currentText(),
            phase=self.phase.value(), bandwidth=float(self.bw.currentText()),
            filter_order=int(self.order.currentText()), acbandwidth=self.acbw.value(),
            quadrature_factor=self.qf.value())

    def _connect(self):
        self.btn_connect.setEnabled(False)
        self.status.showMessage("Connecting... (PyRPL may take ~10-30 s to upload its FPGA image)")
        QtWidgets.QApplication.processEvents()
        try:
            if self.sim.isChecked():
                self.backend = SimLockin()
            else:
                self.backend = PyrplLockin(self.host.text().strip(),
                                           password=self.password.text() or "root")
            self.backend.connect()
        except Exception as e:
            self.backend = None
            self.btn_connect.setEnabled(True)
            self.status.showMessage(f"Connection failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Connection failed", str(e))
            return
        self.password.clear()
        self.bw.clear()
        bws = self.backend.bandwidths()
        self.bw.addItems([f"{b:g}" for b in bws])
        target = min(bws, key=lambda b: abs(b - 500))
        self.bw.setCurrentText(f"{target:g}")
        for w in (self.btn_apply, self.btn_start, self.btn_autophase, self.btn_rot0, self.btn_rec):
            w.setEnabled(True)
        self.sim.setEnabled(False)
        self.status.showMessage(f"Connected: {self.backend.name}")
        self._apply()

    def _apply(self):
        if not self.backend:
            return
        s = self.settings()
        try:
            act = self.backend.apply(s)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Apply failed", str(e))
            return
        self.actual = act
        self.status.showMessage(
            "Applied: f = {:.3f} Hz, amp = {:.4f} V, phase = {:.3f} deg, LPF = {} Hz, AC = {:g} Hz, qf = {:g}".format(
                act["frequency"], act["amplitude"], act["phase"], act["bandwidth"],
                act["acbandwidth"], act["quadrature_factor"]))
        if self.streaming and self.binner and abs(self.binner.rate - self.rate.value()) > 1e-9:
            self._stop_stream()
            self._start_stream()

    # ------------------------------------------------------------ streaming
    def _toggle_stream(self):
        if self.streaming:
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self):
        rate = self.rate.value()
        dec = choose_decimation(rate)
        fs = self.backend.start_stream(dec)
        self.binner = Binner(fs, rate)
        self.dec_label.setText(f"dec {dec} -> {fs/1e3:.3f} kS/s, buffer {2**14/fs:.2f} s")
        self._resize_buffer()
        self._clear_plots()
        self.samples_in = 0
        self.t_stream0 = time.perf_counter()
        self.streaming = True
        self.btn_start.setText("Stop")
        self.timer.start(POLL_MS)

    def _stop_stream(self):
        self.timer.stop()
        self.streaming = False
        self.backend.stop_stream()
        self.btn_start.setText("Start")

    def _resize_buffer(self):
        self.buf.resize(self.window.value() * self.rate.value())

    def _poll(self):
        try:
            raw = self.backend.read_new()
        except Exception as e:
            self._stop_stream()
            self.status.showMessage(f"Read error: {e}")
            return
        self.samples_in += raw.size
        t, z = self.binner.push(raw)
        if t.size:
            self.buf.extend(t, z)
            if self.rec_writer is not None:
                self._write_rows(t, z)
        self._update_plots()

    # ------------------------------------------------------------ display
    def _rotated(self, z):
        return z * np.exp(-1j * self.rotation)

    def _update_plots(self):
        if self.buf.t.size == 0:
            return
        t = self.buf.t
        zr = self._rotated(self.buf.z)
        qf = self.qf.value()
        k = self.iqgain.value()
        to_unit = 1 / np.sqrt(2) if self.units.currentText() == "V rms" else 1.0
        r_all = iq_to_volts(zr, qf, k) * to_unit
        ph_all = np.degrees(np.angle(zr))
        self.c_i.setData(t, zr.real)
        self.c_q.setData(t, zr.imag)
        self.c_r.setData(t, r_all)
        self.c_ph.setData(t, ph_all)
        if self.follow.isChecked():
            t1 = t[-1]
            t0 = max(t1 - self.window.value(), t[0])
            vis = t >= t0
            self.p_iq.setRange(xRange=(t0, t1), yRange=self._yrange(zr.real[vis], zr.imag[vis]), padding=0)
            self.p_r.setRange(xRange=(t0, t1), yRange=self._yrange(r_all[vis]), padding=0)
            self.p_ph.setRange(xRange=(t0, t1), yRange=self._yrange(ph_all[vis]), padding=0)
        tail = zr[-SCATTER_POINTS:]
        self.sc.setData(tail.real, tail.imag)
        self._update_phasor_range(tail)

        n1 = int(self.binner.rate)                   # last 1 s
        z1 = zr[-n1:]
        r = iq_to_volts(z1, qf, k)
        ph = np.degrees(np.angle(z1))
        elapsed = time.perf_counter() - self.t_stream0
        raw_max = max(np.abs(self.buf.z[-n1:].real).max(), np.abs(self.buf.z[-n1:].imag).max())
        warn = "  << NEAR FULL SCALE, lower qf" if raw_max > 0.9 else ""
        self.stats.setText(
            "Last 1 s ({} pts)\n"
            "I     = {:+.6f}  sd {:.2e}\n"
            "Q     = {:+.6f}  sd {:.2e}\n"
            "R     = {:.6f} V pk  ({:.6f} V rms)\n"
            "R sd  = {:.2e} V pk = {:.1f} ppm\n"
            "phase = {:+.4f} deg  sd {:.2f} mdeg\n"
            "rotation (SW) = {:+.3f} deg\n"
            "max |I|,|Q| raw = {:.3f}{}\n\n"
            "scope samples/s = {:.0f}\n"
            "output points   = {}\n"
            "buffer overruns = {}\n"
            "recording       = {}".format(
                z1.size, z1.real.mean(), z1.real.std(), z1.imag.mean(), z1.imag.std(),
                r.mean(), r.mean() / np.sqrt(2), r.std(), r.std() / max(r.mean(), 1e-15) * 1e6,
                ph.mean(), ph.std() * 1e3, np.degrees(self.rotation), raw_max, warn,
                self.samples_in / max(elapsed, 1e-9), int(self.binner.n * self.binner.rate / self.binner.fs),
                self.backend.overruns,
                f"{self.rec_rows} rows" if self.rec_writer else "off"))

    @staticmethod
    def _yrange(*arrays):
        """min/max of the visible data with 8 % margin; a minimum span so flat noise-free
        traces don't collapse to zero height."""
        v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (-1, 1)
        lo, hi = float(v.min()), float(v.max())
        span = max(hi - lo, 1e-9, abs(hi) * 1e-6)
        mid = 0.5 * (lo + hi)
        return (mid - 0.54 * span, mid + 0.54 * span)

    def _units_changed(self, text):
        self.p_r.setTitle(f"Amplitude R at input ({text})")
        self._update_plots()

    def _update_phasor_range(self, tail):
        """Change the phasor view only when needed (hysteresis), so it doesn't wobble."""
        if self.zoom_cluster.isChecked():
            c = complex(np.median(tail.real), np.median(tail.imag))
            need = float(np.percentile(np.abs(tail - c), 99.5))
            need = max(need, 1e-6)
        else:
            c = 0j
            need = max(float(np.abs(tail).max()), 1e-6)
        h = self._sc_half
        if (h is None or need > 0.9 * h or need < 0.25 * h
                or abs(c - self._sc_c) > 0.25 * h):
            self._sc_c = c
            self._sc_half = h = 2.0 * need
            self.p_sc.setRange(xRange=(c.real - h, c.real + h),
                               yRange=(c.imag - h, c.imag + h), padding=0)

    def _reset_phasor_range(self, *args):
        self._sc_c = self._sc_half = None
        self._update_plots()

    def _reset_view(self):
        if not self.follow.isChecked():
            self.follow.setChecked(True)      # re-enters here via the toggled signal
            return
        self._reset_phasor_range()

    def _clear_plots(self):
        self.buf.clear()
        self._sc_c = self._sc_half = None

    def _autophase(self):
        n = max(int(0.5 * (self.binner.rate if self.binner else 1000)), 1)
        if self.buf.z.size == 0:
            return
        self.rotation = float(np.angle(self.buf.z[-n:].mean()))
        self._reset_phasor_range()

    def _reset_rotation(self):
        self.rotation = 0.0
        self._reset_phasor_range()

    # ------------------------------------------------------------ recording
    def _toggle_record(self, on):
        if on:
            os.makedirs(DATA_DIR, exist_ok=True)
            default = os.path.join(os.path.abspath(DATA_DIR),
                                   dt.datetime.now().strftime("lockin_%Y%m%d_%H%M%S.csv"))
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Record to CSV", default, "CSV (*.csv)")
            if not path:
                self.btn_rec.setChecked(False)
                return
            self.rec_file = open(path, "w", newline="")
            s = asdict(self.settings())
            self.rec_file.write(f"# Red Pitaya lock-in recording, started {dt.datetime.now().isoformat()}\n")
            self.rec_file.write(f"# backend: {self.backend.name}\n")
            self.rec_file.write(f"# settings: {s}\n")
            self.rec_file.write(f"# actual: {getattr(self, 'actual', {})}\n")
            self.rec_file.write(f"# output rate: {self.rate.value()} Hz; software rotation at start: "
                                f"{np.degrees(self.rotation):.4f} deg\n")
            self.rec_file.write("# I_raw/Q_raw: IQ module units; I_rot/Q_rot: after software rotation; "
                                f"R_Vpk: |IQ|/(iq_gain*quadrature_factor), iq_gain={self.iqgain.value():.4f}; R_Vrms = R_Vpk/sqrt(2); rot_deg: software rotation in effect\n")
            self.rec_writer = csv.writer(self.rec_file)
            self.rec_writer.writerow(["t_s", "I_raw", "Q_raw", "I_rot", "Q_rot", "R_Vpk", "R_Vrms", "phase_deg", "rot_deg"])
            self.rec_rows = 0
            self.btn_rec.setText("Stop recording")
        else:
            if self.rec_file:
                self.rec_file.close()
                self.status.showMessage(f"Saved {self.rec_rows} rows to {self.rec_file.name}")
            self.rec_file = self.rec_writer = None
            self.btn_rec.setText("Record CSV")

    def _write_rows(self, t, z):
        zr = self._rotated(z)
        r = iq_to_volts(z, self.qf.value(), self.iqgain.value())
        ph = np.degrees(np.angle(zr))
        rot = f"{np.degrees(self.rotation):.5f}"
        self.rec_writer.writerows(
            (f"{a:.6f}", f"{b:.7e}", f"{c:.7e}", f"{d:.7e}", f"{e:.7e}", f"{g:.7e}", f"{g / np.sqrt(2):.7e}", f"{h:.5f}", rot)
            for a, b, c, d, e, g, h in zip(t, z.real, z.imag, zr.real, zr.imag, r, ph))
        self.rec_rows += t.size

    def closeEvent(self, ev):
        self.timer.stop()
        if self.rec_file:
            self.rec_file.close()
        if self.backend:
            self.backend.close()
        super().closeEvent(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="rp-f109f7.local")
    ap.add_argument("--simulate", action="store_true", help="use synthetic data, no board")
    args = ap.parse_args()
    w = LockinWindow(args)
    w.show()
    sys.exit(APP.exec_())


if __name__ == "__main__":
    main()
