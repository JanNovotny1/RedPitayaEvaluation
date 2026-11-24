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

from backends import (HIRES_DEFAULT_SCALE, IQ_GAIN_DEFAULT, LockinSettings,  # noqa: E402
                      PyrplLockin, SimLockin, choose_decimation, iq_to_volts)
from binning import Binner, RawPass  # noqa: E402
from firfilter import FirResampler  # noqa: E402

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
        self.hires = False                  # readout mode
        self.hires_scale = None             # complex factor raw-24-bit -> stream units at qf=1
        self.hires_scale_src = "default"
        self.hires_conj = False
        self.n_points = 0
        self.dead_time = 0.0

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
        self.order.addItems(["1", "2", "3", "4"])
        self.order.setToolTip("Number of first-order low-pass stages in the FPGA (each 20 dB/decade)")
        self.order.setCurrentText("2")
        self.acbw = self._dspin(0, 1e6, 2429, 1, " Hz", 100)
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
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Stream 14-bit (gapless)", "High-res 24-bit (gaps)"])
        self.mode.setToolTip("Stream: 14-bit I/Q through the scope buffer, gapless at the output rate.\n"
                             "High-res: full-precision 24-bit filtered I/Q from the FPGA's 62-bit\n"
                             "accumulators, one value per averaging time, a few ms dead time between values.")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.t_avg = self._dspin(0.01, 10000, 1.0, 2, " ms", 1)
        self.t_avg.setToolTip("High-res mode: averaging time per value")
        self.t_avg.setEnabled(False)
        self.pcfilt = QtWidgets.QComboBox()
        self.pcfilt.addItems(["FIR (flat passband)", "Boxcar (average per point)", "None (raw scope samples)"])
        self.pcfilt.setToolTip("Stream mode: filter applied on the PC before reducing 15.26 kS/s to the output rate.\n"
                               "FIR: Kaiser-windowed, flat to 'FIR pass', >= 80 dB from 'FIR stop', zero phase.\n"
                               "Boxcar: plain average of the samples in each output interval.\n"
                               "None: every scope sample (15.26 kS/s at output rate 1 kHz), unfiltered on the PC.")
        self.fir_pass = self._dspin(1, 5000, 300, 1, " Hz", 10)
        self.fir_stop = self._dspin(2, 5000, 500, 1, " Hz", 10)
        self.fir_stop.setToolTip("Must be <= output rate / 2 (no aliasing into the output)")
        self.pcfilt.currentIndexChanged.connect(self._pcfilt_changed)
        f.addRow("Readout", self.mode)
        f.addRow("Averaging time", self.t_avg)
        f.addRow("Output rate", self.rate)
        f.addRow("PC filter", self.pcfilt)
        f.addRow("FIR pass", self.fir_pass)
        f.addRow("FIR stop", self.fir_stop)
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
            phase=self.phase.value(), bandwidth=float(self.bw.currentText().split()[0]),
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
        self.bw.addItems(["0 (off)"] + [f"{b:g}" for b in bws])
        target = min(bws, key=lambda b: abs(b - 2429))   # wide pre-filter; the PC FIR shapes the band
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
        if self.streaming and (self.hires or getattr(self, "_stream_cfg", None) != self._stream_config()):
            self._stop_stream()
            self._start_stream()

    # ------------------------------------------------------------ streaming
    def _toggle_stream(self):
        if self.streaming:
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self):
        self._resize_buffer()
        self.samples_in = 0
        self.n_points = 0
        self.dead_time = 0.0
        self.t_stream0 = time.perf_counter()
        was_hires = self.hires
        self.hires = self.mode.currentIndex() == 1
        if self.hires:
            self._prepare_hires_scale(from_stream=not was_hires)
            self.backend.start_hires(self.t_avg.value() * 1e-3)
            self.binner = None
            self.dec_label.setText(f"high-res: {self.t_avg.value():g} ms per value (24-bit)")
            interval = 1
        else:
            rate = self.rate.value()
            dec = choose_decimation(rate)
            fs = self.backend.start_stream(dec)
            info = f"dec {dec} -> {fs/1e3:.3f} kS/s, buffer {2**14/fs:.2f} s"
            self.filter_desc = "boxcar"
            if self.pcfilt.currentIndex() == 0:
                f_stop = min(self.fir_stop.value(), rate / 2)
                f_pass = min(self.fir_pass.value(), 0.9 * f_stop)
                if (f_stop, f_pass) != (self.fir_stop.value(), self.fir_pass.value()):
                    self.fir_stop.setValue(f_stop)
                    self.fir_pass.setValue(f_pass)
                    self.status.showMessage(f"FIR edges limited to the output Nyquist ({rate/2:g} Hz)")
                self.binner = FirResampler(fs, rate, f_pass, f_stop, 80.0)
                self.filter_desc = (f"FIR pass {f_pass:g} Hz, stop {f_stop:g} Hz (80 dB), "
                                    f"{self.binner.n_taps} taps, latency {self.binner.latency_s*1e3:.1f} ms")
                info += f"; FIR {self.binner.n_taps} taps, {self.binner.latency_s*1e3:.1f} ms delay"
            elif self.pcfilt.currentIndex() == 2:
                self.binner = RawPass(fs)
                self.filter_desc = f"none (raw scope samples at {fs:.3f} S/s)"
                info += "; raw samples, no PC filter"
                self.buf.resize(max(self.window.value() * fs, 10))
            else:
                self.binner = Binner(fs, rate)
            self._stream_cfg = self._stream_config()
            self.dec_label.setText(info)
            interval = POLL_MS
        self._clear_plots()
        self.streaming = True
        self.btn_start.setText("Stop")
        self.timer.start(interval)

    def _stop_stream(self):
        self.timer.stop()
        self.streaming = False
        self.backend.stop_stream()
        self.btn_start.setText("Start")

    def _resize_buffer(self):
        if self.mode.currentIndex() == 1:
            rate = 1e3 / self.t_avg.value()
        elif isinstance(self.binner, RawPass):
            rate = self.binner.fs
        else:
            rate = self.rate.value()
        self.buf.resize(max(self.window.value() * rate, 10))

    def _poll(self):
        if self.hires:
            self._poll_hires()
            return
        try:
            raw = self.backend.read_new()
        except Exception as e:
            self._stop_stream()
            self.status.showMessage(f"Read error: {e}")
            return
        self.samples_in += raw.size
        t, z = self.binner.push(raw)
        self.n_points += t.size
        if t.size:
            self.buf.extend(t, z)
            if self.rec_writer is not None:
                self._write_rows(t, z)
        self._update_plots()

    def _poll_hires(self):
        try:
            res = self.backend.poll_hires()
        except Exception as e:
            self._stop_stream()
            self.status.showMessage(f"Read error: {e}")
            return
        if res is None:
            return
        t_start, t_end, raw = res
        if self.hires_conj:
            raw = raw.conjugate()
        if self.hires_scale is None:            # first value: calibrate against the stream
            self._calibrate_hires(raw)
            if self.hires_conj:
                raw = raw.conjugate()
        z = raw * self.hires_scale * self.qf.value()
        t_avg = self.t_avg.value() * 1e-3
        t = np.array([t_start - self.t_stream0 + t_avg / 2])
        if self.n_points:
            self.dead_time += max(t_end - t_start - t_avg, 0.0)
        self.n_points += 1
        zz = np.array([z])
        self.buf.extend(t, zz)
        if self.rec_writer is not None:
            self._write_rows(t, zz)
        now = time.perf_counter()
        if now - getattr(self, "_last_draw", 0) > 0.05:   # redraw at most 20x per second
            self._last_draw = now
            self._update_plots()

    def _prepare_hires_scale(self, from_stream):
        """Remember the last second of 14-bit stream data (stream units) for calibration.
        Keeps an existing calibration when restarting high-res mode."""
        if not from_stream and self.hires_scale is not None:
            return
        self._cal_ref = None
        if self.buf.z.size and from_stream:
            sel = self.buf.t >= self.buf.t[-1] - 1.0
            self._cal_ref = (self.buf.z[sel].mean(), self.qf.value())
        self.hires_scale = None
        self.hires_conj = False

    def _calibrate_hires(self, raw):
        ref = getattr(self, "_cal_ref", None)
        self.hires_scale, self.hires_scale_src = HIRES_DEFAULT_SCALE, "default (PyRPL)"
        if ref is None or abs(ref[0]) < 1e-3 or abs(raw) == 0:
            return
        zs, qf = ref
        best = None
        for conj in (False, True):
            r = (zs / qf) / (raw.conjugate() if conj else raw)
            err = min(abs(np.angle(r)), abs(abs(np.angle(r)) - np.pi))
            if best is None or err < best[0]:
                best = (err, conj, r)
        err, conj, r = best
        if np.degrees(err) > 5:          # stream and accumulator disagree in phase: keep default
            self.hires_scale_src = f"default (cal. failed, {np.degrees(err):.1f} deg)"
            return
        scale = abs(r) * (1 if abs(np.angle(r)) < np.pi / 2 else -1)
        p2 = 2.0 ** np.round(np.log2(abs(scale)))
        if abs(abs(scale) / p2 - 1) < 0.03:          # snap to the nearest power of two
            scale = np.sign(scale) * p2
        self.hires_conj = conj
        self.hires_scale = scale
        self.hires_scale_src = f"cal. vs stream ({abs(scale) / HIRES_DEFAULT_SCALE:.4g} x default)"

    def _stream_config(self):
        return (self.rate.value(), self.pcfilt.currentIndex(), self.fir_pass.value(), self.fir_stop.value())

    def _pcfilt_changed(self, idx):
        self.fir_pass.setEnabled(idx == 0 and self.mode.currentIndex() == 0)
        self.fir_stop.setEnabled(idx == 0 and self.mode.currentIndex() == 0)

    def _mode_changed(self, idx):
        self.pcfilt.setEnabled(idx == 0)
        self._pcfilt_changed(self.pcfilt.currentIndex())
        self.t_avg.setEnabled(idx == 1)
        self.rate.setEnabled(idx == 0)
        if self.streaming:
            self._stop_stream()
            self._start_stream()

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

        sel1 = t >= t[-1] - 1.0                      # last 1 s
        z1 = zr[sel1]
        r = iq_to_volts(z1, qf, k)
        ph = np.degrees(np.angle(z1))
        elapsed = time.perf_counter() - self.t_stream0
        zb = self.buf.z[sel1]
        raw_max = max(np.abs(zb.real).max(), np.abs(zb.imag).max())
        warn = "  << NEAR FULL SCALE, lower qf" if (raw_max > 0.9 and not self.hires) else ""
        if self.hires:
            rate = self.n_points / max(elapsed, 1e-9)
            dead = self.dead_time / max(self.n_points - 1, 1) * 1e3
            mode_txt = ("mode            = high-res 24-bit\n"
                        f"values/s        = {rate:.1f}\n"
                        f"dead time/value = {dead:.2f} ms\n"
                        f"scale           = {self.hires_scale_src}")
        else:
            mode_txt = ("mode            = stream 14-bit\n"
                        f"PC filter       = {['FIR', 'boxcar', 'none (raw)'][self.pcfilt.currentIndex()]}\n"
                        f"scope samples/s = {self.samples_in / max(elapsed, 1e-9):.0f}\n"
                        f"buffer overruns = {self.backend.overruns}")
        self.stats.setText(
            "Last 1 s ({} pts)\n"
            "I     = {:+.6f}  sd {:.2e}\n"
            "Q     = {:+.6f}  sd {:.2e}\n"
            "R     = {:.6f} V pk  ({:.6f} V rms)\n"
            "R sd  = {:.2e} V pk = {:.1f} ppm\n"
            "phase = {:+.4f} deg  sd {:.2f} mdeg\n"
            "rotation (SW) = {:+.3f} deg\n"
            "max |I|,|Q| raw = {:.3f}{}\n\n"
            "{}\n"
            "output points   = {}\n"
            "recording       = {}".format(
                z1.size, z1.real.mean(), z1.real.std(), z1.imag.mean(), z1.imag.std(),
                r.mean(), r.mean() / np.sqrt(2), r.std(), r.std() / max(r.mean(), 1e-15) * 1e6,
                ph.mean(), ph.std() * 1e3, np.degrees(self.rotation), raw_max, warn,
                mode_txt, self.n_points,
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
        if self.buf.z.size == 0:
            return
        sel = self.buf.t >= self.buf.t[-1] - 0.5
        self.rotation = float(np.angle(self.buf.z[sel].mean()))
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
            self.rec_file.write(f"# readout: {self.mode.currentText()}"
                                + (f", averaging {self.t_avg.value()} ms, scale {self.hires_scale_src}" if self.hires
                                   else f", PC filter {getattr(self, 'filter_desc', '?')}")
                                + "\n")
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
