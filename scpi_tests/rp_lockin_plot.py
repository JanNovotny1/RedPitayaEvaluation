"""
Live visualisation of the SCPI burst lock-in (Red Pitaya STEMlab 125-14).

Same wiring and prerequisites as rp_lockin_test.py:
  OUT1 --T--> IN1 (reference) and --> IN2 (direct or through DUT), LV jumpers,
  SCPI server running on the board (web UI -> Development -> SCPI server).

Window shows, updated every capture:
  1. raw IN1 / IN2 waveforms (first few carrier cycles)
  2. I/Q phasor scatter of IN1 and IN2
  3. amplitude history (IN1, IN2)
  4. phase history (IN1, IN2, and B-A difference)
  5. |B/A| ratio history

Usage:  python scpi_tests\\rp_lockin_plot.py [host] [--freq HZ] [--amp V]
Close the window (or Ctrl+C) to stop; the generator is switched off on exit.
"""
import argparse
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from rp_lockin_test import FS, SCPI, acquire, lockin

HISTORY = 300          # number of captures kept in the history plots
SHOW_CYCLES = 5        # carrier cycles shown in the raw waveform plot


def setup_generator(rp, freq, amp):
    rp.tx("GEN:RST")
    rp.tx("SOUR1:FUNC SINE")
    rp.tx(f"SOUR1:FREQ:FIX {freq}")
    rp.tx(f"SOUR1:VOLT {amp}")
    rp.tx("OUTPUT1:STATE ON")
    rp.tx("SOUR1:TRig:INT")
    time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host", nargs="?", default="rp-f109f7.local")
    ap.add_argument("--freq", type=float, default=1e6, help="excitation frequency, Hz")
    ap.add_argument("--amp", type=float, default=0.5, help="excitation amplitude, V pk")
    args = ap.parse_args()

    rp = SCPI(args.host)
    print("Connected:", rp.q("*IDN?"))
    setup_generator(rp, args.freq, args.amp)

    n_show = int(SHOW_CYCLES * FS / args.freq)
    t_show = np.arange(n_show) / FS * 1e6          # µs

    hist = {k: deque(maxlen=HISTORY) for k in ("t", "aA", "aB", "pA", "pB", "dp", "r")}

    plt.ion()
    fig = plt.figure(figsize=(13, 8))
    fig.canvas.manager.set_window_title(f"Red Pitaya lock-in @ {args.freq/1e6:.3f} MHz")
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_iq = fig.add_subplot(gs[0, 1])
    ax_amp = fig.add_subplot(gs[1, 0])
    ax_ratio = fig.add_subplot(gs[1, 1])
    ax_ph = fig.add_subplot(gs[2, :])

    lA_raw, = ax_raw.plot([], [], label="IN1 (A)")
    lB_raw, = ax_raw.plot([], [], label="IN2 (B)")
    ax_raw.set(xlabel="time [µs]", ylabel="V", title="Raw waveforms")
    ax_raw.legend(loc="upper right")
    ax_raw.grid(True, alpha=0.3)

    sA_iq = ax_iq.scatter([], [], s=8, label="A")
    sB_iq = ax_iq.scatter([], [], s=8, label="B")
    ax_iq.set(xlabel="I [V]", ylabel="Q [V]", title="Phasors (I/Q)")
    ax_iq.set_aspect("equal", adjustable="box")
    ax_iq.axhline(0, color="k", lw=0.5)
    ax_iq.axvline(0, color="k", lw=0.5)
    ax_iq.legend(loc="upper right")
    ax_iq.grid(True, alpha=0.3)

    lA_amp, = ax_amp.plot([], [], label="|A|")
    lB_amp, = ax_amp.plot([], [], label="|B|")
    ax_amp.set(xlabel="time [s]", ylabel="amplitude [mV pk]", title="Amplitude")
    ax_amp.legend(loc="upper right")
    ax_amp.grid(True, alpha=0.3)

    l_ratio, = ax_ratio.plot([], [], color="C2")
    ax_ratio.set(xlabel="time [s]", ylabel="|B/A|", title="Ratio")
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.ticklabel_format(useOffset=False)

    lA_ph, = ax_ph.plot([], [], label="phase A")
    lB_ph, = ax_ph.plot([], [], label="phase B")
    l_dp, = ax_ph.plot([], [], label="B − A", color="C2", lw=2)
    ax_ph.set(xlabel="time [s]", ylabel="phase [deg]", title="Phase")
    ax_ph.legend(loc="upper right")
    ax_ph.grid(True, alpha=0.3)

    stats = fig.text(0.01, 0.005, "", family="monospace", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    zA_all, zB_all = deque(maxlen=HISTORY), deque(maxlen=HISTORY)
    t0 = time.time()
    try:
        while plt.fignum_exists(fig.number):
            a, b = acquire(rp)
            zA = lockin(a, args.freq)
            zB = lockin(b, args.freq)
            zA_all.append(zA)
            zB_all.append(zB)

            hist["t"].append(time.time() - t0)
            hist["aA"].append(abs(zA) * 1e3)
            hist["aB"].append(abs(zB) * 1e3)
            hist["pA"].append(np.degrees(np.angle(zA)))
            hist["pB"].append(np.degrees(np.angle(zB)))
            hist["dp"].append(np.degrees(np.angle(zB / zA)))
            hist["r"].append(abs(zB / zA))

            # raw waveforms
            lA_raw.set_data(t_show, a[:n_show])
            lB_raw.set_data(t_show, b[:n_show])
            ax_raw.relim(); ax_raw.autoscale_view()

            # phasors
            za, zb = np.array(zA_all), np.array(zB_all)
            sA_iq.set_offsets(np.c_[za.real, za.imag])
            sB_iq.set_offsets(np.c_[zb.real, zb.imag])
            lim = 1.1 * max(np.abs(za).max(), np.abs(zb).max(), 1e-3)
            ax_iq.set_xlim(-lim, lim); ax_iq.set_ylim(-lim, lim)

            # histories
            t = np.array(hist["t"])
            for line, key in ((lA_amp, "aA"), (lB_amp, "aB"), (l_ratio, "r"),
                              (lA_ph, "pA"), (lB_ph, "pB"), (l_dp, "dp")):
                line.set_data(t, np.array(hist[key]))
            for ax in (ax_amp, ax_ratio, ax_ph):
                ax.relim(); ax.autoscale_view()

            r = np.array(hist["r"]); dp = np.array(hist["dp"])
            stats.set_text(
                f"n={len(t):4d}   |A|={np.mean(hist['aA']):8.3f} mV (sd {np.std(hist['aA'])*1e3:6.1f} µV)   "
                f"|B|={np.mean(hist['aB']):8.3f} mV (sd {np.std(hist['aB'])*1e3:6.1f} µV)   "
                f"|B/A|={r.mean():.5f} (sd {r.std()/max(r.mean(), 1e-12)*1e6:6.1f} ppm)   "
                f"B−A={dp.mean():+.3f}° (sd {dp.std()*1e3:6.1f} mdeg)")

            fig.canvas.draw_idle()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        rp.tx("OUTPUT1:STATE OFF")
        rp.s.close()
        print("Generator off, connection closed.")


if __name__ == "__main__":
    main()
