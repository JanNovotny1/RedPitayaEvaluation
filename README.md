# Red Pitaya evaluation — signal generation & lock-in detection

Evaluation scripts for a Red Pitaya STEMlab 125-14 (OS 3.00, `rp-f109f7.local`)
as a test bench for the capacitive nanosensor's 1 MHz excitation and lock-in detection.

## Layout

| Path | What it does | Where the lock-in runs |
|---|---|---|
| `scpi_tests/rp_lockin_test.py` | 1 MHz sine on OUT1, captures IN1 + IN2 bursts (16 384 samples @ 125 MS/s) over SCPI, demodulates in numpy, prints amplitude, phase, IN2/IN1 ratio and their scatter | PC (post-processing of bursts) |
| `scpi_tests/rp_lockin_plot.py` | Same measurement, live window: raw IN1/IN2 waveforms, I/Q phasors, amplitude, phase and \|B/A\| history with running statistics. Options `--freq`, `--amp` | PC (post-processing of bursts) |
| `pyrpl_tests/pyrpl_lockin.py` | PyRPL IQ module: 1 MHz excitation on OUT1, demodulation of IN1 with a 500 Hz low-pass, reads I/Q | FPGA (continuous, real-time) |
| `lockin_gui/rp_lockin_gui.py` | **Lock-in GUI**: set frequency, amplitude, output/input, demod phase, low-pass, AC coupling, gain; gapless I/Q readout at 1 kHz (configurable), live plots, auto-phase, CSV recording. `--simulate` runs without a board | FPGA (PyRPL IQ) + PC binning |
| `tests/` | Unit tests for binning and simulator (`python -m pytest tests`) | — |
| `setup_env.ps1` | Creates the Python virtual environment and installs dependencies (Windows) | — |

## Setup (Windows PowerShell)

```powershell
cd C:\Users\jnovotny\Documents\RedPitayaEvaluation
.\setup_env.ps1            # Python 3.13 by default; -PythonVersion / -PythonExe to change
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the scripts, allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Wiring

- Input jumpers on **LV (±1 V)**.
- SCPI test: OUT1 → SMA T → IN1 (reference) and → IN2 (direct, or through the DUT).
- PyRPL test: OUT1 → IN1 (direct, or through the DUT).
- Generator *Load* setting: **Hi-Z** when driving the 1 MΩ inputs directly.

## 1. SCPI burst lock-in

1. Red Pitaya web UI → **Development → SCPI server → Run** (close other apps first).
2. Run:
   ```powershell
   python scpi_tests\rp_lockin_test.py rp-f109f7.local
   ```
3. Live plots:
   ```powershell
   python scpi_tests\rp_lockin_plot.py rp-f109f7.local            # add --freq 1000 --amp 0.3 etc.
   ```
   Close the window to stop (the generator is switched off).
4. Expected with a T-piece loopback: ~0.5 V pk on both channels, |B/A| ≈ 1.000, Δphase ≈ 0°.

Each result covers 131 carrier cycles (~131 µs → ~7.6 kHz equivalent bandwidth), so noise per reading
is higher than in a 500 Hz lock-in.

## 2. PyRPL FPGA lock-in

1. Close the Red Pitaya web apps. PyRPL connects over SSH and loads **its own FPGA bitstream**;
   reboot the board to get the stock apps back.
2. Run:
   ```powershell
   python pyrpl_tests\pyrpl_lockin.py rp-f109f7.local
   ```
   On first run PyRPL asks for the board's SSH password in its own prompt — don't store it in the code.
3. Adjust `phase=` in the script so the signal sits in I with Q ≈ 0.
4. Set `gui=True` for the interactive PyRPL window (iq2 tab, scope with `iq2` / `iq2_2`).

I/Q are in PyRPL internal units — use the direct loopback as the reference and compare the DUT against it.

## 3. Lock-in GUI (recommended)

```powershell
python lockin_gui\rp_lockin_gui.py             # hardware (close the Red Pitaya web apps first)
python lockin_gui\rp_lockin_gui.py --simulate  # try the GUI without a board
```

Connect → set parameters → **Apply settings** → **Start**. **Readout** selects *Stream 14-bit* (gapless, via the
scope buffer) or *High-res 24-bit* (full-precision filtered I/Q from the FPGA's 62-bit accumulators, one value per
averaging time, a few ms dead time between values; calibrated automatically against the stream on switching). **Auto-phase** rotates I/Q in software so the
signal lies on +I. **Record CSV** writes `t_s, I_raw, Q_raw, I_rot, Q_rot, R_V, phase_deg, rot_deg` to `data/`.

How the 1 kHz readout works: the IQ module's two quadratures go to the scope (ch1 = `iq2`, ch2 = `iq2_2`),
which runs in rolling mode at decimation 8192 (15.26 kS/s, 1.07 s circular buffer). Every 50 ms the GUI
reads only the newly written samples and averages them into exact 1 ms bins (15–16 samples each), so the
stream is gapless as long as the PC keeps up (the *buffer overruns* counter must stay 0).

See the full documentation for the signal chain, scaling and limitations.

## Known limitations / to verify

- Scripts were written against the documented Red Pitaya SCPI command set and the PyRPL API but have
  not yet been run on this board — if a command or parameter is rejected, the error names the line.
- PyRPL comes from Red Pitaya's official fork (github.com/RedPitaya/pyrpl), which supports Gen 2 and Original Gen
  boards on OS 3.00. The community version (pyrpl-fpga/pyrpl) supports Original Gen only.
- PyRPL needs SSH on the board. If `ssh root@<board>` is refused, see the documentation (missing host keys:
  run `ssh-keygen -A` and `systemctl enable --now ssh` on the board).
- Ratiometric IN2/IN1 with two PyRPL IQ modules needs a check that both NCOs stay phase-locked.
- `r.sampler` reads one value per network round-trip; use the PyRPL scope for continuous I/Q records.

## Next steps

- Insert the LC bandpass prototype as the DUT and record amplitude/phase at 1 MHz.
- Compare noise floor (SCPI burst vs PyRPL 500 Hz) with the nanosensor noise budget.
