# Red Pitaya evaluation — signal generation & lock-in detection

Evaluation scripts for a Red Pitaya STEMlab 125-14 (OS 3.00, `rp-f109f7.local`)
as a test bench for the capacitive nanosensor's 1 MHz excitation and lock-in detection.

## Layout

| Path | What it does | Where the lock-in runs |
|---|---|---|
| `scpi_tests/rp_lockin_test.py` | 1 MHz sine on OUT1, captures IN1 + IN2 bursts (16 384 samples @ 125 MS/s) over SCPI, demodulates in numpy, prints amplitude, phase, IN2/IN1 ratio and their scatter | PC (post-processing of bursts) |
| `pyrpl_tests/pyrpl_lockin.py` | PyRPL IQ module: 1 MHz excitation on OUT1, demodulation of IN1 with a 500 Hz low-pass, reads I/Q | FPGA (continuous, real-time) |
| `setup_env.ps1` | Creates the Python virtual environment and installs dependencies (Windows) | — |

## Setup (Windows PowerShell)

```powershell
cd C:\Users\jnovotny\Documents\RedPitayaEvaluation
.\setup_env.ps1            # creates .venv and installs requirements
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
3. Expected with a T-piece loopback: ~0.5 V pk on both channels, |B/A| ≈ 1.000, Δphase ≈ 0°.

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

## Known limitations / to verify

- Scripts were written against the documented Red Pitaya SCPI command set and the PyRPL API but have
  not yet been run on this board — if a command or parameter is rejected, the error names the line.
- PyRPL support for Red Pitaya OS 3 is new; use the GitHub development version (see `requirements.txt`).
- Ratiometric IN2/IN1 with two PyRPL IQ modules needs a check that both NCOs stay phase-locked.
- `r.sampler` reads one value per network round-trip; use the PyRPL scope for continuous I/Q records.

## Next steps

- Insert the LC bandpass prototype as the DUT and record amplitude/phase at 1 MHz.
- Compare noise floor (SCPI burst vs PyRPL 500 Hz) with the nanosensor noise budget.
