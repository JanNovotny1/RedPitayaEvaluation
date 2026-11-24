"""Tests for the lock-in GUI core (no hardware, no display needed).
Run:  python -m pytest tests
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lockin_gui"))

from backends import IQ_GAIN_DEFAULT, SimLockin, choose_decimation  # noqa: E402
from binning import Binner  # noqa: E402

FS_1K = 125e6 / 8192


def test_choose_decimation():
    assert choose_decimation(1000) == 8192
    assert choose_decimation(100) == 65536
    dec = choose_decimation(10000)
    assert 125e6 / dec >= 10000 and 2**14 * dec / 125e6 >= 0.4


def test_binner_exact_rate_and_means_chunked():
    rng = np.random.default_rng(1)
    n = int(FS_1K * 3)
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b = Binner(FS_1K, 1000)
    ts, zs = [], []
    for chunk in np.array_split(z, 137):          # irregular chunk sizes
        t, m = b.push(chunk)
        ts.append(t)
        zs.append(m)
    t = np.concatenate(ts)
    m = np.concatenate(zs)
    # consecutive bins, exactly 1 ms apart, no gaps
    assert np.allclose(np.diff(t), 1e-3)
    # reference: direct binning of the whole record
    bins = np.floor(np.arange(n) * 1000 / FS_1K).astype(int)
    for k in (0, 1, 500, len(t) - 1):
        assert np.isclose(m[k], z[bins == k].mean())
    counts = np.bincount(bins)[:-1]
    assert set(counts) <= {15, 16}


def test_binner_constant_signal():
    b = Binner(FS_1K, 1000)
    t, m = b.push(np.full(20000, 0.25 + 0.1j))
    assert np.allclose(m, 0.25 + 0.1j)


def test_sim_backend_streams_at_fs():
    s = SimLockin()
    s.connect()
    fs = s.start_stream(8192)
    import time
    time.sleep(0.3)
    z = s.read_new()
    assert abs(z.size - 0.3 * fs) < 0.1 * fs
    assert np.isclose(np.abs(z.mean()), IQ_GAIN_DEFAULT * 2 * 0.98 * 0.25, rtol=0.01)


def test_signed_62bit_decoding():
    from backends import _signed
    assert _signed(5, 62) == 5
    assert _signed((1 << 62) - 1, 62) == -1
    v = -123456789012
    assert _signed(v & ((1 << 62) - 1), 62) == v


def test_sim_hires_returns_raw_units():
    import time
    from backends import HIRES_DEFAULT_SCALE, IQ_GAIN_DEFAULT
    s = SimLockin()
    s.connect()
    s.start_hires(1e-3)
    res = None
    t0 = time.time()
    while res is None and time.time() - t0 < 1:
        res = s.poll_hires()
    assert res is not None
    z = res[2] * HIRES_DEFAULT_SCALE
    assert np.isclose(abs(z), IQ_GAIN_DEFAULT * 2 * 0.98 * 0.25, rtol=0.02)
