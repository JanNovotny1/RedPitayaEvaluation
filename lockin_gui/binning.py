"""Boxcar binning of a uniformly sampled complex stream onto an exact output-rate grid."""
import numpy as np


class Binner:
    """
    Averages a continuous complex sample stream (rate fs) into bins of exactly
    1/rate seconds. Sample k belongs to bin floor(k * rate / fs); a bin is only
    emitted once a sample of the next bin has arrived, so every emitted value is
    the mean of a complete bin (15 or 16 samples for fs = 15.26 kS/s, rate = 1 kHz).

    push() returns (t, z): bin centre times [s] since stream start and bin means.
    """

    def __init__(self, fs, rate):
        if rate > fs:
            raise ValueError(f"output rate {rate} Hz exceeds sample rate {fs:.1f} Hz")
        self.fs = float(fs)
        self.rate = float(rate)
        self.n = 0                                   # index of next incoming sample
        self._carry = np.empty(0, dtype=complex)     # samples of the unfinished bin
        self._carry_bin = None

    def push(self, z):
        z = np.asarray(z, dtype=complex)
        if z.size == 0:
            return np.empty(0), np.empty(0, dtype=complex)
        idx = self.n + np.arange(z.size)
        self.n += z.size
        bins = np.floor(idx * self.rate / self.fs).astype(np.int64)

        if self._carry.size:
            z = np.concatenate([self._carry, z])
            bins = np.concatenate([np.full(self._carry.size, self._carry_bin), bins])

        last = bins[-1]
        done = bins < last
        self._carry = z[~done]
        self._carry_bin = last
        if not done.any():
            return np.empty(0), np.empty(0, dtype=complex)

        b = bins[done]
        b0 = b[0]
        rel = b - b0
        counts = np.bincount(rel)
        sums_re = np.bincount(rel, weights=z[done].real)
        sums_im = np.bincount(rel, weights=z[done].imag)
        ok = counts > 0
        means = (sums_re[ok] + 1j * sums_im[ok]) / counts[ok]
        t = (np.nonzero(ok)[0] + b0 + 0.5) / self.rate
        return t, means

    def reset(self):
        self.__init__(self.fs, self.rate)


class RawPass:
    """No PC filtering: passes every scope sample through with its exact timestamp n / fs."""

    def __init__(self, fs):
        self.fs = float(fs)
        self.rate = self.fs
        self.n = 0

    def push(self, z):
        z = np.asarray(z, dtype=complex)
        t = (self.n + np.arange(z.size)) / self.fs
        self.n += z.size
        return t, z
