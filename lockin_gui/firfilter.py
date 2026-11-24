"""
Low-pass FIR + resampling of a uniformly sampled complex stream onto an exact output grid.

Kaiser-windowed sinc evaluated as a continuous kernel, so outputs land exactly on
t_k = k / rate even when fs / rate is not an integer (15258.79 S/s -> 1000 S/s).
Zero phase: each output is centred on its timestamp (latency = half the kernel length).
Weights are normalised per output, so the DC gain is exactly 1.

Same role as the FIR stage in the nanosensor design (flat passband, steep stopband
before decimation), but running on the PC on the 14-bit stream.
"""
import numpy as np


class FirResampler:
    def __init__(self, fs, rate, f_pass=300.0, f_stop=500.0, atten_db=80.0):
        if not 0 < f_pass < f_stop:
            raise ValueError("need 0 < f_pass < f_stop")
        if f_stop > rate / 2 + 1e-9:
            raise ValueError(f"stopband edge {f_stop} Hz must be <= output Nyquist {rate/2:g} Hz")
        self.fs, self.rate = float(fs), float(rate)
        self.f_pass, self.f_stop, self.atten_db = float(f_pass), float(f_stop), float(atten_db)
        self.fc = 0.5 * (f_pass + f_stop)                      # -6 dB point
        dw = 2 * np.pi * (f_stop - f_pass) / fs                # transition width [rad/sample]
        a = atten_db
        self.beta = 0.1102 * (a - 8.7) if a > 50 else (0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21) if a > 21 else 0.0)
        n_taps = int(np.ceil((a - 7.95) / (2.285 * dw))) + 1
        self.half = (n_taps + 1) // 2                          # half length [samples]
        self.T = self.half / fs                                # half length [s]
        self.n_taps = 2 * self.half + 1
        self.latency_s = self.T
        self._buf = np.empty(0, dtype=complex)
        self._buf_start = 0                                    # sample index of _buf[0]
        self._n_in = 0                                         # samples received
        self._k = 0                                            # next output index

    def _kernel(self, tau):
        """tau in seconds, array; continuous windowed sinc, 0 outside |tau| <= T"""
        x = np.clip(tau / self.T, -1.0, 1.0)
        w = np.i0(self.beta * np.sqrt(1.0 - x * x)) / np.i0(self.beta)
        h = np.sinc(2 * self.fc * tau) * w
        h[np.abs(tau) > self.T] = 0.0
        return h

    def push(self, z):
        z = np.asarray(z, dtype=complex)
        self._buf = np.concatenate([self._buf, z])
        self._n_in += z.size
        # outputs t_k whose kernel window [t_k - T, t_k + T] is fully received
        t_last = (self._n_in - 1) / self.fs
        k_max = int(np.floor((t_last - self.T) * self.rate))
        if k_max < self._k:
            return np.empty(0), np.empty(0, dtype=complex)
        ks = np.arange(self._k, k_max + 1)
        t_out = ks / self.rate
        # sample indices around each output: n0 .. n0 + L - 1
        L = 2 * self.half + 3
        n0 = np.ceil((t_out - self.T) * self.fs).astype(np.int64) - 1
        idx = n0[:, None] + np.arange(L)[None, :]
        valid = (idx >= 0) & (idx < self._n_in)
        tau = t_out[:, None] - idx / self.fs
        h = np.where(valid, self._kernel(tau.ravel()).reshape(tau.shape), 0.0)
        local = np.clip(idx - self._buf_start, 0, self._buf.size - 1)
        x = self._buf[local]
        y = (h * x).sum(axis=1) / h.sum(axis=1)
        self._k = k_max + 1
        # drop samples no longer needed by future outputs
        t_next = self._k / self.rate
        keep_from = max(int(np.floor((t_next - self.T) * self.fs)) - 2, self._buf_start)
        cut = keep_from - self._buf_start
        if cut > 0:
            self._buf = self._buf[cut:]
            self._buf_start = keep_from
        return t_out, y

    def response(self, f):
        """amplitude response of the (centred) kernel at frequencies f [Hz]"""
        n = np.arange(-self.half, self.half + 1)
        h = self._kernel(n / self.fs)
        h = h / h.sum()
        f = np.atleast_1d(np.asarray(f, dtype=float))
        return np.abs(np.exp(-2j * np.pi * np.outer(f, n) / self.fs) @ h)
