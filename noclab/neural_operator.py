"""FNO-inspired Neural Operator for leakage prediction (pure numpy)."""

import numpy as np


class NeuralOperatorLeakageModel:
    """FNO-inspired spectral convolution model for key byte prediction.

    Architecture:
        - Input: (batch, time) power trace
        - SpectralConv layers: FFT → multiply by learnable weights → iFFT
        - Linear projection to 256 logits (one per key byte candidate)

    All operations are implemented in numpy only (no torch/tensorflow).
    Weights are randomly initialized (inference only — no training loop).
    """

    def __init__(self, n_modes: int = 16, hidden_dim: int = 32, n_layers: int = 2):
        self.n_modes = n_modes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        rng = np.random.default_rng(0)

        # Spectral conv weights: one complex weight matrix per layer
        # Shape: (n_layers, n_modes) complex — applied per mode
        self.spectral_weights = [
            rng.standard_normal((n_modes,)) + 1j * rng.standard_normal((n_modes,))
            for _ in range(n_layers)
        ]

        # After spectral convs, project flattened repr → hidden_dim → 256
        # We'll use hidden_dim as a simple linear layer size
        # Input after spectral conv: still (batch, time), we use mean pooling → scalar per layer
        # Then project (n_layers,) → hidden_dim → 256
        self.W1 = rng.standard_normal((n_layers, hidden_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, 256)) * 0.1
        self.b2 = np.zeros(256)

    def spectral_conv(self, x: np.ndarray) -> np.ndarray:
        """Apply spectral convolution to input.

        Args:
            x: Input array of shape (batch, time).

        Returns:
            Array of same shape (batch, time) after FFT → weight multiply → iFFT.
        """
        # FFT along time axis
        x_fft = np.fft.rfft(x, axis=-1)  # (batch, time//2+1)
        modes = min(self.n_modes, x_fft.shape[-1])
        # Apply first set of spectral weights (use layer 0 for this utility method)
        weights = self.spectral_weights[0][:modes]
        x_fft_out = np.zeros_like(x_fft)
        x_fft_out[:, :modes] = x_fft[:, :modes] * weights[np.newaxis, :]
        # iFFT back
        out = np.fft.irfft(x_fft_out, n=x.shape[-1], axis=-1)
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: power traces → key byte logits.

        Args:
            x: Input of shape (batch, time).

        Returns:
            Logits of shape (batch, 256).
        """
        batch_size = x.shape[0]
        # Apply each spectral conv layer and collect mean-pooled features
        features = np.zeros((batch_size, self.n_layers))
        current = x.copy()
        for i in range(self.n_layers):
            x_fft = np.fft.rfft(current, axis=-1)
            modes = min(self.n_modes, x_fft.shape[-1])
            weights = self.spectral_weights[i][:modes]
            x_fft_out = np.zeros_like(x_fft)
            x_fft_out[:, :modes] = x_fft[:, :modes] * weights[np.newaxis, :]
            current = np.fft.irfft(x_fft_out, n=current.shape[-1], axis=-1)
            features[:, i] = np.mean(np.abs(current), axis=-1)

        # MLP: (batch, n_layers) → (batch, hidden_dim) → (batch, 256)
        h = np.tanh(features @ self.W1 + self.b1)  # (batch, hidden_dim)
        logits = h @ self.W2 + self.b2  # (batch, 256)
        return logits
