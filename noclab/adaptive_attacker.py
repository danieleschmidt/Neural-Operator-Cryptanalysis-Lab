"""Adaptive attacker: uses a leakage model to select the best trace subset."""

import numpy as np
from .cpa import CorrelationPowerAnalysis


class AdaptiveAttacker:
    """Adaptive attack that selects the most informative traces using a leakage model.

    The model scores each trace; the top-n_select traces are passed to CPA.
    """

    def __init__(self, model, n_select: int = 50):
        """
        Args:
            model: NeuralOperatorLeakageModel (or any object with a .forward(x) method
                   returning (batch, 256) logits).
            n_select: Number of top traces to select for the CPA attack.
        """
        self.model = model
        self.n_select = n_select
        self._cpa = CorrelationPowerAnalysis()

    def select_traces(self, traces: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Select top-n_select trace indices by score.

        Args:
            traces: Power traces of shape (n_traces, n_samples).
            scores: Per-trace score array of shape (n_traces,).

        Returns:
            1-D array of indices (length n_select or n_traces if fewer available).
        """
        n = min(self.n_select, len(scores))
        return np.argsort(scores)[-n:]

    def attack(self, traces: np.ndarray, plaintexts: np.ndarray) -> int:
        """Run adaptive CPA attack.

        1. Score all traces with the model (use max logit as confidence).
        2. Select the top n_select traces.
        3. Run CPA on the subset.

        Args:
            traces: Power traces, shape (n_traces, n_samples).
            plaintexts: Plaintext byte values, shape (n_traces,).

        Returns:
            Recovered key byte (0-255).
        """
        logits = self.model.forward(traces)  # (n_traces, 256)
        scores = logits.max(axis=1)  # (n_traces,) — confidence score
        indices = self.select_traces(traces, scores)
        selected_traces = traces[indices]
        selected_plaintexts = np.asarray(plaintexts)[indices]
        return self._cpa.attack(selected_traces, selected_plaintexts)
