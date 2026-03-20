"""Tests for noclab package."""

import numpy as np
import pytest
import sys
import os

# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from noclab.power_trace import PowerTraceSimulator
from noclab.neural_operator import NeuralOperatorLeakageModel
from noclab.cpa import CorrelationPowerAnalysis
from noclab.adaptive_attacker import AdaptiveAttacker
from noclab.demo import demo_recover_key_byte


# ---------------------------------------------------------------------------
# PowerTraceSimulator tests
# ---------------------------------------------------------------------------

def test_power_trace_shape():
    """PowerTraceSimulator produces correct shape."""
    sim = PowerTraceSimulator(noise_std=0.1, n_samples=500)
    pts = np.random.randint(0, 256, 100, dtype=np.uint8)
    traces = sim.simulate(pts, key_byte=0x2B, n_traces=80)
    assert traces.shape == (80, 500), f"Expected (80, 500), got {traces.shape}"


def test_power_trace_default_shape():
    """PowerTraceSimulator default shape (100, 1000)."""
    sim = PowerTraceSimulator()
    pts = np.arange(100, dtype=np.uint8)
    traces = sim.simulate(pts, key_byte=0x00)
    assert traces.shape == (100, 1000)


def test_power_trace_has_noise():
    """PowerTraceSimulator injects noise (traces are not identical)."""
    sim = PowerTraceSimulator(noise_std=0.5, n_samples=200)
    pts = np.full(50, 42, dtype=np.uint8)
    traces = sim.simulate(pts, key_byte=0x10, n_traces=50)
    # If all traces were identical the std across traces would be 0
    std_across_traces = traces.std(axis=0).mean()
    assert std_across_traces > 0.01, "Traces appear identical — noise not applied"


def test_power_trace_returns_ndarray():
    """simulate() returns a numpy ndarray."""
    sim = PowerTraceSimulator(n_samples=100)
    pts = np.arange(10, dtype=np.uint8)
    traces = sim.simulate(pts, key_byte=0xFF, n_traces=10)
    assert isinstance(traces, np.ndarray)


# ---------------------------------------------------------------------------
# NeuralOperatorLeakageModel tests
# ---------------------------------------------------------------------------

def test_neural_operator_forward_shape():
    """NeuralOperatorLeakageModel forward pass produces (batch, 256)."""
    model = NeuralOperatorLeakageModel(n_modes=8, hidden_dim=16, n_layers=2)
    x = np.random.randn(10, 200)
    out = model.forward(x)
    assert out.shape == (10, 256), f"Expected (10, 256), got {out.shape}"


def test_neural_operator_forward_batch_1():
    """Forward pass works with batch size 1."""
    model = NeuralOperatorLeakageModel()
    x = np.random.randn(1, 1000)
    out = model.forward(x)
    assert out.shape == (1, 256)


def test_spectral_conv_preserves_shape():
    """spectral_conv preserves spatial shape."""
    model = NeuralOperatorLeakageModel(n_modes=16)
    x = np.random.randn(5, 300)
    out = model.spectral_conv(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_neural_operator_output_range():
    """NeuralOperatorLeakageModel logits are finite (no NaN/Inf)."""
    model = NeuralOperatorLeakageModel()
    x = np.random.randn(20, 500)
    out = model.forward(x)
    assert np.all(np.isfinite(out)), "Output contains NaN or Inf"
    # Logits should span some range — at minimum 256 distinct values are possible
    assert out.shape[-1] == 256


# ---------------------------------------------------------------------------
# CorrelationPowerAnalysis tests
# ---------------------------------------------------------------------------

def test_cpa_recovers_correct_key():
    """CPA attack on clean traces recovers the correct key byte."""
    TRUE_KEY = 0x3C
    rng = np.random.default_rng(0)
    pts = rng.integers(0, 256, 300, dtype=np.uint8)
    sim = PowerTraceSimulator(noise_std=0.02, n_samples=300)
    traces = sim.simulate(pts, key_byte=TRUE_KEY, n_traces=300)
    cpa = CorrelationPowerAnalysis()
    recovered = cpa.attack(traces, pts)
    assert recovered == TRUE_KEY, f"CPA recovered {recovered:#04x}, expected {TRUE_KEY:#04x}"


def test_cpa_returns_valid_range():
    """CPA attack always returns a value in 0-255."""
    pts = np.arange(50, dtype=np.uint8)
    sim = PowerTraceSimulator(noise_std=1.0, n_samples=100)
    traces = sim.simulate(pts, key_byte=0xAA, n_traces=50)
    cpa = CorrelationPowerAnalysis()
    key = cpa.attack(traces, pts)
    assert 0 <= key <= 255


def test_cpa_custom_sbox():
    """CPA accepts a custom sbox."""
    custom_sbox = np.arange(256, dtype=np.uint8)
    cpa = CorrelationPowerAnalysis(sbox=custom_sbox)
    pts = np.arange(50, dtype=np.uint8)
    sim = PowerTraceSimulator(noise_std=0.5, n_samples=50)
    traces = sim.simulate(pts, key_byte=0x01, n_traces=50)
    key = cpa.attack(traces, pts)
    assert 0 <= key <= 255


# ---------------------------------------------------------------------------
# AdaptiveAttacker tests
# ---------------------------------------------------------------------------

def test_adaptive_attacker_selects_correct_count():
    """AdaptiveAttacker.select_traces returns exactly n_select indices."""
    model = NeuralOperatorLeakageModel(n_modes=4, hidden_dim=8, n_layers=1)
    attacker = AdaptiveAttacker(model, n_select=30)
    traces = np.random.randn(100, 200)
    scores = np.random.rand(100)
    indices = attacker.select_traces(traces, scores)
    assert len(indices) == 30


def test_adaptive_attacker_returns_valid_key():
    """AdaptiveAttacker.attack returns a valid key byte."""
    model = NeuralOperatorLeakageModel(n_modes=4, hidden_dim=8, n_layers=1)
    attacker = AdaptiveAttacker(model, n_select=40)
    pts = np.arange(80, dtype=np.uint8)
    sim = PowerTraceSimulator(noise_std=0.3, n_samples=100)
    traces = sim.simulate(pts, key_byte=0x55, n_traces=80)
    key = attacker.attack(traces, pts)
    assert 0 <= key <= 255


# ---------------------------------------------------------------------------
# Demo test
# ---------------------------------------------------------------------------

def test_demo_recover_key_byte_runs():
    """demo_recover_key_byte() runs without error."""
    demo_recover_key_byte()  # should not raise


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

def test_full_pipeline_key_in_range():
    """Full pipeline: simulate → CPA → verify key is in 0-255."""
    TRUE_KEY = 0x7F
    rng = np.random.default_rng(42)
    pts = rng.integers(0, 256, 150, dtype=np.uint8)
    sim = PowerTraceSimulator(noise_std=0.05, n_samples=200)
    traces = sim.simulate(pts, key_byte=TRUE_KEY, n_traces=150)
    cpa = CorrelationPowerAnalysis()
    key = cpa.attack(traces, pts)
    assert 0 <= key <= 255
