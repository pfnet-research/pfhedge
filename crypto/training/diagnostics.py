"""Diagnostic utilities for debugging deep hedging models.

This module provides tools to inspect MLP inputs, outputs, and gradients
during training and inference to diagnose issues like zero-hedge outputs.
"""

import torch
import numpy as np
from typing import Optional, Dict, List
from collections import defaultdict


class MLPDiagnostics:
    """Diagnostic tool to track MLP inputs/outputs during training and inference.

    Usage:
        >>> from crypto.training.diagnostics import MLPDiagnostics
        >>> diagnostics = MLPDiagnostics(model)
        >>> diagnostics.attach()
        >>>
        >>> # Train model...
        >>> model.fit(option, ...)
        >>>
        >>> # Print statistics
        >>> diagnostics.print_summary()
        >>> diagnostics.detach()
    """

    def __init__(self, hedger, sample_frequency: int = 5):
        """Initialize diagnostics.

        Args:
            hedger: PFHedge Hedger instance
            sample_frequency: Sample every N forward passes (to reduce overhead)
        """
        self.hedger = hedger
        self.model = hedger.model
        self.sample_frequency = sample_frequency

        # Storage for statistics
        self.call_count = 0
        self.input_stats = defaultdict(list)
        self.output_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)

        # Hooks
        self.hooks = []

    def attach(self):
        """Attach hooks to model to capture inputs/outputs."""

        # Hook for model inputs/outputs
        def forward_hook(_module, input, output):
            self.call_count += 1

            # Sample to reduce overhead
            if self.call_count % self.sample_frequency != 0:
                return

            # Input tensor (batch of features)
            if len(input) > 0:
                x = input[0]
                self.input_stats["shape"].append(x.shape)
                self.input_stats["mean"].append(x.mean().item())
                self.input_stats["std"].append(x.std().item())
                self.input_stats["min"].append(x.min().item())
                self.input_stats["max"].append(x.max().item())
                self.input_stats["has_nan"].append(torch.isnan(x).any().item())
                self.input_stats["has_inf"].append(torch.isinf(x).any().item())

                # Track per-feature statistics (if batch)
                if x.dim() == 2:  # (batch, features)
                    for i in range(x.shape[1]):
                        self.input_stats[f"feature_{i}_mean"].append(
                            x[:, i].mean().item()
                        )
                        self.input_stats[f"feature_{i}_std"].append(
                            x[:, i].std().item()
                        )

            # Output tensor (hedge ratios)
            if output is not None:
                self.output_stats["shape"].append(output.shape)
                self.output_stats["mean"].append(output.mean().item())
                self.output_stats["std"].append(output.std().item())
                self.output_stats["min"].append(output.min().item())
                self.output_stats["max"].append(output.max().item())
                self.output_stats["abs_mean"].append(output.abs().mean().item())
                self.output_stats["has_nan"].append(torch.isnan(output).any().item())
                self.output_stats["has_inf"].append(torch.isinf(output).any().item())

        # Hook for gradients (during backward pass)
        def backward_hook(_module, _grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0]
                self.gradient_stats["mean"].append(grad.mean().item())
                self.gradient_stats["std"].append(grad.std().item())
                self.gradient_stats["norm"].append(grad.norm().item())
                self.gradient_stats["has_nan"].append(torch.isnan(grad).any().item())

        # Attach hooks
        h1 = self.model.register_forward_hook(forward_hook)
        h2 = self.model.register_full_backward_hook(backward_hook)
        self.hooks = [h1, h2]

        print("✅ Diagnostics attached to model")

    def detach(self):
        """Remove hooks from model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("✅ Diagnostics detached from model")

    def print_summary(self, verbose: bool = True):
        """Print summary statistics of inputs/outputs.

        Args:
            verbose: If True, print per-feature statistics
        """
        print("\n" + "=" * 70)
        print("MLP DIAGNOSTICS SUMMARY")
        print("=" * 70)

        print(f"\n📊 Forward passes sampled: {len(self.output_stats['mean'])}")
        print(f"   Total forward calls: {self.call_count}")
        print(f"   Sampling frequency: 1/{self.sample_frequency}")

        # Input statistics
        if self.input_stats["mean"]:
            print("\n📥 INPUT STATISTICS:")
            print(
                f"   Shape: {self.input_stats['shape'][-1] if self.input_stats['shape'] else 'N/A'}"
            )
            print(
                f"   Mean: {np.mean(self.input_stats['mean']):.6f} ± {np.std(self.input_stats['mean']):.6f}"
            )
            print(
                f"   Std:  {np.mean(self.input_stats['std']):.6f} ± {np.std(self.input_stats['std']):.6f}"
            )
            print(
                f"   Range: [{np.mean(self.input_stats['min']):.6f}, {np.mean(self.input_stats['max']):.6f}]"
            )
            print(f"   Has NaN: {any(self.input_stats['has_nan'])}")
            print(f"   Has Inf: {any(self.input_stats['has_inf'])}")

            # Per-feature stats
            if verbose:
                feature_keys = [
                    k
                    for k in self.input_stats.keys()
                    if k.startswith("feature_") and k.endswith("_mean")
                ]
                if feature_keys:
                    print("\n   Per-feature statistics:")
                    for i in range(len(feature_keys)):
                        feat_mean = np.mean(self.input_stats[f"feature_{i}_mean"])
                        feat_std = np.mean(self.input_stats[f"feature_{i}_std"])
                        print(
                            f"     Feature {i}: mean={feat_mean:+.6f}, std={feat_std:.6f}"
                        )

        # Output statistics
        if self.output_stats["mean"]:
            print("\n📤 OUTPUT STATISTICS (Hedge Ratios):")
            print(
                f"   Shape: {self.output_stats['shape'][-1] if self.output_stats['shape'] else 'N/A'}"
            )
            print(
                f"   Mean: {np.mean(self.output_stats['mean']):.6f} ± {np.std(self.output_stats['mean']):.6f}"
            )
            print(
                f"   Std:  {np.mean(self.output_stats['std']):.6f} ± {np.std(self.output_stats['std']):.6f}"
            )
            print(
                f"   Abs mean: {np.mean(self.output_stats['abs_mean']):.6f} (avg |hedge|)"
            )
            print(
                f"   Range: [{np.mean(self.output_stats['min']):.6f}, {np.mean(self.output_stats['max']):.6f}]"
            )
            print(f"   Has NaN: {any(self.output_stats['has_nan'])}")
            print(f"   Has Inf: {any(self.output_stats['has_inf'])}")

            # Check for near-zero outputs
            abs_mean = np.mean(self.output_stats["abs_mean"])
            if abs_mean < 0.01:
                print(
                    f"\n   ⚠️  WARNING: Average |hedge| = {abs_mean:.6f} is very small!"
                )
                print(f"      This suggests the model is outputting near-zero hedges.")

        # Gradient statistics
        if self.gradient_stats["mean"]:
            print("\n🔄 GRADIENT STATISTICS:")
            print(
                f"   Mean: {np.mean(self.gradient_stats['mean']):.6f} ± {np.std(self.gradient_stats['mean']):.6f}"
            )
            print(f"   Std:  {np.mean(self.gradient_stats['std']):.6f}")
            print(
                f"   Norm: {np.mean(self.gradient_stats['norm']):.6f} ± {np.std(self.gradient_stats['norm']):.6f}"
            )
            print(f"   Has NaN: {any(self.gradient_stats['has_nan'])}")

            # Check for vanishing gradients
            grad_norm = np.mean(self.gradient_stats["norm"])
            if grad_norm < 1e-6:
                print(
                    f"\n   ⚠️  WARNING: Gradient norm = {grad_norm:.2e} is very small!"
                )
                print(f"      This suggests vanishing gradients.")
            elif grad_norm > 1e3:
                print(
                    f"\n   ⚠️  WARNING: Gradient norm = {grad_norm:.2e} is very large!"
                )
                print(f"      This suggests exploding gradients.")

        print("\n" + "=" * 70 + "\n")

    def reset(self):
        """Reset all collected statistics."""
        self.call_count = 0
        self.input_stats.clear()
        self.output_stats.clear()
        self.gradient_stats.clear()

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics as dictionary.

        Returns:
            Dictionary with 'input', 'output', 'gradient' keys containing stats
        """
        return {
            "input": {
                "mean": (
                    np.mean(self.input_stats["mean"])
                    if self.input_stats["mean"]
                    else None
                ),
                "std": (
                    np.mean(self.input_stats["std"])
                    if self.input_stats["std"]
                    else None
                ),
                "min": (
                    np.mean(self.input_stats["min"])
                    if self.input_stats["min"]
                    else None
                ),
                "max": (
                    np.mean(self.input_stats["max"])
                    if self.input_stats["max"]
                    else None
                ),
            },
            "output": {
                "mean": (
                    np.mean(self.output_stats["mean"])
                    if self.output_stats["mean"]
                    else None
                ),
                "std": (
                    np.mean(self.output_stats["std"])
                    if self.output_stats["std"]
                    else None
                ),
                "abs_mean": (
                    np.mean(self.output_stats["abs_mean"])
                    if self.output_stats["abs_mean"]
                    else None
                ),
                "min": (
                    np.mean(self.output_stats["min"])
                    if self.output_stats["min"]
                    else None
                ),
                "max": (
                    np.mean(self.output_stats["max"])
                    if self.output_stats["max"]
                    else None
                ),
            },
            "gradient": {
                "mean": (
                    np.mean(self.gradient_stats["mean"])
                    if self.gradient_stats["mean"]
                    else None
                ),
                "std": (
                    np.mean(self.gradient_stats["std"])
                    if self.gradient_stats["std"]
                    else None
                ),
                "norm": (
                    np.mean(self.gradient_stats["norm"])
                    if self.gradient_stats["norm"]
                    else None
                ),
            },
        }


def diagnose_hedger(hedger, option, n_paths: int = 1000):
    """Quick diagnostic helper function.

    Args:
        hedger: Trained PFHedge Hedger
        option: Option to evaluate on
        n_paths: Number of paths to simulate

    Returns:
        Dictionary with diagnostic statistics
    """
    diagnostics = MLPDiagnostics(hedger, sample_frequency=1)
    diagnostics.attach()

    # Run inference
    with torch.no_grad():
        hedges = hedger.compute_hedge(option)

    # Print results
    diagnostics.print_summary(verbose=True)

    # Get stats
    stats = diagnostics.get_stats()

    # Add hedge statistics
    stats["hedges"] = {
        "mean": hedges.mean().item(),
        "std": hedges.std().item(),
        "abs_mean": hedges.abs().mean().item(),
        "min": hedges.min().item(),
        "max": hedges.max().item(),
    }

    diagnostics.detach()

    return stats
