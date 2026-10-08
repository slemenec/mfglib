"""Benchmark of neural algorithms against MFGLib baselines on Beach Bar.

Run with::

    python benchmarks/benchmark_neural_fictitious_play.py

For each environment x algorithm pair, the solver is run for ``MAX_ITER``
iterations without early stopping and the exploitability trajectory is
reported. When matplotlib is available, a plot is saved under ``images/``.
"""

from __future__ import annotations

import time

import torch

from mfglib.alg import (
    MFOMO,
    FictitiousPlay,
    NeuralFictitiousPlay,
    NeuralPolicyMLP,
    NeuralTwoNetworks,
    OnlineMirrorDescent,
    PriorDescent,
)
from mfglib.env import Environment
from mfglib.scoring import exploitability_score

MAX_ITER = 300
SEED = 0


def main() -> None:
    """Run the Beach Bar benchmark and print/save results."""
    torch.manual_seed(SEED)

    envs = {
        "beach_bar(T=2, n=4)": Environment.beach_bar(),
        "beach_bar(T=5, n=10)": Environment.beach_bar(T=5, n=10),
        "beach_bar(T=10, n=20)": Environment.beach_bar(T=10, n=20),
    }

    algs = {
        "NeuralPolicyMLP()": NeuralPolicyMLP(),
        "NeuralTwoNetworks()": NeuralTwoNetworks(),
        "NeuralFictitiousPlay(lr=1.0)": NeuralFictitiousPlay(lr=1.0),
        "FictitiousPlay": FictitiousPlay(),
        "OnlineMirrorDescent(alpha=10.0)": OnlineMirrorDescent(alpha=10.0),
        "PriorDescent": PriorDescent(),
        "MFOMO": MFOMO(),
    }

    trajectories = {}

    for env_name, env in envs.items():
        for alg_name, alg in algs.items():
            t_0 = time.time()
            solns, _, runtimes = alg.solve(env, max_iter=MAX_ITER, atol=None, rtol=None)
            elapsed = time.time() - t_0
            expls = exploitability_score(env, solns)
            trajectories[(env_name, alg_name)] = expls
            print(
                f"{env_name} | {alg_name} | "
                f"expl_final={expls[-1]:.3e} expl_min={min(expls):.3e} "
                f"runtime={elapsed:.2f}s"
            )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        return

    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 5))
    for ax, (env_name, _) in zip(axes, envs.items()):
        env = envs[env_name]
        for alg_name, alg in algs.items():
            solns, _, runtimes = alg.solve(env, max_iter=MAX_ITER, atol=None, rtol=None)
            expls = exploitability_score(env, solns)
            ax.semilogy(runtimes[1:], expls[1:], label=alg_name)
        ax.set_xlabel("Runtime (seconds)")
        ax.set_ylabel("Exploitability")
        ax.set_title(env_name)
        ax.grid(True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("images/benchmark_neural_algorithms.png", dpi=150)
    print("Plot saved to images/benchmark_neural_algorithms.png")


if __name__ == "__main__":
    main()
