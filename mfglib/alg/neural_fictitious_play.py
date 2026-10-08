from __future__ import annotations

from dataclasses import dataclass

import optuna
import torch

from mfglib.alg.abc import Iterative
from mfglib.alg.q_fn import QFn
from mfglib.env import Environment
from mfglib.utils import mean_field_from_policy


@dataclass
class State:
    env: Environment
    pi: torch.Tensor
    y: torch.Tensor
    i: int
    opt: torch.optim.Optimizer


class NeuralFictitiousPlay(Iterative[State]):
    """Neural Fictitious Play algorithm.

    A PyTorch-based, fully differentiable variant of fictitious play. A
    parameter tensor of logits over actions is optimized with Adam so as to
    maximize the expected return against the mean field induced by the
    running-average policy, while the policy estimate itself is tracked as
    the uniform average of the softmax policies, in the spirit of
    :cite:t:`perrin2020` and :cite:t`heinrich2016`.

    Attributes
    ----------
    lr
        Learning rate of the Adam optimizer applied to the logits.
    tau
        Softmax temperature applied to the logits.
    avg
        If True, track the uniform average of past policies (fictitious-play
        style). If False, use the last softmax policy.
    """

    def __init__(self, lr: float = 1.0, tau: float = 1.0, avg: bool = False) -> None:
        """Neural Fictitious Play algorithm.

        Attributes
        ----------
        lr
            Learning rate of the Adam optimizer applied to the logits.
        tau
            Softmax temperature applied to the logits.
        avg
            If True, track the uniform average of past policies.
        """
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("`lr` must be a positive float")
        if not isinstance(tau, (int, float)) or tau <= 0:
            raise ValueError("`tau` must be a positive float")
        self.lr = lr
        self.tau = tau
        self.avg = avg

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"NeuralFictitiousPlay(lr={self.lr}, tau={self.tau}, avg={self.avg})"

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        n_action_coords = 1
        for a in env.A:
            n_action_coords *= a
        y = pi_0.clone().flatten(start_dim=1 + len(env.S))
        y = y.expand(env.T + 1, *env.S, n_action_coords).clone()
        y = (y / self.tau).log()
        y = y - y.mean(dim=-1, keepdim=True)
        y.requires_grad_(True)
        opt = torch.optim.Adam([y], lr=self.lr)
        return State(env=env, pi=pi_0, y=y, i=0, opt=opt)

    def step_next_state(
        self, state: State, atol: float | None, rtol: float | None
    ) -> State:
        pi_avg = state.pi
        L = mean_field_from_policy(pi_avg, env=state.env)
        Q = QFn(state.env, L, verify_integrity=False).optimal()
        softmax = torch.nn.Softmax(dim=-1)
        pi_last = softmax(state.y / self.tau)
        Q_flat = Q.flatten(start_dim=1 + len(state.env.S)).detach()
        adv = Q_flat - (Q_flat * pi_last).sum(dim=-1, keepdim=True)
        grad = -(pi_last * adv)
        state.opt.zero_grad()
        state.y.grad = grad
        state.opt.step()
        pi_last = softmax(state.y.detach() / self.tau).reshape(
            state.env.T + 1, *state.env.S, *state.env.A
        )
        if self.avg:
            n = state.i + 1.0
            pi = (pi_avg * n + pi_last) / (n + 1.0)
        else:
            pi = pi_last
        return State(env=state.env, pi=pi, y=state.y, i=state.i + 1, opt=state.opt)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"lr": self.lr, "tau": self.tau, "avg": self.avg}

    def _init_tuner_instance(self, trial: optuna.Trial) -> "NeuralFictitiousPlay":
        return type(self)(
            lr=trial.suggest_float("lr", 1e-3, 1e3, log=True),
            tau=trial.suggest_float("tau", 1e-3, 1e3, log=True),
            avg=trial.suggest_categorical("avg", [True, False]),
        )

    def from_study(self, study: optuna.Study) -> "NeuralFictitiousPlay":
        expected_params = {"lr", "tau", "avg"}
        err_msg = f"{study.best_params.keys()=} != {expected_params}."
        assert study.best_params.keys() == expected_params, err_msg
        return type(self)(
            lr=study.best_params["lr"],
            tau=study.best_params["tau"],
            avg=study.best_params["avg"],
        )
