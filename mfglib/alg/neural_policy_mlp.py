from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import optuna
import torch
from torch import nn

from mfglib.alg.abc import Iterative
from mfglib.alg.q_fn import QFn
from mfglib.env import Environment
from mfglib.utils import mean_field_from_policy


@dataclass
class State:
    env: Environment
    pi: torch.Tensor
    net: nn.Module
    opt: torch.optim.Optimizer
    i: int


class NeuralPolicyMLP(Iterative[State]):
    """Neural Fictitious Play with a multilayer perceptron policy network.

    A multilayer perceptron maps a one-hot (optionally time-indexed) encoding
    of each state to a vector of action logits, with weights shared across all
    states and timesteps, in the spirit of :cite:t:`heinrich2016`. Each outer
    iteration (i) computes the optimal Q-values of the mean field induced by
    the running-average policy, (ii) fits the network to the resulting best
    response with ``n_inner`` Adam steps on the KL divergence, and (iii)
    updates the running-average policy fictitious-play style
    :cite:p:`perrin2020`.

    Designed for finite MFGs with large state spaces, where the shared
    network generalizes across states instead of learning one independent
    logit per state.

    Parameters
    ----------
    lr : float
        Learning rate of the Adam optimizer used in the inner fit.
    hidden_dim : int
        Width of each hidden layer.
    n_layers : int
        Number of hidden layers.
    time_encoding : bool
        If True, the network input is augmented with a one-hot encoding of
        the timestep, letting the policy depend non-stationarily on time.
    n_inner : int
        Number of Adam steps of the inner best-response fit per outer
        iteration.
    temp : float | None
        Softmax temperature used to smooth the best-response target. If
        None, the hard (argmax) best response is used.
    """

    def __init__(
        self,
        lr: float = 3e-3,
        hidden_dim: int = 64,
        n_layers: int = 2,
        time_encoding: bool = True,
        n_inner: int = 20,
        temp: float | None = None,
    ) -> None:
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("`lr` must be a positive float")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be a positive integer")
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("`n_layers` must be a positive integer")
        if not isinstance(n_inner, int) or n_inner <= 0:
            raise ValueError("`n_inner` must be a positive integer")
        if temp is not None and (not isinstance(temp, (int, float)) or temp <= 0):
            raise ValueError("`temp` must be None or a positive float")
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.time_encoding = time_encoding
        self.n_inner = n_inner
        self.temp = temp

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return (
            f"NeuralPolicyMLP(lr={self.lr}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}, time_encoding={self.time_encoding}, "
            f"n_inner={self.n_inner}, temp={self.temp})"
        )

    def _build_network(self, env: Environment) -> nn.Module:
        n_state_coords = env.n_states
        in_dim = n_state_coords + (env.T + 1 if self.time_encoding else 0)
        out_dim = env.n_actions
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(d, self.hidden_dim))
            layers.append(nn.ReLU())
            d = self.hidden_dim
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_inputs(env: Environment, time_encoding: bool) -> torch.Tensor:
        T, S = env.T, env.S
        grids = torch.meshgrid(
            [torch.arange(T + 1)] + [torch.arange(s) for s in S], indexing="ij"
        )
        one_hot_states = nn.functional.one_hot(
            grids[1].flatten(), num_classes=env.n_states
        ).float()
        if time_encoding:
            one_hot_time = nn.functional.one_hot(
                grids[0].flatten(), num_classes=T + 1
            ).float()
            inputs = torch.cat([one_hot_states, one_hot_time], dim=1)
        else:
            inputs = one_hot_states
        return inputs.reshape((T + 1) * env.n_states, -1)

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        torch.manual_seed(0)
        net = self._build_network(env)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        return State(env=env, pi=pi_0, net=net, opt=opt, i=0)

    def _best_response_target(self, Q: torch.Tensor) -> torch.Tensor:
        if self.temp is None:
            br = torch.zeros_like(Q)
            br.scatter_(-1, Q.argmax(dim=-1, keepdim=True), 1.0)
        else:
            br = (Q / self.temp).softmax(dim=-1)
        return br

    def step_next_state(
        self, state: State, atol: float | None, rtol: float | None
    ) -> State:
        pi_avg = state.pi
        L = mean_field_from_policy(pi_avg, env=state.env)
        Q = QFn(state.env, L, verify_integrity=False).optimal()
        br = self._best_response_target(Q)
        inputs = self._build_inputs(state.env, self.time_encoding)
        shape_prefix = (state.env.T + 1, *state.env.S)
        for _ in range(self.n_inner):
            logits = state.net(inputs).reshape(*shape_prefix, state.env.n_actions)
            log_probs = logits.log_softmax(dim=-1)
            loss = nn.functional.kl_div(log_probs, br, reduction="batchmean")
            state.opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            state.opt.step()
        with torch.no_grad():
            pi_net = (
                state.net(inputs)
                .reshape(*shape_prefix, state.env.n_actions)
                .softmax(dim=-1)
                .reshape(state.env.T + 1, *state.env.S, *state.env.A)
            )
        n = state.i + 1.0
        pi = (pi_avg * n + pi_net) / (n + 1.0)
        return State(env=state.env, pi=pi, net=state.net, opt=state.opt, i=state.i + 1)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "time_encoding": self.time_encoding,
            "n_inner": self.n_inner,
            "temp": self.temp,
        }

    def _init_tuner_instance(self: Self, trial: optuna.Trial) -> Self:
        return type(self)(
            lr=trial.suggest_float("lr", 1e-4, 1e0, log=True),
            hidden_dim=trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
            n_layers=trial.suggest_int("n_layers", 1, 4),
            time_encoding=trial.suggest_categorical("time_encoding", [True, False]),
            n_inner=trial.suggest_int("n_inner", 5, 50),
            temp=None,
        )

    def from_study(self, study: optuna.Study) -> Self:
        expected_params = {
            "lr",
            "hidden_dim",
            "n_layers",
            "time_encoding",
            "n_inner",
        }
        err_msg = f"{study.best_params.keys()=} != {expected_params}."
        assert study.best_params.keys() == expected_params, err_msg
        return type(self)(
            lr=study.best_params["lr"],
            hidden_dim=study.best_params["hidden_dim"],
            n_layers=study.best_params["n_layers"],
            time_encoding=study.best_params["time_encoding"],
            n_inner=study.best_params["n_inner"],
            temp=None,
        )
