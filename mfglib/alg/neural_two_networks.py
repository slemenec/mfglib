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
    policy_net: nn.Module
    density_net: nn.Module
    policy_opt: torch.optim.Optimizer
    density_opt: torch.optim.Optimizer
    i: int


class NeuralTwoNetworks(Iterative[State]):
    """Neural MFG solver with separate policy and density networks.

    Two multilayer perceptrons are trained in alternation, in the spirit of
    :cite:t:`heinrich2016` and :cite:t:`mao2022`: a policy network mapping
    (one-hot) state and time encodings to action logits, and a density
    network mapping a (one-hot) time encoding to a joint state-action mean
    field. On even iterations the policy network is fit to the best response
    against the density network's mean field; on odd iterations the density
    network is fit to the forward occupation measure induced by the policy
    network. The returned policy is tracked as the uniform average of past
    policies, fictitious-play style :cite:p:`perrin2020`.

    Parameters
    ----------
    lr : float
        Learning rate of both Adam optimizers.
    hidden_dim : int
        Width of each hidden layer of both networks.
    n_layers : int
        Number of hidden layers of both networks.
    time_encoding : bool
        If True, the policy network input is augmented with a one-hot
        encoding of the timestep.
    n_inner : int
        Number of Adam steps per network update.
    """

    def __init__(
        self,
        lr: float = 3e-3,
        hidden_dim: int = 64,
        n_layers: int = 2,
        time_encoding: bool = True,
        n_inner: int = 20,
    ) -> None:
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("`lr` must be a positive float")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("`hidden_dim` must be a positive integer")
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("`n_layers` must be a positive integer")
        if not isinstance(n_inner, int) or n_inner <= 0:
            raise ValueError("`n_inner` must be a positive integer")
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.time_encoding = time_encoding
        self.n_inner = n_inner

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return (
            f"NeuralTwoNetworks(lr={self.lr}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}, time_encoding={self.time_encoding}, "
            f"n_inner={self.n_inner})"
        )

    def _build_policy_net(self, env: Environment) -> nn.Module:
        in_dim = env.n_states + (env.T + 1 if self.time_encoding else 0)
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(d, self.hidden_dim))
            layers.append(nn.ReLU())
            d = self.hidden_dim
        layers.append(nn.Linear(d, env.n_actions))
        return nn.Sequential(*layers)

    def _build_density_net(self, env: Environment) -> nn.Module:
        out_dim = env.n_states * env.n_actions
        layers: list[nn.Module] = [nn.Linear(env.T + 1, self.hidden_dim), nn.ReLU()]
        d = self.hidden_dim
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(d, self.hidden_dim))
            layers.append(nn.ReLU())
            d = self.hidden_dim
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _policy_inputs(env: Environment, time_encoding: bool) -> torch.Tensor:
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
            return torch.cat([one_hot_states, one_hot_time], dim=1)
        return one_hot_states

    @staticmethod
    def _density_inputs(env: Environment) -> torch.Tensor:
        return nn.functional.one_hot(
            torch.arange(env.T + 1), num_classes=env.T + 1
        ).float()

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        torch.manual_seed(0)
        policy_net = self._build_policy_net(env)
        density_net = self._build_density_net(env)
        policy_opt = torch.optim.Adam(policy_net.parameters(), lr=self.lr)
        density_opt = torch.optim.Adam(density_net.parameters(), lr=self.lr)
        return State(
            env=env,
            pi=pi_0,
            policy_net=policy_net,
            density_net=density_net,
            policy_opt=policy_opt,
            density_opt=density_opt,
            i=0,
        )

    def _policy_step(self, state: State) -> None:
        env = state.env
        with torch.no_grad():
            time_inputs = self._density_inputs(env)
            L = (
                state.density_net(time_inputs)
                .reshape(env.T + 1, env.n_states * env.n_actions)
                .softmax(dim=-1)
                .reshape(env.T + 1, *env.S, *env.A)
            )
        Q = QFn(env, L, verify_integrity=False).optimal()
        br = torch.zeros_like(Q)
        br.scatter_(-1, Q.argmax(dim=-1, keepdim=True), 1.0)
        inputs = self._policy_inputs(env, self.time_encoding)
        shape_prefix = (env.T + 1, *env.S)
        for _ in range(self.n_inner):
            logits = state.policy_net(inputs).reshape(*shape_prefix, env.n_actions)
            loss = nn.functional.kl_div(
                logits.log_softmax(dim=-1), br, reduction="batchmean"
            )
            state.policy_opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            state.policy_opt.step()

    def _density_step(self, state: State) -> None:
        env = state.env
        with torch.no_grad():
            inputs = self._policy_inputs(env, self.time_encoding)
            pi_net = (
                state.policy_net(inputs)
                .reshape(env.T + 1, *env.S, env.n_actions)
                .softmax(dim=-1)
                .reshape(env.T + 1, *env.S, *env.A)
            )
        L_true = mean_field_from_policy(pi_net, env=env)
        target = L_true.flatten(start_dim=1).detach()
        time_inputs = self._density_inputs(env)
        for _ in range(self.n_inner):
            logits = state.density_net(time_inputs)
            loss = nn.functional.kl_div(
                logits.log_softmax(dim=-1), target, reduction="batchmean"
            )
            state.density_opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            state.density_opt.step()

    def step_next_state(
        self, state: State, atol: float | None, rtol: float | None
    ) -> State:
        if state.i % 2 == 0:
            self._policy_step(state)
        else:
            self._density_step(state)
        with torch.no_grad():
            inputs = self._policy_inputs(state.env, self.time_encoding)
            pi_net = (
                state.policy_net(inputs)
                .reshape(state.env.T + 1, *state.env.S, state.env.n_actions)
                .softmax(dim=-1)
                .reshape(state.env.T + 1, *state.env.S, *state.env.A)
            )
        n = state.i + 1.0
        pi = (state.pi * n + pi_net) / (n + 1.0)
        return State(
            env=state.env,
            pi=pi,
            policy_net=state.policy_net,
            density_net=state.density_net,
            policy_opt=state.policy_opt,
            density_opt=state.density_opt,
            i=state.i + 1,
        )

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "time_encoding": self.time_encoding,
            "n_inner": self.n_inner,
        }

    def _init_tuner_instance(self: Self, trial: optuna.Trial) -> Self:
        return type(self)(
            lr=trial.suggest_float("lr", 1e-4, 1e0, log=True),
            hidden_dim=trial.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
            n_layers=trial.suggest_int("n_layers", 1, 4),
            time_encoding=trial.suggest_categorical("time_encoding", [True, False]),
            n_inner=trial.suggest_int("n_inner", 5, 50),
        )

    def from_study(self, study: optuna.Study) -> Self:
        expected_params = {"lr", "hidden_dim", "n_layers", "time_encoding", "n_inner"}
        err_msg = f"{study.best_params.keys()=} != {expected_params}."
        assert study.best_params.keys() == expected_params, err_msg
        return type(self)(
            lr=study.best_params["lr"],
            hidden_dim=study.best_params["hidden_dim"],
            n_layers=study.best_params["n_layers"],
            time_encoding=study.best_params["time_encoding"],
            n_inner=study.best_params["n_inner"],
        )
