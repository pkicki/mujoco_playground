"""
Coarse-to-fine dt curriculum for locomotion policy learning.

Research idea
-------------
Instead of training a policy at a fixed fine-grained control timestep (dt),
start with a coarser dt (easier dynamics, longer "horizon per step") and
gradually refine toward the target dt.  The policy always observes the current
dt so it can condition its behaviour on the control frequency.

Two curriculum modes are implemented:

  sequential  (default)
      Three phases with a *fixed* dt per phase.  Each phase creates a fresh
      env and warm-starts from the previous phase's final parameters.

          Phase 1 – coarse  (ctrl_dt = 0.08 s, 40 M steps)
          Phase 2 – medium  (ctrl_dt = 0.04 s, 40 M steps)
          Phase 3 – fine    (ctrl_dt = 0.02 s, 120 M steps)  ← target dt

      Per-phase adjustments for physical consistency:
        • episode_length   scaled so wall-clock episode duration is constant
        • reward_scaling   divided by dt_ratio so reward/s is comparable
        • discounting      raised to dt_ratio so per-second discount matches

  stochastic
      Three phases, each using the *finest* env (ctrl_dt = 0.02 s) but with
      DtCurriculumWrapper sampling a dt multiplier k ∈ {4, 2, 1} per episode.
      The sampling distribution shifts from coarse-heavy to fine-heavy across
      phases.  Warm-start between phases is applied as in the sequential mode.

  continuous
      Twenty mini-phases of 10 M steps each (200 M total).  Each phase uses
      ContinuousDtCurriculumWrapper which samples k from a *continuous*
      log-uniform (or linear-uniform) distribution over [1, k_max_sample],
      rounds to the nearest integer, and runs k fine-dt env steps per wrapper
      step.  k_max_sample is linearly (or log) annealed from k_max_scan=4
      down to 1 across phases, giving a smooth curriculum.  The lax.scan
      length is fixed at k_max_scan so the compiled graph is shared across
      all mini-phases.

Comparison
----------
All three modes are designed to be compared against train_dt_baseline.py with
the *same* total wrapper-step budget (200 M) and the same observation structure
(state + dt_norm feature = 49-d, privileged_state + dt_norm = 98-d).

Logging
-------
All phases share a single W&B run.  The x-axis is the cumulative training
step (summed across phases) so training curves are directly comparable with
the baseline run.

Usage
-----
    python train_dt_curriculum.py                        # sequential mode
    python train_dt_curriculum.py mode=stochastic        # stochastic mode
    python train_dt_curriculum.py mode=continuous        # continuous mode
    python train_dt_curriculum.py mode=continuous anneal_mode=log log_uniform=False
    python train_dt_curriculum.py seed=2
    DEBUG=1 python train_dt_curriculum.py                # fast smoke-test
"""

import copy
import functools
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ── JAX / GPU setup ──────────────────────────────────────────────────────────
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from dt_curriculum_wrapper import (
    ContinuousDtCurriculumWrapper,
    DtCurriculumWrapper,
    DtObsWrapper,
    expected_k,
)
from experiment_launcher import single_experiment, run_experiment

log = logging.getLogger(__name__)
DEBUG = bool(os.environ.get("DEBUG", ""))

# ── Constants ─────────────────────────────────────────────────────────────────
DT_FINE = 0.02    # target (finest) control timestep [s]
DT_COARSE = 0.08  # coarsest timestep in the curriculum [s]

# Physical episode duration [s] — kept constant across phases so that each
# episode sees the same amount of simulated locomotion regardless of ctrl_dt.
EPISODE_DURATION_S = 20.0   # = 1000 steps × 0.02 s

# Base PPO hyperparameters (for the finest dt).  Phase-specific values are
# derived from these via the dt_ratio = ctrl_dt / DT_FINE scale factor.
BASE_DISCOUNTING = 0.97      # per fine-step discount
BASE_REWARD_SCALING = 1.0

# One W&B / console evaluation every this many cumulative wrapper steps.
EVAL_INTERVAL = 20_000_000

# Continuous curriculum settings.
# k_max_scan is the FIXED static scan length shared across all mini-phases.
# k_max_sample starts at K_MAX_SCAN and is annealed to 1 over the phases.
K_MAX_SCAN: int = 4              # must be a Python int (compile-time constant)
N_CONTINUOUS_PHASES: int = 20    # number of mini-phases (10 M steps each)
STEPS_PER_CONTINUOUS_PHASE: int = 200_000_000 // N_CONTINUOUS_PHASES  # 10 M


# ── Continuous curriculum helper ─────────────────────────────────────────────

def _expected_k_continuous(
    k_max_sample: float,
    k_min: float = 1.0,
    log_uniform: bool = True,
) -> float:
    """Analytic expected value of the continuous k multiplier before rounding.

    Log-uniform on [k_min, k_max_sample]:
        E[k] = (k_max - k_min) / log(k_max / k_min)
    Linear-uniform on [k_min, k_max_sample]:
        E[k] = (k_min + k_max_sample) / 2
    """
    if k_max_sample <= k_min + 1e-9:
        return float(k_min)
    if log_uniform:
        return (k_max_sample - k_min) / math.log(k_max_sample / k_min)
    return (k_min + k_max_sample) / 2.0


# ── Phase definitions ─────────────────────────────────────────────────────────

@dataclass
class SequentialPhase:
    """One phase of the sequential curriculum."""
    name: str
    ctrl_dt: float          # control timestep for this phase
    n_timesteps: int        # number of wrapper steps to train for

    @property
    def dt_ratio(self) -> float:
        """ctrl_dt relative to DT_FINE (always >= 1)."""
        return self.ctrl_dt / DT_FINE

    @property
    def episode_length(self) -> int:
        """Episode length in steps so physical duration = EPISODE_DURATION_S."""
        return int(round(EPISODE_DURATION_S / self.ctrl_dt))

    @property
    def reward_scaling(self) -> float:
        """Reward scaling to compensate for the dt-proportional reward."""
        return BASE_REWARD_SCALING / self.dt_ratio

    @property
    def discounting(self) -> float:
        """Per-wrapper-step discount matching BASE_DISCOUNTING per fine step."""
        return BASE_DISCOUNTING ** self.dt_ratio

    @property
    def num_evals(self) -> int:
        return max(1, self.n_timesteps // EVAL_INTERVAL)


@dataclass
class StochasticPhase:
    """One phase of the stochastic curriculum (DtCurriculumWrapper)."""
    name: str
    k_levels: List[int]     # dt multipliers available in this phase
    k_probs: List[float]    # sampling probabilities over k_levels
    n_timesteps: int        # number of wrapper steps to train for

    @property
    def expected_dt(self) -> float:
        return DT_FINE * expected_k(self.k_levels, self.k_probs)

    @property
    def dt_ratio(self) -> float:
        return self.expected_dt / DT_FINE

    @property
    def episode_length(self) -> int:
        # Use the standard fine-dt episode length; stochastic episodes differ
        # in physical duration by design (coarser episodes are longer in time).
        return int(round(EPISODE_DURATION_S / DT_FINE))

    @property
    def reward_scaling(self) -> float:
        # The wrapper sums rewards over k fine steps, so average magnitude
        # scales with k.  Adjust by the expected k.
        return BASE_REWARD_SCALING / max(self.dt_ratio, 1.0)

    @property
    def discounting(self) -> float:
        # Per-wrapper-step discount for the expected k.
        return BASE_DISCOUNTING ** max(self.dt_ratio, 1.0)

    @property
    def num_evals(self) -> int:
        return max(1, self.n_timesteps // EVAL_INTERVAL)


# ── Phase schedules ───────────────────────────────────────────────────────────

SEQUENTIAL_PHASES: List[SequentialPhase] = [
    SequentialPhase(name="coarse", ctrl_dt=0.08, n_timesteps=40_000_000),
    SequentialPhase(name="medium", ctrl_dt=0.04, n_timesteps=40_000_000),
    SequentialPhase(name="fine",   ctrl_dt=0.02, n_timesteps=120_000_000),
]

STOCHASTIC_PHASES: List[StochasticPhase] = [
    StochasticPhase(
        name="mostly-coarse",
        k_levels=[4, 2, 1],
        k_probs=[0.70, 0.20, 0.10],   # P(coarse) = 0.70
        n_timesteps=67_000_000,
    ),
    StochasticPhase(
        name="mixed",
        k_levels=[4, 2, 1],
        k_probs=[0.20, 0.60, 0.20],   # P(medium) = 0.60
        n_timesteps=67_000_000,
    ),
    StochasticPhase(
        name="mostly-fine",
        k_levels=[4, 2, 1],
        k_probs=[0.05, 0.15, 0.80],   # P(fine) = 0.80
        n_timesteps=66_000_000,
    ),
]


# ── Training helper ───────────────────────────────────────────────────────────

def _run_phase(
    *,
    env,
    eval_env,
    ppo_config,
    network_factory,
    randomizer_fn,
    ckpt_dir: str,
    phase_name: str,
    phase_idx: int,
    ctrl_dt: float,
    global_step_offset: int,
    restore_params: Optional[Any],
    wandb_extra: Dict[str, Any],
    wrap_env_fn=None,
) -> Any:
    """Run one training phase and return the final params for warm-starting."""

    if wrap_env_fn is None:
        wrap_env_fn = wrapper.wrap_for_brax_training

    train_params = {k: v for k, v in ppo_config.items()
                    if k != "network_factory"}

    phase_ckpt = os.path.join(ckpt_dir, phase_name)
    os.makedirs(phase_ckpt, exist_ok=True)

    def progress_fn(num_steps: int, metrics: dict) -> None:
        global_step = global_step_offset + num_steps
        wandb.log(
            {
                "curriculum/phase_idx": phase_idx,
                "curriculum/phase_name": phase_name,
                "curriculum/ctrl_dt": ctrl_dt,
                **wandb_extra,
                **metrics,
            },
            step=global_step,
        )
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(
            f"  [Phase {phase_idx} – {phase_name:>14s}  dt={ctrl_dt:.3f}]"
            f"  global_step={global_step:>12,}  reward={reward:.3f}"
        )

    print(f"\n{'='*70}")
    print(f"  Phase {phase_idx}: {phase_name}  (ctrl_dt={ctrl_dt:.4f} s)")
    print(f"  Steps: {train_params['num_timesteps']:,}   "
          f"episode_length: {train_params['episode_length']}   "
          f"reward_scaling: {train_params['reward_scaling']:.3f}   "
          f"discounting: {train_params['discounting']:.4f}")
    print(f"{'='*70}\n")

    _, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        progress_fn=progress_fn,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomizer_fn,
        save_checkpoint_path=phase_ckpt,
        restore_params=restore_params,
        restore_value_fn=True,
        **train_params,
    )
    return params


# ── Main ──────────────────────────────────────────────────────────────────────

@single_experiment
def main(
    env_name: str = "Go1JoystickFlatTerrain",
    mode: str = "sequential",       # "sequential" | "stochastic" | "continuous"
    seed: int = 1,
    results_dir: str = "./results",
    anneal_mode: str = "linear",    # continuous only: "linear" | "log"
    log_uniform: bool = True,       # continuous only: log-uniform vs linear-uniform
):
    """Run the dt-curriculum training experiment.

    Parameters
    ----------
    env_name:
        MuJoCo Playground environment identifier.
    mode:
        ``"sequential"`` – one fixed dt per phase, annealed coarse→fine.
        ``"stochastic"`` – dt sampled per episode from an evolving categorical.
        ``"continuous"`` – dt sampled per episode from a continuous log/linear
        distribution whose upper bound is annealed over 20 mini-phases.
    seed:
        Random seed for reproducibility.
    results_dir:
        Unused (kept for experiment_launcher compatibility).
    anneal_mode:
        ``"linear"`` or ``"log"`` annealing of k_max_sample.  Only used when
        mode is ``"continuous"``.
    log_uniform:
        If True (default), sample k log-uniformly in the continuous mode.
    """

    assert mode in ("sequential", "stochastic", "continuous"), \
        f"mode must be 'sequential', 'stochastic', or 'continuous', got '{mode}'"

    # ── Checkpoint root ───────────────────────────────────────────────────────
    run_name = f"{env_name}_dt_curriculum_{mode}_seed{seed}"
    ckpt_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints", run_name)
    os.makedirs(ckpt_root, exist_ok=True)
    print(f"Run name      : {run_name}")
    print(f"Checkpoint dir: {ckpt_root}")
    print(f"Mode          : {mode}")

    # ── Shared PPO hyperparameters (overridden per phase below) ───────────────
    base_ppo_config = locomotion_params.brax_ppo_config(env_name)
    base_ppo_config.seed = seed

    # ── Domain randomiser (same for all phases) ───────────────────────────────
    randomizer_fn = registry.get_domain_randomizer(env_name)

    # The eval env always uses the *fine* dt so evaluation metrics are
    # directly comparable to the baseline throughout all training phases.
    fine_env_cfg = registry.get_default_config(env_name)
    eval_env = DtObsWrapper(
        registry.load(env_name, config=fine_env_cfg),
        dt_fine=DT_FINE, dt_coarse=DT_COARSE,
    )

    # ── W&B initialisation ────────────────────────────────────────────────────
    if mode == "continuous":
        wandb_phases = [
            {
                "phase_idx": i,
                "k_max_sample": (
                    K_MAX_SCAN ** (1.0 - i / max(N_CONTINUOUS_PHASES - 1, 1))
                    if anneal_mode == "log"
                    else K_MAX_SCAN - i / max(N_CONTINUOUS_PHASES - 1, 1) * (K_MAX_SCAN - 1)
                ),
                "n_timesteps": STEPS_PER_CONTINUOUS_PHASE,
            }
            for i in range(N_CONTINUOUS_PHASES)
        ]
    else:
        disc_phases = SEQUENTIAL_PHASES if mode == "sequential" else STOCHASTIC_PHASES
        wandb_phases = [
            {
                "name": p.name,
                "n_timesteps": p.n_timesteps,
                **({"ctrl_dt": p.ctrl_dt} if hasattr(p, "ctrl_dt") else {}),
                **({"k_probs": p.k_probs, "k_levels": p.k_levels}
                   if hasattr(p, "k_probs") else {}),
            }
            for p in disc_phases
        ]
    wandb.init(
        project=f"dt-curriculum-{env_name}",
        name=run_name,
        group=f"{env_name}_dt_curriculum_{mode}",
        config={
            "approach": f"curriculum_{mode}",
            "mode": mode,
            "env_name": env_name,
            "dt_fine": DT_FINE,
            "dt_coarse": DT_COARSE,
            "seed": seed,
            **({"anneal_mode": anneal_mode,
                "log_uniform": log_uniform,
                "k_max_scan": K_MAX_SCAN,
                "n_continuous_phases": N_CONTINUOUS_PHASES,
               } if mode == "continuous" else {}),
            "phases": wandb_phases,
        },
        mode="disabled" if DEBUG else "online",
    )

    # ── Phase loop ────────────────────────────────────────────────────────────
    params = None          # will hold (normalizer, policy, value) for warm-start
    global_step_offset = 0

    if mode == "sequential":
        _run_sequential_phases(
            phases=SEQUENTIAL_PHASES,
            base_ppo_config=base_ppo_config,
            env_name=env_name,
            eval_env=eval_env,
            randomizer_fn=randomizer_fn,
            ckpt_root=ckpt_root,
            params=params,
            global_step_offset=global_step_offset,
        )

    elif mode == "stochastic":
        _run_stochastic_phases(
            phases=STOCHASTIC_PHASES,
            base_ppo_config=base_ppo_config,
            env_name=env_name,
            eval_env=eval_env,
            randomizer_fn=randomizer_fn,
            ckpt_root=ckpt_root,
            params=params,
            global_step_offset=global_step_offset,
        )

    else:  # continuous
        _run_continuous_curriculum(
            base_ppo_config=base_ppo_config,
            env_name=env_name,
            eval_env=eval_env,
            randomizer_fn=randomizer_fn,
            ckpt_root=ckpt_root,
            anneal_mode=anneal_mode,
            log_uniform=log_uniform,
        )

    # ── Persist run config ────────────────────────────────────────────────────
    with open(os.path.join(ckpt_root, "config.json"), "w") as fh:
        json.dump({
            "approach": f"curriculum_{mode}",
            "mode": mode,
            "env_name": env_name,
            "dt_fine": DT_FINE,
            "dt_coarse": DT_COARSE,
            "seed": seed,
        }, fh, indent=4)

    print(f"\nCurriculum training ({mode}) complete.  Checkpoint: {ckpt_root}")
    wandb.finish()


# ── Sequential phase runner ───────────────────────────────────────────────────

def _run_sequential_phases(
    *,
    phases: List[SequentialPhase],
    base_ppo_config,
    env_name: str,
    eval_env,
    randomizer_fn,
    ckpt_root: str,
    params,
    global_step_offset: int,
):
    """Execute the sequential coarse→fine phase curriculum.

    Each phase:
      - Creates a fresh env with the phase-specific ctrl_dt.
      - Adjusts episode_length, reward_scaling, and discounting to keep
        physical episode duration and reward magnitude consistent.
      - Warm-starts the network weights from the previous phase's final params.
    """
    for phase_idx, phase in enumerate(phases):
        ppo_config = copy.deepcopy(base_ppo_config)
        ppo_config.num_timesteps = (
            20_000 if DEBUG else phase.n_timesteps
        )
        ppo_config.num_evals = max(1, ppo_config.num_timesteps // max(EVAL_INTERVAL, 1))
        ppo_config.episode_length = phase.episode_length
        ppo_config.reward_scaling = phase.reward_scaling
        ppo_config.discounting = phase.discounting

        # Build training env for this phase's ctrl_dt.
        env_cfg = registry.get_default_config(env_name)
        env_cfg.ctrl_dt = phase.ctrl_dt
        train_env = DtObsWrapper(
            registry.load(env_name, config=env_cfg),
            dt_fine=DT_FINE, dt_coarse=DT_COARSE,
        )

        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_config.network_factory,
        )

        params = _run_phase(
            env=train_env,
            eval_env=eval_env,
            ppo_config=ppo_config,
            network_factory=network_factory,
            randomizer_fn=randomizer_fn,
            ckpt_dir=ckpt_root,
            phase_name=phase.name,
            phase_idx=phase_idx,
            ctrl_dt=phase.ctrl_dt,
            global_step_offset=global_step_offset,
            restore_params=params,
            wandb_extra={
                "curriculum/episode_length": phase.episode_length,
                "curriculum/reward_scaling": phase.reward_scaling,
                "curriculum/discounting": phase.discounting,
            },
        )
        global_step_offset += phase.n_timesteps

    return params


# ── Stochastic phase runner ───────────────────────────────────────────────────

def _run_stochastic_phases(
    *,
    phases: List[StochasticPhase],
    base_ppo_config,
    env_name: str,
    eval_env,
    randomizer_fn,
    ckpt_root: str,
    params,
    global_step_offset: int,
):
    """Execute the stochastic dt curriculum.

    Each phase uses the fine-dt base env but DtCurriculumWrapper samples k per
    episode.  Creating a new wrapper with different k_probs triggers a JAX
    recompile — this is a one-time cost (~5–10 min) per phase transition.

    Discount and reward_scaling are set to values appropriate for the expected
    k under the phase's distribution (see StochasticPhase dataclass).
    """
    for phase_idx, phase in enumerate(phases):
        ppo_config = copy.deepcopy(base_ppo_config)
        ppo_config.num_timesteps = (
            20_000 if DEBUG else phase.n_timesteps
        )
        ppo_config.num_evals = max(1, ppo_config.num_timesteps // max(EVAL_INTERVAL, 1))
        ppo_config.episode_length = phase.episode_length
        ppo_config.reward_scaling = phase.reward_scaling
        ppo_config.discounting = phase.discounting

        # The base env always uses the finest dt; DtCurriculumWrapper
        # implements coarser frequencies via action repeat.
        env_cfg = registry.get_default_config(env_name)
        train_env = DtCurriculumWrapper(
            registry.load(env_name, config=env_cfg),
            k_levels=phase.k_levels,
            k_probs=phase.k_probs,
            dt_fine=DT_FINE,
            dt_coarse=DT_COARSE,
        )

        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_config.network_factory,
        )

        # DtCurriculumWrapper samples k at each episode reset.  full_reset=True
        # ensures the AutoResetWrapper calls our reset() on every episode end
        # so that a fresh k_idx (and thus a fresh dt_norm feature) is sampled.
        stochastic_wrap_fn = functools.partial(
            wrapper.wrap_for_brax_training, full_reset=True
        )

        params = _run_phase(
            env=train_env,
            eval_env=eval_env,
            ppo_config=ppo_config,
            network_factory=network_factory,
            randomizer_fn=randomizer_fn,
            ckpt_dir=ckpt_root,
            phase_name=phase.name,
            phase_idx=phase_idx,
            ctrl_dt=phase.expected_dt,
            global_step_offset=global_step_offset,
            restore_params=params,
            wrap_env_fn=stochastic_wrap_fn,
            wandb_extra={
                "curriculum/k_probs": str(phase.k_probs),
                "curriculum/expected_k": expected_k(phase.k_levels, phase.k_probs),
                "curriculum/expected_dt": phase.expected_dt,
                "curriculum/reward_scaling": phase.reward_scaling,
                "curriculum/discounting": phase.discounting,
            },
        )
        global_step_offset += phase.n_timesteps

    return params


# ── Continuous curriculum runner ─────────────────────────────────────────────

def _run_continuous_curriculum(
    *,
    base_ppo_config,
    env_name: str,
    eval_env,
    randomizer_fn,
    ckpt_root: str,
    anneal_mode: str = "linear",
    log_uniform: bool = True,
    k_max_scan: int = K_MAX_SCAN,
    n_phases: int = N_CONTINUOUS_PHASES,
):
    """Execute the continuous dt curriculum as n_phases mini-phases.

    k_max_sample is annealed from k_max_scan → 1 over n_phases steps (linear
    or log schedule).  Within each mini-phase k is drawn from a continuous
    log/linear-uniform distribution over [1, k_max_sample] and rounded to int.

    The lax.scan length is always k_max_scan (a Python int), so the compiled
    graph is shared across all mini-phases.  Only the lightweight sampling
    constants change between phases, incurring a cheap partial retrace.

    Discount and reward_scaling are adjusted per phase using the analytic
    expected value E[k] under the current distribution.
    """
    params = None
    global_step_offset = 0
    steps_per_phase = 20_000 if DEBUG else STEPS_PER_CONTINUOUS_PHASE

    # full_reset=True so AutoResetWrapper calls reset() each episode,
    # allowing a fresh k sample (and thus a fresh dt_norm feature) per episode.
    continuous_wrap_fn = functools.partial(
        wrapper.wrap_for_brax_training, full_reset=True
    )

    for phase_idx in range(n_phases):
        progress = phase_idx / max(n_phases - 1, 1)   # 0.0 → 1.0

        # Anneal k_max_sample from k_max_scan down to 1.0
        if anneal_mode == "log":
            k_max_sample = float(k_max_scan) ** (1.0 - progress)
        else:  # linear
            k_max_sample = k_max_scan - progress * (k_max_scan - 1.0)
        k_max_sample = max(1.0, k_max_sample)

        # Analytic expected k under the current sampling distribution
        ek = _expected_k_continuous(k_max_sample, log_uniform=log_uniform)

        ppo_config = copy.deepcopy(base_ppo_config)
        ppo_config.num_timesteps = steps_per_phase
        ppo_config.num_evals = max(1, steps_per_phase // max(EVAL_INTERVAL, 1))
        # Episode length in fine-dt steps; physical duration is fixed
        ppo_config.episode_length = int(round(EPISODE_DURATION_S / DT_FINE))
        # Reward sums over k substeps → scale back by E[k]
        ppo_config.reward_scaling = BASE_REWARD_SCALING / ek
        # Per-wrapper-step discount consistent with BASE_DISCOUNTING per fine step
        ppo_config.discounting = BASE_DISCOUNTING ** ek

        env_cfg = registry.get_default_config(env_name)
        train_env = ContinuousDtCurriculumWrapper(
            registry.load(env_name, config=env_cfg),
            k_max_scan=k_max_scan,       # static scan length, unchanged each phase
            k_max_sample=k_max_sample,   # annealed sampling upper bound
            log_uniform=log_uniform,
            dt_fine=DT_FINE,
            dt_coarse=DT_COARSE,
        )

        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_config.network_factory,
        )

        print(f"\n{'='*70}")
        print(f"  Continuous phase {phase_idx:02d}/{n_phases-1}  "
              f"progress={progress:.2f}  k_max_sample={k_max_sample:.3f}  "
              f"E[k]={ek:.3f}")
        print(f"  gamma={ppo_config.discounting:.4f}  "
              f"rew_scale={ppo_config.reward_scaling:.3f}  "
              f"steps={steps_per_phase:,}")
        print(f"{'='*70}\n")

        params = _run_phase(
            env=train_env,
            eval_env=eval_env,
            ppo_config=ppo_config,
            network_factory=network_factory,
            randomizer_fn=randomizer_fn,
            ckpt_dir=ckpt_root,
            phase_name=f"phase_{phase_idx:02d}",
            phase_idx=phase_idx,
            ctrl_dt=ek * DT_FINE,
            global_step_offset=global_step_offset,
            restore_params=params,
            wrap_env_fn=continuous_wrap_fn,
            wandb_extra={
                "curriculum/k_max_sample": k_max_sample,
                "curriculum/expected_k": ek,
                "curriculum/expected_dt": ek * DT_FINE,
                "curriculum/anneal_progress": progress,
                "curriculum/reward_scaling": ppo_config.reward_scaling,
                "curriculum/discounting": ppo_config.discounting,
            },
        )
        global_step_offset += steps_per_phase

    return params


if __name__ == "__main__":
    run_experiment(main)
