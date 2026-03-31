"""
Baseline: train directly on the fine-grained control timestep (dt = 0.02 s).

The observation is augmented with a *constant* normalised dt feature (= 0.0,
indicating the finest dt).  This keeps the network architecture identical to
the curriculum approach (train_dt_curriculum.py) so that a fair comparison
can be made.

Experiment design
-----------------
* Environment  : Go1JoystickFlatTerrain (default, ctrl_dt = 0.02 s)
* Total steps  : 200 M wrapper steps  (same budget as the curriculum)
* Observation  : standard state (48-d) + dt_norm scalar = 49-d
                 standard privileged_state (97-d) + dt_norm = 98-d
* dt_norm      : always 0.0 for this baseline

Usage
-----
    python train_dt_baseline.py                      # default settings
    python train_dt_baseline.py seed=2               # different seed
    python train_dt_baseline.py num_timesteps=50000000
    DEBUG=1 python train_dt_baseline.py              # fast smoke-test
"""

import functools
import json
import logging
import os

# ── JAX / GPU setup (must happen before any JAX import) ──────────────────────
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

from dt_curriculum_wrapper import DtObsWrapper
from experiment_launcher import single_experiment, run_experiment

log = logging.getLogger(__name__)
DEBUG = bool(os.environ.get("DEBUG", ""))

# ── Curriculum constants (shared with train_dt_curriculum.py) ─────────────────
DT_FINE = 0.02    # target dt – the one we're training on
DT_COARSE = 0.08  # coarsest dt in the curriculum (only used for normalisation)

EVAL_INTERVAL = 200_000  # one evaluation every this many wrapper steps


@single_experiment
def main(
    env_name: str = "Go1JoystickFlatTerrain",
    seed: int = 1,
    num_timesteps: int = 200_000_000,
    results_dir: str = "./results",
):
    """Train a locomotion policy directly on the fine-grained dt (baseline)."""

    # ── Env / PPO config ──────────────────────────────────────────────────────
    env_cfg = registry.get_default_config(env_name)
    ppo_config = locomotion_params.brax_ppo_config(env_name)

    ppo_config.num_timesteps = num_timesteps
    ppo_config.num_evals = max(1, num_timesteps // EVAL_INTERVAL)
    ppo_config.seed = seed

    if DEBUG:
        ppo_config.num_evals = 2
        ppo_config.num_timesteps = 50_000

    # ── Checkpoint directory ─────────────────────────────────────────────────
    run_name = f"{env_name}_dt_baseline_seed{seed}"
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Run name      : {run_name}")
    print(f"Checkpoint dir: {ckpt_dir}")

    # ── Build environments ────────────────────────────────────────────────────
    # Training and evaluation envs both use the fine dt; DtObsWrapper adds
    # dt_norm=0.0 as a constant feature to keep the obs size consistent with
    # the curriculum approach.
    env = DtObsWrapper(
        registry.load(env_name, config=env_cfg),
        dt_fine=DT_FINE, dt_coarse=DT_COARSE,
    )
    eval_env = DtObsWrapper(
        registry.load(env_name, config=env_cfg),
        dt_fine=DT_FINE, dt_coarse=DT_COARSE,
    )
    randomizer_fn = registry.get_domain_randomizer(env_name)

    # ── Network factory ───────────────────────────────────────────────────────
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_config.network_factory,
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    wandb.init(
        project=f"dt-curriculum-{env_name}",
        name=run_name,
        group=f"{env_name}_dt_baseline",
        config={
            "approach": "baseline",
            "env_name": env_name,
            "ctrl_dt": DT_FINE,
            "dt_fine": DT_FINE,
            "dt_coarse": DT_COARSE,
            "env_config": env_cfg.to_dict(),
            "ppo_config": ppo_config.to_dict(),
            "seed": seed,
        },
        mode="disabled" if DEBUG else "online",
    )

    def progress_fn(num_steps: int, metrics: dict) -> None:
        wandb.log({"approach": "baseline", **metrics}, step=num_steps)
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(f"[Baseline] step {num_steps:>12,}: reward={reward:.3f}")

    # ── Training ──────────────────────────────────────────────────────────────
    train_params = {k: v for k, v in ppo_config.items()
                    if k != "network_factory"}

    print("\nJIT-compiling and starting training …")
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        progress_fn=progress_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomizer_fn,
        save_checkpoint_path=ckpt_dir,
        **train_params,
    )

    # ── Persist config ────────────────────────────────────────────────────────
    with open(os.path.join(ckpt_dir, "config.json"), "w") as fh:
        json.dump({
            "approach": "baseline",
            "env_name": env_name,
            "ctrl_dt": DT_FINE,
            "dt_fine": DT_FINE,
            "dt_coarse": DT_COARSE,
            "seed": seed,
            "env_config": env_cfg.to_dict(),
            "ppo_config": ppo_config.to_dict(),
        }, fh, indent=4)

    print(f"\nBaseline training complete.  Checkpoint: {ckpt_dir}")
    wandb.finish()


if __name__ == "__main__":
    run_experiment(main)
