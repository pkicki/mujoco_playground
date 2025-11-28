import os
import datetime
import functools
import json

# Set necessary environment variables for JAX/MuJoCo performance
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import jax
import wandb
from orbax import checkpoint as ocp
from flax.training import orbax_utils

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acting import generate_unroll
from brax.training.acting_statefully import generate_stateful_unroll

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from experiment_launcher import single_experiment, run_experiment

@single_experiment
def main(
    #noise_type: str = "white",
    noise_type: str = "lp",
    cutoff_freq: float = 3.0,
    order: int = 2,
    entropy_cost: float = 0e-4,
    unroll_length: int = 20,
    env_name: str = "Go1JoystickFlatTerrain",
    results_dir: str = "./results",
    seed: int = 1,
):
    # Load default environment config and PPO hyperparameters
    env_cfg = registry.get_default_config(env_name)
    ppo_config = locomotion_params.brax_ppo_config(env_name)
    
    # Customize training parameters if desired
    ppo_config.num_timesteps = 100_000_000
    ppo_config.num_evals = 50
    ppo_config.entropy_cost = entropy_cost
    ppo_config.seed = seed
    ppo_config.unroll_length = unroll_length
    
    # Create a unique experiment name
    group_name = f"{env_name}_{noise_type}_ent{entropy_cost}_ul{unroll_length}"
    if noise_type == "lp":
        group_name += f"_cf{cutoff_freq}_o{order}"
    run_name = f"{group_name}_seed{seed}"
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Starting training run: {run_name}")
    print(f"Checkpoints directory: {ckpt_dir}")

    # 2. Initialize Weights & Biases
    wandb.init(
        project=f"mujoco-playground-{env_name}",
        name=run_name,
        group=group_name,
        config={
            "env_config": env_cfg.to_dict(),
            "ppo_config": ppo_config.to_dict(),
            "env_name": env_name,
            "noise_type": noise_type,
        }
    )

    # 3. Setup Environment and Randomization
    # Load the environment
    env = registry.load(env_name, config=env_cfg)
    # Load the evaluation environment (usually same config, but distinct instance)
    eval_env = registry.load(env_name, config=env_cfg)
    
    # Get the domain randomization function specific to Go1
    randomizer_fn = registry.get_domain_randomizer(env_name)

    unroll_fn = generate_unroll
    if noise_type == "lp":
        unroll_fn = functools.partial(
            generate_stateful_unroll,
            order=order, cutoff_freq=cutoff_freq
        )

    # 4. Setup Network Factory
    # Helper to create PPO networks based on config
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_config.network_factory
    )

    # 5. Define Callbacks
    
    # Progress callback: logs to wandb and prints to console
    def progress_fn(num_steps, metrics):
        wandb.log(metrics, step=num_steps)
        print(f"Step {num_steps}: Reward = {metrics['eval/episode_reward']:.3f}")

    # Checkpointing callback: saves Flax params using Orbax
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    
    def policy_params_fn(current_step, make_policy, params):
        # Save parameters
        save_args = orbax_utils.save_args_from_target(params)
        path = os.path.join(ckpt_dir, f"{current_step}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        
        # Save config for reproducibility
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            json.dump(env_cfg.to_dict(), f, indent=4)

    # 6. Start Training
    # We extract the training arguments from the ConfigDict
    train_params = dict(ppo_config)
    if "network_factory" in train_params:
        del train_params["network_factory"]

    print("JIT compiling and starting training loop...")
    
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomizer_fn,
        unroll_fn=unroll_fn,
        **train_params
    )

    print(f"Training complete. Model saved to {ckpt_dir}")
    wandb.finish()

if __name__ == "__main__":
    run_experiment(main)