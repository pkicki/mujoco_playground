import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper
from typing import Any, Dict, Optional, Tuple

class HermiteSplineWrapper(Wrapper):
    """
    Hermite Spline Wrapper for MuJoCo Playground (MJX).
    Interpolates position and velocity targets between policy steps.
    """

    def __init__(
        self, 
        env: mjx_env.MjxEnv, 
        substeps: int, 
        vmax: float, 
        kp: float, 
        kd: float
    ):
        super().__init__(env)
        self._substeps = substeps
        self._vmax = vmax
        self._kp = kp
        self._kd = kd
        
        # Duration of one policy step
        self._T = env.dt 
        
        # Map actuators to joint indices
        mj_model = env.mj_model
        n_actuators = mj_model.nu
        
        qpos_indices = []
        qvel_indices = []
        joint_limits = []

        for i in range(n_actuators):
            # trnid maps actuator to joint ID
            joint_id = mj_model.actuator_trnid[i, 0]
            qpos_indices.append(mj_model.jnt_qposadr[joint_id])
            qvel_indices.append(mj_model.jnt_dofadr[joint_id])
            joint_limits.append(mj_model.jnt_range[joint_id])

        self._qpos_indices = jp.array(qpos_indices)
        self._qvel_indices = jp.array(qvel_indices)
        self._joint_limits = jp.array(joint_limits)

    @property
    def action_size(self) -> int:
        # Policy now outputs [Target_Pos, Target_Vel] for each actuator
        return self.env.action_size * 2

    def _scale_action(self, action: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Maps normalized policy output [-1, 1] to physical targets."""
        n_acts = len(self._qpos_indices)
        pos_raw = action[:n_acts]
        vel_raw = action[n_acts:]
        
        j_min = self._joint_limits[:, 0]
        j_max = self._joint_limits[:, 1]
        
        # Scale Position to [min, max]
        target_pos = j_min + (pos_raw + 1.0) * 0.5 * (j_max - j_min)

        # Velocity constraint to prevent hitting limits within the timestep T
        # v <= 2 * dist_to_limit / T
        dq = jp.minimum(j_max - target_pos, target_pos - j_min)
        dynamic_v_max = 2.0 * dq / self._T
        
        # Scale Velocity to [-vmax, vmax] then clip by dynamic safety limit
        target_vel = vel_raw * self._vmax
        target_vel = jp.clip(target_vel, -dynamic_v_max, dynamic_v_max)
        
        return target_pos, target_vel

    def _hermite_spline(self, p0, v0, p1, v1, tau):
        """Cubic Hermite Spline math."""
        m0 = v0 * self._T
        m1 = v1 * self._T
        
        tau2 = tau * tau
        tau3 = tau2 * tau
        
        h00 = 2*tau3 - 3*tau2 + 1
        h10 = tau3 - 2*tau2 + tau
        h01 = -2*tau3 + 3*tau2
        h11 = tau3 - tau2
        
        p_des = h00*p0 + h10*m0 + h01*p1 + h11*m1
        
        dh00 = 6*tau2 - 6*tau
        dh10 = 3*tau2 - 4*tau + 1
        dh01 = -6*tau2 + 6*tau
        dh11 = 3*tau2 - 2*tau
        
        v_des = (dh00*p0 + dh10*m0 + dh01*p1 + dh11*m1) / self._T
        
        return p_des, v_des

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # 1. Scale Targets
        p1, v1 = self._scale_action(action)
        
        # 2. Get Start state
        p0 = state.data.qpos[self._qpos_indices]
        v0 = state.data.qvel[self._qvel_indices]

        # 3. Define the sub-stepping loop via lax.scan
        def sub_step_fn(carry_state, step_idx):
            tau = (step_idx + 1) / self._substeps
            p_des, v_des = self._hermite_spline(p0, v0, p1, v1, tau)
            
            curr_p = carry_state.data.qpos[self._qpos_indices]
            curr_v = carry_state.data.qvel[self._qvel_indices]
            
            # PD Control
            torque = self._kp * (p_des - curr_p) + self._kd * (v_des - curr_v)
            
            # Use the underlying env's step (which likely does its own physics sub-stepping)
            # Note: We divide action_repeat by substeps if necessary, 
            # but usually we assume the env.step here is a single sim step or a small group.
            next_state = self.env.step(carry_state, torque)
            
            return next_state, next_state.reward

        # 4. Execute the scan
        # We step the environment 'substeps' times within one policy step.
        final_state, rewards = jax.lax.scan(
            sub_step_fn, state, jp.arange(self._substeps)
        )

        # 5. Aggregate state
        # In MJX Playground, 'done' is usually a max() over the sub-steps, 
        # and reward is usually a sum.
        return final_state.replace(
            reward=jp.sum(rewards)
        )