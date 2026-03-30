import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper
from typing import Any, Dict, Optional, Tuple

class FastHermiteSplineWrapper(Wrapper):
    """
    High-level Hermite Spline controller for MJX Playground.
    - Policy runs at spline_freq (e.g., 25Hz).
    - Internal PD loop runs at 1kHz (forced via lax.scan).
    - Action Space: [delta_pos, delta_vel] mapped via vmax/amax.
    """

    def __init__(
        self, 
        env: mjx_env.MjxEnv, 
        spline_freq: float, 
        vmax: float, 
        amax: float, 
        kp: float, 
        kd: float,
        physics_dt: float = 0.001
    ):
        super().__init__(env)
        
        # 1. Timing Configuration
        self._physics_dt = physics_dt
        self._spline_freq = spline_freq
        self._spline_dt = 1.0 / spline_freq
        self._pd_steps = int(self._spline_dt / self._physics_dt)
        
        # 2. Control Constants
        self._vmax = vmax
        self._amax = amax
        self._kp = kp
        self._kd = kd

        # 3. Map Actuators to State Indices
        mj_model = env.mj_model
        n_act = mj_model.nu
        
        qpos_idx = []
        qvel_idx = []
        joint_limits = []

        for i in range(n_act):
            joint_id = mj_model.actuator_trnid[i, 0]
            qpos_idx.append(mj_model.jnt_qposadr[joint_id])
            qvel_idx.append(mj_model.jnt_dofadr[joint_id])
            joint_limits.append(mj_model.jnt_range[joint_id])

        self._qpos_idx = jp.array(qpos_idx)
        self._qvel_idx = jp.array(qvel_idx)
        self._joint_limits = jp.array(joint_limits)

    @property
    def action_size(self) -> int:
        return self.env.action_size * 2

    def _scale_action(self, action: jax.Array, p0: jax.Array, v0: jax.Array) -> Tuple[jax.Array, jax.Array]:
        n = len(self._qpos_idx)
        pos_raw = action[:n]
        vel_raw = action[n:]

        jmin = self._joint_limits[:, 0]
        jmax = self._joint_limits[:, 1]

        # Calculate relative target position constrained by vmax
        target_pos = p0 + pos_raw * self._vmax * self._spline_dt
        target_pos = jp.clip(target_pos, jmin, jmax)

        # Safety: max velocity to ensure we don't hit limits in this spline_dt
        dq = jp.minimum(jmax - target_pos, target_pos - jmin)
        vmax_pos = 2 * dq / self._spline_dt

        # Calculate relative target velocity constrained by amax
        target_vel = v0 + vel_raw * self._amax * self._spline_dt
        target_vel = jp.clip(target_vel, -vmax_pos, vmax_pos)

        return target_pos, target_vel

    def _hermite(self, p0, v0, p1, v1, tau):
        m0 = v0 * self._spline_dt
        m1 = v1 * self._spline_dt
        t2 = tau * tau
        t3 = t2 * tau

        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + tau
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2

        pos = h00*p0 + h10*m0 + h01*p1 + h11*m1

        dh00 = 6*t2 - 6*tau
        dh10 = 3*t2 - 4*tau + 1
        dh01 = -6*t2 + 6*tau
        dh11 = 3*t2 - 2*tau

        vel = (dh00*p0 + dh10*m0 + dh01*p1 + dh11*m1) / self._spline_dt
        return pos, vel

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Initial state at start of spline
        p0 = state.data.qpos[self._qpos_idx]
        v0 = state.data.qvel[self._qvel_idx]
        
        # Determine targets
        p1, v1 = self._scale_action(action, p0, v0)

        def sub_step_fn(carry, step_idx):
            current_state = carry
            
            # Progress along the spline (0.0 to 1.0)
            tau = (step_idx + 1) / self._pd_steps
            p_des, v_des = self._hermite(p0, v0, p1, v1, tau)
            
            # Current physical state
            cp = current_state.data.qpos[self._qpos_idx]
            cv = current_state.data.qvel[self._qvel_idx]
            
            # Low-level PD
            torque = self._kp * (p_des - cp) + self._kd * (v_des - cv)
            
            # Step the underlying environment (assumes env.step handles 1ms sim)
            next_state = self.env.step(current_state, torque)
            
            return next_state, next_state.reward

        # Run the internal 1kHz loop
        final_state, rewards = jax.lax.scan(
            sub_step_fn, state, jp.arange(self._pd_steps)
        )

        # Scale reward by physics_dt to match original logic (integral of reward)
        return final_state.replace(
            reward=jp.sum(rewards) * self._physics_dt
        )