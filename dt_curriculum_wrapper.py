"""
Wrappers for dt-curriculum learning experiments.

Three wrappers are provided:

DtObsWrapper
    Appends a single normalised dt scalar to every observation component.
    The env's ctrl_dt is fixed at instantiation, so the feature is constant
    within a training phase.  Use this for the *sequential* curriculum where
    each phase creates a fresh env with a different ctrl_dt.

DtCurriculumWrapper
    Stochastic dt curriculum: at every episode reset a dt multiplier k is
    sampled from a configurable categorical distribution (k_probs).  The same
    action is then repeated k times per wrapper step (action repeat), giving
    an effective control frequency of k × base_ctrl_dt.  The normalised dt
    feature varies per episode so the policy learns to condition on it.

    The distribution k_probs is captured as JAX constants at instantiation.
    To shift the distribution between phases, create a new wrapper with
    updated k_probs (this invalidates JAX's JIT cache, triggering a one-time
    recompile).

ContinuousDtCurriculumWrapper
    Continuous dt curriculum: at every episode reset a real-valued multiplier
    k_float is drawn from a truncated log-uniform (or linear-uniform)
    distribution over [k_min, k_max_sample], then rounded to the nearest
    integer k.  The env is stepped k times via jax.lax.scan with reward
    masking.  The sampling upper bound k_max_sample is annealed per
    mini-phase (captured as a JAX constant at instantiation), while
    k_max_scan (the static scan length) stays fixed so the compiled graph
    is shared across all mini-phases without full recompilation.

Normalisation convention
    dt_norm = (ctrl_dt - dt_fine) / (dt_coarse - dt_fine)
    → 0.0 = finest (target) dt,  1.0 = coarsest dt.
"""

from typing import Dict, List, Mapping, Optional, Union

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _augment(obs, feat: jax.Array) -> Union[jax.Array, Dict[str, jax.Array]]:
    """Concatenate *feat* to the end of every leaf in *obs*."""
    if isinstance(obs, Mapping):
        return {k: jp.concatenate([v, feat]) for k, v in obs.items()}
    return jp.concatenate([obs, feat])


def _add_one(size: mjx_env.ObservationSize) -> mjx_env.ObservationSize:
    """Increment every leaf size by 1 (to account for the dt feature)."""
    if isinstance(size, dict):
        return {k: (v[0] + 1,) for k, v in size.items()}
    return size + 1


# ---------------------------------------------------------------------------
# DtObsWrapper – fixed dt per env instance
# ---------------------------------------------------------------------------

class DtObsWrapper(Wrapper):
    """Append a normalised dt feature (constant scalar) to all observations.

    The feature encodes the current env's control timestep on the scale
    [0, 1] where 0 = finest dt (target) and 1 = coarsest dt.

    Parameters
    ----------
    env:
        MuJoCo Playground environment to wrap.
    dt_fine:
        Finest control timestep in the curriculum (denominator for normalisation).
    dt_coarse:
        Coarsest control timestep in the curriculum.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        dt_fine: float,
        dt_coarse: float,
    ) -> None:
        super().__init__(env)
        self._dt_fine = dt_fine
        self._dt_coarse = dt_coarse
        dt_range = max(dt_coarse - dt_fine, 1e-9)
        dt_norm_val = float((env.dt - dt_fine) / dt_range)
        # Precompute as a JAX constant – captured at trace time.
        self._dt_feat = jp.array([dt_norm_val], dtype=jp.float32)

    # ------------------------------------------------------------------ API

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        return _add_one(self.env.observation_size)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        return state.replace(obs=_augment(state.obs, self._dt_feat))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        return state.replace(obs=_augment(state.obs, self._dt_feat))


# ---------------------------------------------------------------------------
# DtCurriculumWrapper – stochastic dt per episode
# ---------------------------------------------------------------------------

class DtCurriculumWrapper(Wrapper):
    """Stochastic dt curriculum via per-episode action repeat.

    At each episode reset a dt multiplier k is drawn from *k_probs*.
    The wrapped environment (which must use the *finest* ctrl_dt) is then
    stepped k times per wrapper step with the same action, realising an
    effective control frequency of k × base_ctrl_dt.

    The normalised dt is appended to all observation components so that the
    policy can condition on the current control frequency.

    Parameters
    ----------
    env:
        Base environment configured with the **finest** ctrl_dt.
    k_levels:
        Ordered list of integer dt multipliers, e.g. ``[4, 2, 1]``.
        Index 0 is the coarsest (largest k), the last index is the finest.
    k_probs:
        Sampling probabilities aligned with *k_levels*.  Defaults to uniform.
        These are captured as JAX constants; recreate the wrapper to update
        them between curriculum phases.
    dt_fine:
        Finest dt used for normalisation.  Defaults to ``env.dt`` (the base).
    dt_coarse:
        Coarsest dt used for normalisation.  Defaults to ``max(k_levels) * env.dt``.

    Notes
    -----
    ``jax.lax.switch`` traces **all** branches during compilation, so compile
    time scales with ``sum(k_levels)``.  Only the selected branch executes at
    runtime; dead branches are eliminated by XLA.

    The PPO discount factor is applied per *wrapper* step.  For consistent
    per-second discounting across k values set::

        gamma_wrapper ≈ gamma_fine_step ** k_expected

    where ``k_expected = sum(k * p for k, p in zip(k_levels, k_probs))``.
    The training script adjusts this automatically.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        k_levels: List[int] = [4, 2, 1],
        k_probs: Optional[List[float]] = None,
        dt_fine: Optional[float] = None,
        dt_coarse: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        self._k_levels = list(k_levels)
        self._n_levels = len(k_levels)

        if k_probs is None:
            k_probs = [1.0 / self._n_levels] * self._n_levels
        arr = jp.array(k_probs, dtype=jp.float32)
        self._k_probs = arr / arr.sum()

        base_dt = env.dt
        self._dt_fine = dt_fine if dt_fine is not None else base_dt
        self._dt_coarse = dt_coarse if dt_coarse is not None else max(k_levels) * base_dt
        # Precompute effective dt for each k level (JAX constant array).
        self._dt_levels = jp.array([k * base_dt for k in k_levels], dtype=jp.float32)

    # ------------------------------------------------------------------ API

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        return _add_one(self.env.observation_size)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        state = self.env.reset(rng)
        # Sample a dt level for this episode.
        k_idx = jax.random.choice(key, self._n_levels, p=self._k_probs)
        state.info["dt_k_idx"] = k_idx
        return state.replace(obs=_augment(state.obs, self._dt_feat(k_idx)))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        k_idx = state.info["dt_k_idx"]

        # Build one branch per k level.  Python-level loops are unrolled into
        # a static computation graph by JAX at trace time.
        branches = []
        for k in self._k_levels:
            def _make_branch(k_val: int):
                def _branch(sa):
                    s, a = sa
                    total_reward = jp.zeros(())
                    any_done = jp.zeros(())
                    for _ in range(k_val):
                        s = self.env.step(s, a)
                        total_reward = total_reward + s.reward
                        # Use maximum so done=1 latches once triggered.
                        any_done = jp.maximum(any_done, s.done)
                    return s.replace(reward=total_reward, done=any_done)
                return _branch
            branches.append(_make_branch(k))

        new_state = jax.lax.switch(k_idx, branches, (state, action))
        # Restore k_idx; env.step does not touch this key so it's unchanged,
        # but we're explicit here to satisfy JAX pytree structure checks.
        new_state.info["dt_k_idx"] = k_idx
        return new_state.replace(obs=_augment(new_state.obs, self._dt_feat(k_idx)))

    # ---------------------------------------------------------------- helpers

    def _dt_feat(self, k_idx: jax.Array) -> jax.Array:
        """Normalised dt feature for a given k_idx.  Shape (1,)."""
        dt = self._dt_levels[k_idx]
        dt_range = jp.maximum(self._dt_coarse - self._dt_fine, 1e-9)
        dt_norm = (dt - self._dt_fine) / dt_range
        return jp.reshape(dt_norm, (1,))


# ---------------------------------------------------------------------------
# Utility: compute expected k from a distribution (for discount adjustment)
# ---------------------------------------------------------------------------

def expected_k(k_levels: List[int], k_probs: List[float]) -> float:
    """Expected number of action repeats under k_probs."""
    return sum(k * p for k, p in zip(k_levels, k_probs))


# ---------------------------------------------------------------------------
# ContinuousDtCurriculumWrapper – continuous log/linear-uniform dt per episode
# ---------------------------------------------------------------------------

class ContinuousDtCurriculumWrapper(Wrapper):
    """Continuous dt curriculum via per-episode sampling from [k_min, k_max_sample].

    At each episode reset a real-valued multiplier is drawn from a truncated
    log-uniform (or linear-uniform) distribution over [k_min, k_max_sample],
    then rounded to the nearest integer k.  The base environment is then
    stepped k times per wrapper step using jax.lax.scan with reward masking,
    giving an effective control timestep of k × base_ctrl_dt.

    Design
    ------
    Two separate k bounds serve distinct roles:

      k_max_scan  (Python int, compile-time constant)
          Static upper bound for jax.lax.scan.  Fixed across ALL curriculum
          mini-phases so the scan graph is compiled once and reused, avoiding
          expensive full recompilations between phases.

      k_max_sample  (float, captured as JAX constant per phase)
          Current upper bound for the sampling distribution.  Must satisfy
          1 ≤ k_max_sample ≤ k_max_scan.  Recreate the wrapper with a new
          k_max_sample for each mini-phase and pass the previous phase's
          params via restore_params to warm-start.

    Parameters
    ----------
    env:
        Base environment using the **finest** ctrl_dt.
    k_max_scan:
        Static maximum action-repeat factor (Python int, e.g. 4).
    k_max_sample:
        Current sampling upper bound.  Defaults to k_max_scan (fully coarse).
    k_min:
        Minimum action-repeat factor (default 1 = finest dt).
    log_uniform:
        If True (default), sample log-uniformly over [k_min, k_max_sample],
        giving denser coverage near k_min.  If False, sample linearly.
    dt_fine:
        Finest dt for normalisation.  Defaults to ``env.dt``.
    dt_coarse:
        Coarsest dt for normalisation.  Defaults to ``k_max_scan × env.dt``.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        k_max_scan: int = 4,
        k_max_sample: Optional[float] = None,
        k_min: int = 1,
        log_uniform: bool = True,
        dt_fine: Optional[float] = None,
        dt_coarse: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        assert isinstance(k_max_scan, int), "k_max_scan must be a Python int"
        self._k_max_scan = k_max_scan
        self._k_max_sample = float(
            k_max_sample if k_max_sample is not None else k_max_scan
        )
        self._k_min = k_min
        self._log_uniform = log_uniform

        base_dt = env.dt
        self._dt_fine = dt_fine if dt_fine is not None else base_dt
        self._dt_coarse = (
            dt_coarse if dt_coarse is not None else k_max_scan * base_dt
        )

    # ------------------------------------------------------------------ API

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        return _add_one(self.env.observation_size)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_k = jax.random.split(rng)
        k = self._sample_k(rng_k)
        state = self.env.reset(rng)
        state.info["curriculum_k"] = k
        return state.replace(obs=_augment(state.obs, self._dt_feat(k)))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        k = state.info["curriculum_k"]

        def body(carry: tuple, i: jax.Array) -> tuple:
            s, cum_r, done_early = carry
            # Active if no done received yet AND within the sampled k
            should_step = ~done_early & (i < k)
            s_next = self.env.step(s, action)
            # Accumulate reward only for active substeps
            cum_r = cum_r + jp.where(should_step, s_next.reward, 0.0)
            # Latch: once done in an active step, stop advancing
            done_early = done_early | (
                should_step & s_next.done.astype(jp.bool_)
            )
            # Advance state only for active substeps
            s_out = jax.tree_util.tree_map(
                lambda a, b: jp.where(should_step, a, b), s_next, s
            )
            return (s_out, cum_r, done_early), None

        # Strip the dt feature appended by the previous reset/step so that the
        # initial scan carry has the same obs shape as what self.env.step()
        # returns (base obs, without the dt scalar).  Without this, the first
        # body call would try to jp.where between obs of shape (N,) and (N+1,),
        # raising a broadcast error.
        if isinstance(state.obs, dict):
            inner_obs = {ky: v[:-1] for ky, v in state.obs.items()}
        else:
            inner_obs = state.obs[:-1]
        inner_state = state.replace(obs=inner_obs)

        # k_max_scan is a Python int → scan length is a compile-time constant.
        # Changing k_max_sample between phases does NOT change the scan length,
        # so the compiled lax.scan graph is shared across all mini-phases.
        (final_state, total_reward, _), _ = jax.lax.scan(
            body,
            (inner_state, jp.zeros(()), jp.zeros((), dtype=jp.bool_)),
            jp.arange(self._k_max_scan),
        )

        final_state.info["curriculum_k"] = k
        return final_state.replace(
            obs=_augment(final_state.obs, self._dt_feat(k)),
            reward=total_reward,
        )

    # ---------------------------------------------------------------- helpers

    def _sample_k(self, rng: jax.Array) -> jax.Array:
        """Sample integer k from [k_min, k_max_sample] (continuous → rounded).

        Log-uniform (default):  k_float = k_min * (k_max/k_min)^u,  u ~ U(0,1)
        Linear-uniform:         k_float = k_min + u * (k_max_sample - k_min)
        """
        u = jax.random.uniform(rng)
        k_min_f = jp.array(float(self._k_min), dtype=jp.float32)
        k_max_f = jp.array(float(self._k_max_sample), dtype=jp.float32)

        if self._log_uniform and self._k_max_sample > self._k_min:
            k_float = jp.exp(
                jp.log(k_min_f) + u * (jp.log(k_max_f) - jp.log(k_min_f))
            )
        else:
            k_float = k_min_f + u * (k_max_f - k_min_f)

        k_int = jp.round(k_float).astype(jp.int32)
        return jp.clip(k_int, self._k_min, self._k_max_scan)

    def _dt_feat(self, k: jax.Array) -> jax.Array:
        """Normalised dt feature for a given integer k.  Shape (1,)."""
        dt = k.astype(jp.float32) * self.env.dt
        dt_range = jp.maximum(self._dt_coarse - self._dt_fine, 1e-9)
        dt_norm = (dt - self._dt_fine) / dt_range
        return jp.reshape(jp.clip(dt_norm, 0.0, 1.0), (1,))
