"""
custom_env.py
=============
Defines CustomHighwayEnv — a subclass of HighwayEnv with a richer reward
function — and registers it as "custom-highway-v0" with gymnasium.

Also exports MLP and observation constants (OBS_DIM, N_ACTIONS) so that
subprocesses only need to import this one file.

All three experiments (CMA-ES, DQN, PPO) should import this file so they
share an identical environment. Just do:

    import custom_env          # triggers registration
    env = gym.make("custom-highway-v0")

Reward terms
------------
  collision_term            large penalty on crash
  speed_term                reward for driving near desired speed
  acceleration_term         penalty for large physical acceleration
  jerk_term                 penalty for change in acceleration (smoothness)
  lane_change_term          penalty for lane-change actions
  action_acceleration_term  penalty for FASTER / SLOWER actions
  stable_speed_term         bonus for near-constant speed

All terms are zeroed out when the vehicle leaves the road.
"""

import numpy as np
from gymnasium.envs.registration import register

from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv


# ---------------------------------------------------------------------------
# Observation / action constants
# ---------------------------------------------------------------------------

OBS_VEHICLES = 15
OBS_FEATURES = 5   # presence, x, y, vx, vy
OBS_DIM      = OBS_VEHICLES * OBS_FEATURES  # 75
N_ACTIONS    = 5


# ---------------------------------------------------------------------------
# Policy network — lives here so subprocesses only need to import custom_env
# ---------------------------------------------------------------------------

class MLP:
    """
    Two-layer MLP: obs → hidden (tanh) → logits → argmax action.
    Weights are stored as a single flat numpy array for CMA-ES.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        self.shapes = [
            (obs_dim, hidden_dim),    # W1
            (hidden_dim,),            # b1
            (hidden_dim, n_actions),  # W2
            (n_actions,),             # b2
        ]
        self.n_params = sum(int(np.prod(s)) for s in self.shapes)

    def unpack(self, weights: np.ndarray):
        params, idx = [], 0
        for shape in self.shapes:
            size = int(np.prod(shape))
            params.append(weights[idx: idx + size].reshape(shape))
            idx += size
        return params

    def forward(self, obs: np.ndarray, weights: np.ndarray) -> int:
        W1, b1, W2, b2 = self.unpack(weights)
        x = np.tanh(obs.flatten() @ W1 + b1)
        return int(np.argmax(x @ W2 + b2))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomHighwayEnv(HighwayEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # --- traffic ---
                "vehicles_density": 1.0,

                # --- observation ---
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": False,
                    "order": "sorted",
                    "normalize": True,
                },

                # --- episode ---
                "duration": 50,              # steps
                "simulation_frequency": 15,   # Hz
                "policy_frequency": 2,        # Hz

                # --- reward weights ---
                "collision_reward":             -10.0,
                "speed_reward_weight":            0.60,
                "lane_change_penalty":            0.12,
                "acceleration_penalty_weight":    0.04,
                "jerk_penalty_weight":            0.08,
                "action_acceleration_penalty":    0.05,
                "stable_speed_bonus":             0.03,
                "reward_speed_range":           [20, 30],
            }
        )
        return config

    def _reset(self) -> None:
        super()._reset()
        self.prev_speed = self.vehicle.speed
        self.prev_acceleration = 0.0
        self._last_reward_terms = {}

    def _reward(self, action: int) -> float:
        # --- collision ---
        collision_term = (
            self.config["collision_reward"] if self.vehicle.crashed else 0.0
        )

        # --- speed ---
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed,
            self.config["reward_speed_range"],
            [0, 1],
        )
        speed_term = self.config["speed_reward_weight"] * np.clip(scaled_speed, 0.0, 1.0)

        # --- acceleration & jerk ---
        dt = 1.0 / self.config["policy_frequency"]
        acceleration = (self.vehicle.speed - self.prev_speed) / dt
        acceleration_term = -self.config["acceleration_penalty_weight"] * abs(acceleration)

        jerk = (acceleration - self.prev_acceleration) / dt
        jerk_term = -self.config["jerk_penalty_weight"] * abs(jerk)

        # --- lane change ---
        lane_left  = self.action_type.actions_indexes["LANE_LEFT"]
        lane_right = self.action_type.actions_indexes["LANE_RIGHT"]
        faster     = self.action_type.actions_indexes["FASTER"]
        slower     = self.action_type.actions_indexes["SLOWER"]

        lane_change_term = (
            -self.config["lane_change_penalty"]
            if action in [lane_left, lane_right]
            else 0.0
        )

        # --- action acceleration ---
        action_acceleration_term = (
            -self.config["action_acceleration_penalty"]
            if action in [faster, slower]
            else 0.0
        )

        # --- stable speed ---
        stable_speed_term = (
            self.config["stable_speed_bonus"] if abs(acceleration) < 0.5 else 0.0
        )

        reward = (
            collision_term
            + speed_term
            + acceleration_term
            + jerk_term
            + lane_change_term
            + action_acceleration_term
            + stable_speed_term
        )

        # Update state for next step
        self.prev_speed = self.vehicle.speed
        self.prev_acceleration = acceleration

        # Zero out everything if off-road
        on_road = float(self.vehicle.on_road)
        self._last_reward_terms = {
            "collision":    collision_term * on_road,
            "speed":        speed_term * on_road,
            "acceleration": acceleration_term * on_road,
            "jerk":         jerk_term * on_road,
            "lane_change":  lane_change_term * on_road,
            "stable_speed": stable_speed_term * on_road,
        }
        return float(reward * on_road)

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        info["reward_terms"] = self._last_reward_terms
        return info

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]


# ---------------------------------------------------------------------------
# Registration — happens once when this module is imported
# ---------------------------------------------------------------------------

register(
    id="custom-highway-v0",
    entry_point="custom_env:CustomHighwayEnv",
)