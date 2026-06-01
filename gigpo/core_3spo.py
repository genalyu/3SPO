# Copyright 2026 The 3SPO Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from collections import defaultdict
import uuid
from typing import Dict, Any, List, Optional
from .core_gigpo import to_hashable

import json
import os

class StateScoreManager:
    """
    Manages historical interaction statistics and computes Dynamic State-Score S(st).
    Supports persistent storage to share state statistics globally across tasks and trajectories.
    """
    def __init__(self,
                 alpha: float = 1.0,
                 xi: float = 10.0,
                 zeta: float = 0.1,
                 epsilon: float = 1e-6,
                 g_max: int = 8,
                 persistence_path: str = "state_stats.json"):
        self.alpha = alpha
        self.xi = xi
        self.zeta = zeta
        self.epsilon = epsilon
        self.g_max = g_max
        self.persistence_path = persistence_path
        
        # Load existing stats if file exists
        if os.path.exists(self.persistence_path):
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
                self.stats = defaultdict(lambda: {"n_total": 0, "n_success": 0}, data.get("stats", {}))
                self.explored_actions = defaultdict(set, {k: set(v) for k, v in data.get("explored_actions", {}).items()})
        else:
            self.stats = defaultdict(lambda: {"n_total": 0, "n_success": 0})
            self.explored_actions = defaultdict(set)
            
        self.t = sum(s["n_total"] for s in self.stats.values()) + 1

    def save_to_disk(self):
        """Persist statistics and explored actions to a JSON file."""
        data = {
            "stats": dict(self.stats),
            "explored_actions": {k: list(v) for k, v in self.explored_actions.items()}
        }
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f)

    def mark_action_explored(self, state_hash: str, action_str: str):
        """Record that this action has been explored for this state."""
        self.explored_actions[state_hash].add(action_str)

    def is_action_explored(self, state_hash: str, action_str: str) -> bool:
        """Check if this action has already been explored for this state."""
        return action_str in self.explored_actions[state_hash]

    def update_statistics(self, trajectory_obs: List[Any], is_final_success: bool):
        """
        Update N_total and N_success for all states. 
        States are shared across all tasks/trajectories based on their hash (GiGPO style).
        """
        seen_states = set()
        for obs in trajectory_obs:
            h = str(to_hashable(obs)) # JSON keys must be strings
            if h not in seen_states:
                self.stats[h]["n_total"] += 1
                if is_final_success:
                    self.stats[h]["n_success"] += 1
                seen_states.add(h)
        self.t += 1
        self.save_to_disk() # Ensure persistence

    def get_score(self, state_hash) -> float:
        """
        Compute S(st) according to Equation 1.
        """
        stat = self.stats[state_hash]
        n_total = stat["n_total"]
        n_success = stat["n_success"]
        n_fail = n_total - n_success
        
        if n_total == 0:
            return 1.0  # Learning potential is high for unseen states
        
        success_rate = n_success / (n_total + self.epsilon)
        lambda_t = self.alpha * np.log(self.t)
        
        # Indicator function: (N_fail < xi) OR (SuccessRate > zeta)
        indicator = (n_fail < self.xi) or (success_rate > self.zeta)
        
        if not indicator:
            return 0.0
        
        # S(st) = exp(-lambda * SuccessRate)
        score = np.exp(-lambda_t * success_rate)
        return float(score)

    def get_n_rollouts(self, state_hash) -> int:
        """
        Compute n(st) according to Equation 3.
        n(st) = ceil(G_max * S(st))
        n=0 truncates the trajectory, n=1 proceeds without policy optimization.
        """
        score = self.get_score(state_hash)
        n = int(np.ceil(self.g_max * score))
        return n

    def get_omega(self, state_hash, omega_k: float = 0.1) -> float:
        """
        Compute w(N_total(st)) used in Equation 2.
        w(N) = 0.5 * exp(-omega_k * N)
        """
        n_total = self.stats[state_hash]["n_total"]
        return float(0.5 * np.exp(-omega_k * n_total))

def compute_3spo_step_reward(
    s_t_obs: Any,
    s_next_obs: Any,
    r_success: float,
    state_manager: StateScoreManager,
    omega_k: float = 0.1
) -> float:
    """
    Compute the step-wise composite reward R_3SPO(st, st+1).
    Equation 2:
    R_3SPO = w(N)*R_novel + (0.5 - w(N))*(S(st) - S(st+1)) + 0.5*R_success
    """
    h_t = to_hashable(s_t_obs)
    h_next = to_hashable(s_next_obs)

    # R_novel: 1 if state changed, else 0
    r_novel = 1.0 if h_t != h_next else 0.0

    w = state_manager.get_omega(h_t, omega_k=omega_k)
    s_t = state_manager.get_score(h_t)
    s_next = state_manager.get_score(h_next)

    reward = w * r_novel + (0.5 - w) * (s_t - s_next) + 0.5 * r_success
    return reward

def compute_3spo_advantage(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True
):
    """
    Compute step-level advantages for 3SPO using the grouped rewards.
    Similar to GiGPO's step_norm_reward but specifically for 3SPO rewards.
    """
    from .core_gigpo import step_norm_reward
    return step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)
