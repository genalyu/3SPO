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

import torch
import numpy as np
from typing import List, Dict, Any
from .core_3spo import StateScoreManager, compute_3spo_step_reward, to_hashable
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

class SPO3TrajectoryCollector(TrajectoryCollector):
    """
    Extended TrajectoryCollector for 3SPO (State-Score-Supervised Policy Optimization).
    Implements step-level GRPO with composite rewards and adaptive rollouts.
    """
    def __init__(self, config, tokenizer, processor=None, state_manager: StateScoreManager = None):
        super().__init__(config, tokenizer, processor)
        self.state_manager = state_manager or StateScoreManager()

    def spo3_dfs_rollout(self, envs, actor_rollout_wg, gen_batch, max_depth=50, max_samples_per_prompt=32):
        """
        Implementation of 3SPO's Ranked Backtracking DFS.
        1. Explores actions by reward rank.
        2. When a trajectory ends, it backtracks to the last decision point 
           to explore the NEXT best action.
        3. Uses StateScoreManager to skip already explored actions across trajectories.
        """
        initial_obs, _ = envs.reset()
        
        # Frontier: stack of [ (state, ranked_actions_list, current_action_idx) ]
        # Each entry represents a decision point in the current DFS path.
        frontier = []
        all_trajectories_data = []
        samples_collected = 0
        
        # Start DFS from initial state
        current_state = initial_obs
        current_path = [] # Current trajectory segments
        
        while samples_collected < max_samples_per_prompt:
            h_s = str(to_hashable(current_state['anchor'][0]))
            
            # 1. If this state is new in current path, generate and rank actions
            # (In a real system, you'd only do this if it's not already in the stack)
            n = self.state_manager.get_n_rollouts(h_s)
            
            # A. Generate n actions
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=current_state)
            batch_output = actor_rollout_wg.generate_sequences(batch) # Simplified: should generate n
            actions = self.tokenizer.batch_decode(batch_output.batch['responses'], skip_special_tokens=True)
            
            # B. Get immediate 3SPO rewards for ranking
            next_obs, rewards, dones, _ = envs.step(actions)
            step_rewards = []
            for i in range(len(actions)):
                r = compute_3spo_step_reward(
                    s_t_obs=current_state['anchor'][0],
                    s_next_obs=next_obs['anchor'][i],
                    r_osworld=rewards[i],
                    state_manager=self.state_manager
                )
                step_rewards.append(r)
            
            # C. Rank actions by reward (Descending)
            ranked_indices = np.argsort(step_rewards)[::-1]
            
            # Filter out already globally explored actions for this state
            unexplored_indices = [idx for idx in ranked_indices if not self.state_manager.is_action_explored(h_s, actions[idx])]
            
            if not unexplored_indices:
                # All actions at this state are explored, backtrack
                if not frontier: break
                current_state, unexplored_indices, path_len = frontier.pop()
                current_path = current_path[:path_len]
                continue

            # D. Push to frontier (save state and unexplored branches)
            # We save the next best indices for later backtracking
            best_idx = unexplored_indices[0]
            if len(unexplored_indices) > 1:
                frontier.append((current_state, unexplored_indices[1:], len(current_path)))
            
            # E. Move forward with the BEST action
            self.state_manager.mark_action_explored(h_s, actions[best_idx])
            
            segment = {
                "state": current_state,
                "action": actions[best_idx],
                "reward": step_rewards[best_idx],
                "is_done": dones[best_idx]
            }
            current_path.append(segment)
            
            if dones[best_idx]:
                # Trajectory ended!
                samples_collected += 1
                all_trajectories_data.append(list(current_path))
                
                # Update global stats
                is_success = rewards[best_idx] > 0
                self.state_manager.update_statistics([s["state"]['anchor'][0] for s in current_path], is_success)
                
                # BACKTRACK: Try the next best action from the last decision point
                if not frontier: break
                current_state, unexplored_indices, path_len = frontier.pop()
                current_path = current_path[:path_len]
            else:
                # Continue deeper from the best next state
                current_state = {
                    'anchor': [next_obs['anchor'][best_idx]],
                    'text': [next_obs['text'][best_idx]] if next_obs['text'] is not None else None
                }
                
        return all_trajectories_data
