# Copyright 2026 The 3SPO Team.
import hydra
from verl.trainer.main_ppo import run_ppo
from gigpo.trainer_3spo_extension import SPO3TrajectoryCollector, StateScoreManager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import hf_tokenizer, hf_processor
from verl.trainer.ppo.reward import load_reward_manager
from agent_system.environments import make_envs
from omegaconf import OmegaConf
import ray

@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    # run_ppo will handle ray initialization with the correct runtime_env
    # which is critical for MIG environments.
    
    # (In a real scenario, we would subclass RayPPOTrainer or pass this collector)
    # The current run_ppo implementation will create its own envs inside TaskRunner.
    # We comment out the manual env creation here to avoid double-allocation of resources
    # in limited CPU/GPU environments (like SLURM nodes).
    
    # envs, val_envs = make_envs(config)
    
    # Initialize 3SPO specific state manager
    # state_manager = StateScoreManager(
    #     alpha=config.algorithm.get("spo3_alpha", 1.0),
    #     xi=config.algorithm.get("spo3_xi", 10.0),
    #     zeta=config.algorithm.get("spo3_zeta", 0.1)
    # )
    
    # Initialize 3SPO trajectory collector
    # traj_collector = SPO3TrajectoryCollector(
    #     config=config, 
    #     tokenizer=tokenizer, 
    #     processor=processor,
    #     state_manager=state_manager
    # )
    
    print("Starting 3SPO Training with Step-level GRPO...")
    run_ppo(config) # This would need further integration to use the custom collector

if __name__ == "__main__":
    main()
