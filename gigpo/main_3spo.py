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
    
    # 1. 基础组件初始化
    local_path = config.actor_rollout_ref.model.path
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path)
    
    # 2. 初始化 3SPO 核心状态管理器
    state_manager = StateScoreManager(
        alpha=config.algorithm.get("spo3_alpha", 1.0),
        xi=config.algorithm.get("spo3_xi", 10.0),
        zeta=config.algorithm.get("spo3_zeta", 0.1)
    )
    
    # 3. 【关键修改】通过 Monkey Patch 注入 3SPO 轨迹收集器
    # verl 内部会动态导入 TrajectoryCollector，我们将其替换为 3SPO 版本
    # 这样 run_ppo 内部创建 TaskRunner 时，Worker 就会自动使用 3SPO 逻辑
    import agent_system.multi_turn_rollout.rollout_loop as rollout_module
    
    def spo3_collector_factory(*args, **kwargs):
        # 强制注入 state_manager
        kwargs['state_manager'] = state_manager
        return SPO3TrajectoryCollector(*args, **kwargs)
    
    # 动态替换类定义
    rollout_module.TrajectoryCollector = spo3_collector_factory
    
    print("Starting 3SPO Training with Step-level GRPO (Monkey Patched)...")
    run_ppo(config)

if __name__ == "__main__":
    main()
