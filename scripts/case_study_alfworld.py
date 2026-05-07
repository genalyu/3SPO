#!/usr/bin/env python3
"""
Case Study: Interact with trained model on ALFWorld environment.

Usage:
    # Using a HuggingFace model (e.g., trained checkpoint saved in HF format):
    python scripts/case_study_alfworld.py \
        --model-path checkpoints/verl_agent_alfworld/3spo_qwen2.5_1.5b/global_step_100/actor/huggingface

    # Using a HuggingFace model ID (e.g., base model or uploaded checkpoint):
    python scripts/case_study_alfworld.py \
        --model-path Qwen/Qwen2.5-1.5B-Instruct

    # Control parameters:
    python scripts/case_study_alfworld.py \
        --model-path <path-or-id> \
        --max-steps 20 \
        --temperature 0.4 \
        --task-idx 0
"""

import os
import sys
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project imports ──────────────────────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import (
    build_alfworld_envs,
    alfworld_projection,
)


# ── Helper: extract <action> from model response ─────────────────
def parse_action(response: str) -> Optional[str]:
    """Extract action text from <action>...</action> tags."""
    m = re.search(r"<action>\s*(.*?)\s*</action>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: if no tags, return the whole response
    return response.strip()


def parse_think(response: str) -> Optional[str]:
    """Extract thinking text from <think>...</think> tags."""
    m = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ── Model wrapper using vLLM ────────────────────────────────────
class CaseStudyAgent:
    """Thin wrapper around vLLM LLM for step-by-step generation."""

    def __init__(
        self,
        model_path: str,
        max_response_length: int = 512,
        temperature: float = 0.4,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required. Install it via: pip install vllm"
            )

        print(f"[Agent] Loading model from: {model_path}")
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=4096,
        )
        print("[Agent] Model loaded successfully.")

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_response_length,
            stop=["</action>"],  # stop after action tag
        )

    def generate(self, prompt: str) -> str:
        """Generate a response for a single prompt string."""
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        text = outputs[0].outputs[0].text
        # Append the stop token back since vLLM strips it
        if not text.endswith("</action>"):
            text += "</action>"
        return text


# ── Environment setup ────────────────────────────────────────────
def build_env(
    env_num: int = 1,
    seed: int = 42,
    eval_dataset: str = "eval_in_distribution",
    task_idx: Optional[int] = None,
) -> AlfWorldEnvironmentManager:
    """Build ALFWorld environment manager for case study.

    Args:
        env_num: number of parallel environments (use 1 for case study)
        seed: random seed for env initialization
        eval_dataset: "eval_in_distribution" or "eval_out_of_distribution"
        task_idx: if specified, reset until we get this specific task index
    """
    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        "../agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
    )
    alf_config_path = os.path.abspath(alf_config_path)

    env_kwargs = {"eval_dataset": eval_dataset}
    resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0.0}

    raw_envs = build_alfworld_envs(
        alf_config_path=alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )

    from functools import partial
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "env": {
            "env_name": "alfworld/AlfredTWEnv",
            "seed": seed,
            "max_steps": 50,
            "history_length": 10,  # number of history steps to include
            "rollout": {"n": 1},
        }
    })

    projection_f = partial(alfworld_projection)
    env_manager = AlfWorldEnvironmentManager(raw_envs, projection_f, config)
    return env_manager


# ── Main case-study loop ─────────────────────────────────────────
def run_case_study(
    agent: CaseStudyAgent,
    env_manager: AlfWorldEnvironmentManager,
    max_steps: int = 50,
    verbose: bool = True,
) -> List[Dict]:
    """Run a single episode and collect step-by-step trace.

    Returns a list of dicts with keys:
        step, env_obs, prompt, response, think, action, done, reward, won
    """
    kwargs = {}
    obs, infos = env_manager.reset(kwargs)

    # Extract task description for logging
    task_desc = env_manager.tasks[0] if hasattr(env_manager, "tasks") else "unknown"

    trace = []
    done = False

    for step in range(max_steps):
        prompt = obs["text"][0]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Step {step + 1}")
            print(f"  Task: {task_desc}")
            print(f"{'='*60}")
            print(f"\n[ENV]\n{obs['anchor'][0]}")

        # ── Generate ─
        response = agent.generate(prompt)
        think = parse_think(response)
        action = parse_action(response)

        if verbose:
            if think:
                print(f"\n[THINK]\n{think}")
            print(f"\n[ACTION]\n{action}")

        # ── Step environment ──
        obs_next, rewards, dones, infos = env_manager.step([action])
        done = bool(dones[0])
        reward = float(rewards[0])
        won = bool(infos[0].get("won", False))

        if verbose:
            print(f"\n  Done: {done}  |  Won: {won}  |  Reward: {reward:.2f}")

        trace.append({
            "step": step + 1,
            "env_obs": obs["anchor"][0],
            "prompt": prompt,
            "response": response,
            "think": think,
            "action": action,
            "done": done,
            "reward": reward,
            "won": won,
        })

        obs = obs_next

        if done:
            break

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Episode finished. Task: {task_desc}")
        print(f"  Steps: {len(trace)}  |  Success: {trace[-1]['won']}")
        print(f"{'='*60}")

    env_manager.envs.close()
    return trace


# ── Output helpers ────────────────────────────────────────────────
def format_case_study_latex(trace: List[Dict], task_idx: int = 0) -> str:
    """Format trace as LaTeX for paper appendix (matching the screenshot style)."""
    lines = []
    lines.append(r"\begin{figure*}[t]")
    lines.append(r"\centering")
    lines.append(r"\fbox{\begin{minipage}{\linewidth}")
    lines.append(r"\textbf{Case Study: ALFWorld Task #" + str(task_idx + 1) + r"}\\")
    lines.append(r"\rule{\linewidth}{0.4pt}\\[0.5em]")

    for item in trace:
        step = item["step"]

        # Environment box
        lines.append(r"\noindent\fcolorbox{orange!30}{white}{\begin{minipage}{\linewidth}")
        lines.append(r"\textbf{Environment (Step " + str(step) + r")}\\")
        lines.append(r"\small\sffamily " + item["env_obs"].replace("_", r"\_") + r"\\")
        lines.append(r"\end{minipage}}\\[0.5em]")

        # Agent box
        lines.append(r"\noindent\fcolorbox{blue!30}{white}{\begin{minipage}{\linewidth}")
        lines.append(r"\textbf{Agent (Step " + str(step) + r")}\\")
        if item["think"]:
            lines.append(r"\small\sffamily \textcolor{green!60!black}{<think>}" + item["think"].replace("_", r"\_") + r"\textcolor{green!60!black}{</think>}" + r"\\")
        lines.append(r"\small\sffamily \textcolor{blue}{<action>}" + item["action"].replace("_", r"\_") + r"\textcolor{blue}{</action>}" + r"\\")
        lines.append(r"\end{minipage}}\\[0.5em]")

    lines.append(r"\end{minipage}}")
    lines.append(r"\end{figure*}")
    return "\n".join(lines)


def print_summary(trace: List[Dict]):
    """Print a compact summary of the episode."""
    won = trace[-1]["won"]
    steps = len(trace)
    action_seq = [item["action"] for item in trace]
    print(f"\n  {'✓' if won else '✗'} Success: {won}")
    print(f"  Steps: {steps}")
    print(f"  Action sequence: {' -> '.join(action_seq)}")


# ── Entry point ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Case study: trained model × ALFWorld")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model (HF-format checkpoint) or HF model ID",
    )
    parser.add_argument(
        "--max-steps", type=int, default=20,
        help="Maximum steps per episode (default: 20)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4,
        help="Sampling temperature (default: 0.4)",
    )
    parser.add_argument(
        "--task-idx", type=int, default=0,
        help="Which task index to run (default: 0)",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=3,
        help="Number of tasks to run sequentially (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="case_study_output",
        help="Directory to save trace JSONs and LaTeX (default: case_study_output)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        help="Model dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Initialize agent ──
    agent = CaseStudyAgent(
        model_path=args.model_path,
        temperature=args.temperature,
        dtype=args.dtype,
    )

    # ── Run multiple tasks ──
    all_traces = []
    for task_i in range(args.num_tasks):
        print(f"\n{'#'*60}")
        print(f"#  Running task {task_i + 1}/{args.num_tasks}")
        print(f"{'#'*60}")

        env_manager = build_env(env_num=1, seed=42 + task_i, task_idx=task_i)
        trace = run_case_study(
            agent=agent,
            env_manager=env_manager,
            max_steps=args.max_steps,
            verbose=True,
        )
        all_traces.append(trace)

        # ── Save trace as JSON ──
        import json
        trace_path = os.path.join(args.output_dir, f"trace_task_{task_i}.json")
        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        print(f"  Trace saved to: {trace_path}")

        # ── Generate LaTeX ──
        latex = format_case_study_latex(trace, task_idx=task_i)
        latex_path = os.path.join(args.output_dir, f"trace_task_{task_i}.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"  LaTeX saved to: {latex_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  All {args.num_tasks} tasks completed.")
    print(f"  Traces saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
