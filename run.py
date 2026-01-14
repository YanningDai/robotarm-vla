#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

# ibex 训练
# cd /ibex/user/daiy0a/vla/RL4VLA/
# python run.py



# ============================================================
# 训练任务配置
# 格式: (alg_name, trust_type, rollback_alpha, trust_region_delta, seed, vla_load_path)
# vla_load_path 为空字符串 "" 表示不使用 (训练新模型)
# ============================================================
TRAIN_EXPERIMENTS = [
    # PPO with clip (baseline)
    # ("ppo", "clip", 0.0, 0.0, 0, ""),
    
    # PPO with rollback
    # ("ppo", "rollback", 0.001, 0.0, 0, ""),
    # ("ppo", "rollback", 0.005, 0.0, 0, ""),
    # ("ppo", "rollback", 0.01, 0.0, 0, ""),
    # ("ppo", "rollback", 0.05, 0.0, 0, ""),
    # ("ppo", "rollback", 0.1, 0.0, 0, ""),

    # ("ppo", "rollback", 0.5, 0.0, 0, ""),
    # ("ppo", "rollback", 1.0, 0.0, 0, ""),
    # ("ppo", "rollback", 5.0, 0.0, 0, ""),
    # ("ppo", "rollback", 10.0, 0.0, 0, ""),
    # ("ppo", "rollback", 20.0, 0.0, 0, ""),
    # ("ppo", "rollback", 50.0, 0.0, 0, ""),
    
    # PPO with trust_region
    # ("ppo", "trust_region", 0.0, 0.0001, 0, ""),
    # ("ppo", "trust_region", 0.0, 0.001, 0, ""),
    # ("ppo", "trust_region", 0.0, 0.01, 0, ""),
    # ("ppo", "trust_region", 0.0, 0.1, 0, ""),
    # ("ppo", "trust_region", 0.0, 0.5, 0, ""),
    # ("ppo", "trust_region", 0.0, 1.0, 0, ""),
    # ("ppo", "trust_region", 0.0, 5.0, 0, ""),
    # ("ppo", "trust_region", 0.0, 10.0, 0, ""),
    
    # PPO with truly
    # ("ppo", "truly", 0.5, 0.1, 0, ""),
    # ("ppo", "truly", 0.5, 0.01, 0, ""),
    # ("ppo", "truly", 1.0, 0.01, 0, ""),
    # ("ppo", "truly", 2.0, 0.01, 0, ""),
    # ("ppo", "truly", 0.5, 0.05, 0, ""),
    # ("ppo", "truly", 1.0, 0.05, 0, ""),
    # ("ppo", "truly", 2.0, 0.05, 0, ""),
    # ("ppo", "truly", 0.1, 0.1, 0, ""),
    
    
    # ("ppo", "truly", 0.1, 0.5, 0, ""),
    # ("ppo", "truly", 0.5, 0.5, 0, ""),
    # ("ppo", "truly", 1.0, 0.5, 0, ""),
    # ("ppo", "truly", 2.0, 0.5, 0, ""),
    # ("ppo", "truly", 2.0, 0.1, 0, ""),
    
    
    # Multiple seeds
    # ("ppo", "clip", 0.0, 0.0, 1, ""),
    
    # GRPO (uncomment if needed)
    # ("grpo", "clip", 0.0, 0.0, 0, ""),
    
    # Resume examples
    # ("ppo", "clip", 0.0, 0.0, 0, "../SimplerEnv/outputs/ppo,clip,rollback_alpha=0.0,trust_region_delta=0.0,seed=0/steps_0239"),
    # ("ppo", "truly", 0.1, 0.1, 0, "../SimplerEnv/outputs/ppo,truly,rollback_alpha=0.1,trust_region_delta=0.1,seed=0/steps_0239"),
    # ("ppo", "truly", 0.5, 0.1, 0, "../SimplerEnv/outputs/ppo,truly,rollback_alpha=0.5,trust_region_delta=0.1,seed=0/steps_0239"),
    # ("ppo", "truly", 0.5, 0.5, 0, "../SimplerEnv/outputs/ppo,truly,rollback_alpha=0.5,trust_region_delta=0.5,seed=0/steps_0239"),
    # ("ppo", "truly", 1.0, 0.05, 0, "../SimplerEnv/outputs/ppo,truly,rollback_alpha=1.0,trust_region_delta=0.05,seed=0/steps_0239"),
    # ("ppo", "truly", 1.0, 0.5, 0, "../SimplerEnv/outputs/ppo,truly,rollback_alpha=1.0,trust_region_delta=0.5,seed=0/steps_0239"),
    # ("ppo", "truly", 2.0, 0.1, 0, "../SimplerEnv/outputs/ppo,truly,rollback_alpha=2.0,trust_region_delta=0.1,seed=0/steps_0239"),

]

# ============================================================
# 评估任务配置
# 格式: checkpoint 路径
# ============================================================
EVAL_CHECKPOINTS = [
    
    "../SimplerEnv/outputs/resume_ppo,clip,rollback_alpha=0.0,trust_region_delta=0.0,seed=0/steps_0159",  # standard ppo
    "../SimplerEnv/outputs/resume_ppo,truly,rollback_alpha=0.1,trust_region_delta=0.1,seed=0/steps_0159",
    "../SimplerEnv/outputs/resume_ppo,truly,rollback_alpha=0.5,trust_region_delta=0.1,seed=0/steps_0159",
    "../SimplerEnv/outputs/resume_ppo,truly,rollback_alpha=0.5,trust_region_delta=0.5,seed=0/steps_0159",
    "../SimplerEnv/outputs/resume_ppo,truly,rollback_alpha=1.0,trust_region_delta=0.05,seed=0/steps_0159",
    "../SimplerEnv/outputs/resume_ppo,truly,rollback_alpha=1.0,trust_region_delta=0.5,seed=0/steps_0159",
    "../SimplerEnv/outputs/resume_ppo,truly,rollback_alpha=2.0,trust_region_delta=0.1,seed=0/steps_0159",
]
# ============================================================


def submit_train_job(alg_name, trust_type, rollback_alpha, trust_region_delta, seed, vla_load_path=""):
    """提交一个训练任务"""
    
    base_job_name = f"{alg_name},{trust_type},rollback_alpha={rollback_alpha},trust_region_delta={trust_region_delta},seed={seed}"

    # 如果有 vla_load_path（resume），在前面加 "resume" 前缀
    if vla_load_path:
        job_name = f"resume_{base_job_name}"
    else:
        job_name = base_job_name
    
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "rl.sh",
        alg_name,
        trust_type,
        str(rollback_alpha),
        str(trust_region_delta),
        str(seed),
        vla_load_path  # 第 6 个参数
    ]
    
    print(f"Submitting training: {job_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  ✓ Job ID: {job_id}")
        return job_id
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return None


def submit_eval_job(checkpoint_path):
    """提交一个评估任务"""
    
    ckpt_name = Path(checkpoint_path).name
    second_last_dir = Path(checkpoint_path).parent.name
    job_name = f"eval_{ckpt_name}_{second_last_dir}"
    
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "eval.sh",
        checkpoint_path
    ]
    
    print(f"Submitting evaluation: {job_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  ✓ Job ID: {job_id}")
        return job_id
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return None


def main():
    # 确保日志目录存在
    Path("/ibex/user/daiy0a/runlogs").mkdir(parents=True, exist_ok=True)
    
    all_jobs = []
    
    # ============ 提交训练任务 ============
    if TRAIN_EXPERIMENTS:
        print("="*60)
        print(f"TRAINING JOBS: {len(TRAIN_EXPERIMENTS)} experiments\n")
        
        for i, exp in enumerate(TRAIN_EXPERIMENTS, 1):
            # 支持 5 元组（向后兼容）和 6 元组格式
            if len(exp) == 5:
                alg_name, trust_type, rollback_alpha, trust_region_delta, seed = exp
                vla_load_path = ""
            else:
                alg_name, trust_type, rollback_alpha, trust_region_delta, seed, vla_load_path = exp

            print(f"[Train {i}/{len(TRAIN_EXPERIMENTS)}]", end=" ")
            job_id = submit_train_job(alg_name, trust_type, rollback_alpha, trust_region_delta, seed, vla_load_path)
            if job_id:
                all_jobs.append(("train", job_id))
            time.sleep(0.5)
        
        print()
    
    # ============ 提交评估任务 ============
    if EVAL_CHECKPOINTS:
        print("="*60)
        print(f"EVALUATION JOBS: {len(EVAL_CHECKPOINTS)} checkpoints\n")
        
        for i, checkpoint_path in enumerate(EVAL_CHECKPOINTS, 1):
            print(f"[Eval {i}/{len(EVAL_CHECKPOINTS)}]", end=" ")
            job_id = submit_eval_job(checkpoint_path)
            if job_id:
                all_jobs.append(("eval", job_id))
            time.sleep(0.5)
        
        print()
    
    # ============ 总结 ============
    print("="*60)
    train_jobs = [jid for jtype, jid in all_jobs if jtype == "train"]
    eval_jobs = [jid for jtype, jid in all_jobs if jtype == "eval"]
    
    print(f"✓ Successfully submitted:")
    print(f"  - Training jobs: {len(train_jobs)}")
    print(f"  - Evaluation jobs: {len(eval_jobs)}")
    print(f"  - Total: {len(all_jobs)}")
    
    if train_jobs:
        print(f"\nTraining Job IDs: {', '.join(train_jobs)}")
    if eval_jobs:
        print(f"Evaluation Job IDs: {', '.join(eval_jobs)}")
    
    print(f"\nCheck status with: squeue -u $USER")
    print("="*60)


if __name__ == "__main__":
    main()