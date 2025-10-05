#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH -J ppo_warmup
#SBATCH -o /ibex/user/daiy0a/runlogs/%J.out.txt
#SBATCH -e /ibex/user/daiy0a/runlogs/%J.err.txt
#SBATCH --mail-user=yanning.dai@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=56:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=40G

eval "$(micromamba shell hook -s bash)"
micromamba activate rlvla_env

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi || true
python -V
which python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-10}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
ulimit -n 65535 || true

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export HF_HOME=/ibex/user/$USER/.cache/huggingface
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

cd SimplerEnv

cuda="0,1" # env on GPU-0, model on GPU-1 (for 40G GPU)
#cuda="0" # env and model on the same GPU (for 80G GPU)

CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py \
  --name="PPO-pc25m_v3-warmup" \
  --env_id="PutOnPlateInScene25Main-v3" \
  --vla_path="openvla/openvla-7b" --vla_unnorm_key="bridge_orig" \
  --vla_load_path="../openvla/checkpoints/warmup/steps_2000/lora_002000" \
  --seed=0

# GRPO: add --alg_name="grpo"
# GRPO (s): add --alg_name="grpo" and --use_same_init
# PPO from scratch: remove --vla_load_path arg