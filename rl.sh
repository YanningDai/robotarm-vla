#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH -o /ibex/user/daiy0a/runlogs/%J.out.txt
#SBATCH -e /ibex/user/daiy0a/runlogs/%J.err.txt
#SBATCH --mail-user=yanning.dai@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=56:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=80G
#SBATCH --account=conf-icml-2026.01.29-schmidhj

alg_name=${1:-"ppo"}
trust_type=${2:-"clip"}
rollback_alpha=${3:-0.0}
trust_region_delta=${4:-0.0}
seed=${5:-0}
vla_load_path=${6:-""}

exp_name="${alg_name},${trust_type},rollback_alpha=${rollback_alpha},trust_region_delta=${trust_region_delta},seed=${seed}"

eval "$(micromamba shell hook -s bash)"
micromamba activate rlvla_env

echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "Experiment Name: $exp_name"
echo "Parameters:"
echo "  - alg_name: $alg_name"
echo "  - trust_type: $trust_type"
echo "  - rollback_alpha: $rollback_alpha"
echo "  - trust_region_delta: $trust_region_delta"
echo "  - seed: $seed"
echo "=========================================="

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
export NCCL_DEBUG=WARN
export HF_HOME=/ibex/user/$USER/.cache/huggingface
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1

cd SimplerEnv || exit

# cuda="0,1" # env on GPU-0, model on GPU-1 (for 40G GPU)
cuda="0" # env and model on the same GPU (for 80G GPU)

python_args=(
    "--name=${SLURM_JOB_NAME}"
    "--alg_name=${alg_name}"
    "--seed=${seed}"
    "--trust_type=${trust_type}"
    "--rollback_alpha=${rollback_alpha}"
    "--trust_region_delta=${trust_region_delta}"
)

if [ -n "$vla_load_path" ]; then
    python_args+=("--vla_load_path=${vla_load_path}")
fi


CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simpler_env/train_ms3_ppo.py "${python_args[@]}"


# "clip", "rollback", "truly", "trust_region"


# GRPO: add --alg_name="grpo"
# GRPO (s): add --alg_name="grpo" and --use_same_init

# 默认
# --vla_path="gen-robot/openvla-7b-rlvla-warmup" --vla_unnorm_key="bridge_orig" 模型加载位置
# --env_id="PutOnPlateInScene25Main-v3" 训练都是这个任务