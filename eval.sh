#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH -o /ibex/user/daiy0a/runlogs/%x_%J.out.txt
#SBATCH -e /ibex/user/daiy0a/runlogs/%x_%J.err.txt
#SBATCH --mail-user=yanning.dai@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=80G

# 接收参数：checkpoint 路径
vla_load_path=${1}

# 检查参数
if [ -z "$vla_load_path" ]; then
    echo "Error: Missing checkpoint path"
    echo "Usage: sbatch eval.sh <checkpoint_path>"
    exit 1
fi

eval "$(micromamba shell hook -s bash)"
micromamba activate rlvla_env

echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "Checkpoint: $vla_load_path"
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

cuda="0"
ckpt_path="gen-robot/openvla-7b-rlvla-warmup"

# start evaluation
for seed in 0 1 2; do
    for env_id in \
        "PutOnPlateInScene25VisionImage-v1" \
        "PutOnPlateInScene25VisionTexture03-v1" \
        "PutOnPlateInScene25VisionTexture05-v1" \
        "PutOnPlateInScene25VisionWhole03-v1" \
        "PutOnPlateInScene25VisionWhole05-v1" \
        "PutOnPlateInScene25Carrot-v1" \
        "PutOnPlateInScene25Plate-v1" \
        "PutOnPlateInScene25Instruct-v1" \
        "PutOnPlateInScene25MultiCarrot-v1" \
        "PutOnPlateInScene25MultiPlate-v1" \
        "PutOnPlateInScene25Position-v1" \
        "PutOnPlateInScene25EEPose-v1" \
        "PutOnPlateInScene25PositionChangeTo-v1"; do
        
        echo "Evaluating: env=${env_id}, seed=${seed}"
        
        CUDA_VISIBLE_DEVICES=$cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python simpler_env/train_ms3_ppo.py \
            --vla_path="${ckpt_path}" \
            --vla_load_path="${vla_load_path}" \
            --env_id="${env_id}" \
            --seed=${seed} \
            --buffer_inferbatch=64 \
            --no_wandb --only_render
    done
done

echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="

# for 40G GPU, set `--buffer_inferbatch=16` to avoid OOM