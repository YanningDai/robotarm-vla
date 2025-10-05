#!/bin/bash --login
#SBATCH --nodes=2
#SBATCH --partition=batch
#SBATCH -J vla-warmup
#SBATCH -o /ibex/user/daiy0a/runlogs/%J.out.txt
#SBATCH -e /ibex/user/daiy0a/runlogs/%J.err.txt
#SBATCH --mail-user=yanning.dai@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1

eval "$(micromamba shell hook -s bash)"
micromamba activate rlvla_env

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
hostname
nvidia-smi || true
python -V
which python

export OMP_NUM_THREADS="${SLURM_CPUS_PER_GPU:-10}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/ibex/user/$USER/.cache/huggingface"
export NCCL_ASYNC_ERROR_HANDLING=1

# 多节点配置
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)
export MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# 1. Train LoRA - 2节点，每节点2GPU，共4GPU
cd openvla || exit
task_name="warmup"

NUM_NODES=2
GPUS_PER_NODE=2
TOTAL_GPUS=4

echo "Training with $NUM_NODES nodes, $GPUS_PER_NODE GPUs per node, total $TOTAL_GPUS GPUs"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
srun torchrun \
  --nnodes "${NUM_NODES}" \
  --nproc-per-node "${GPUS_PER_NODE}" \
  --rdzv-backend c10d \
  --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "../datasets" \
  --dataset_name "${task_name}" \
  --run_root_dir "checkpoints/${task_name}" \
  --lora_rank 32 \
  --batch_size 20 \
  --max_steps 2000 \
  --eval_steps 50 \
  --save_steps "0,500,1000,1500,2000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --unnorm_key="bridge_orig" \
  --wandb_project "RLVLA"

# 2. Merge LoRA (只在主节点运行)
if [ "${SLURM_PROCID}" -eq 0 ]; then
    cuda="0"
    task_name="warmup"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$cuda" \
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
      --vla_path "openvla/openvla-7b" \
      --run_path "checkpoints/${task_name}/steps_2000" \
      --lora_name "lora_002000"
    
    echo "Done: warm-up + merge"
fi