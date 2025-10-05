#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH -J vla-warmup
#SBATCH -o /ibex/user/daiy0a/runlogs/%J.out.txt
#SBATCH -e /ibex/user/daiy0a/runlogs/%J.err.txt
#SBATCH --mail-user=yanning.dai@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=80G

eval "$(micromamba shell hook -s bash)"
micromamba activate rlvla_env

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
hostname
nvidia-smi || true
python -V
which python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-10}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/ibex/user/$USER/.cache/huggingface
export NCCL_ASYNC_ERROR_HANDLING=1


# 1. Train LoRA
cd openvla
cuda="0,1,2,3"
task_name="warmup"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
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

# for 80G GPU, max batch size is 20
# for 40G GPU, max batch size is 8

# 2. Merge LoRA
cuda="0"
task_name="warmup"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/steps_2000" \
  --lora_name "lora_002000"


echo "Done: warm-up + merge"