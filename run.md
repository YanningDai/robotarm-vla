# 路径-workstation
cd ~/ibex/data/vla/RL4VLA/
# 路径-ibex
cd /ibex/user/daiy0a/vla/RL4VLA/
sbatch warmup.sh
sbatch ppo_no_warmup.sh
sbatch ppo.sh

# 安装和数据准备：
/ibex/user/daiy0a/micromamba/envs/rlvla_env/bin/python

目前数据用workstation就已经全部采完了，速度很快。所以没在ibex上面装这部分的环境。只装训练环境：

```shell
micromamba create -y -n rlvla_env python=3.10
micromamba activate rlvla_env

# 下面和官方的是一样的
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd openvla && pip install -e . && cd ..
pip install -U tyro
pip install datasets==3.3.2
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..
```