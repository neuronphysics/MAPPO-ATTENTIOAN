#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=95000M   
#SBATCH --time=02:00:00
#SBATCH --account=def-gdumas85
#SBATCH --output=/home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/logs/MAPPO-attention-meltingpot-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/logs/MAPPO-attention-meltingpot-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load gcc python/3.10 opencv/4.7 mujoco mpi4py arrow/11
module load scipy-stack
module load rust/1.65.0
DIR=/home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN

#virtualenv --no-download --clear /home/memole/MAPPO
source /home/memole/MAPPO/bin/activate


CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
pip install --no-index --upgrade pip
#pip install --no-index --no-cache-dir numpy 
#pip install --no-index --no-cache-dir opencv-python
#pip install --no-index --no-cache-dir ml-collections
#pip install --no-index torch torchvision torchtext torchaudio
#pip install --no-index wandb
#pip install --no-cache-dir -r ~/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/requirements.txt
#python -m pip install git+https://github.com/mpi4py/mpi4py
#pip install --no-cache-dir mpyq
#pip install -U "ray[rllib]"
#

#pip install 'git+https://github.com/lcswillems/torch-ac.git'
#pip install 'git+https://github.com/IntelPython/mkl_fft.git'
#pip install 'git+https://github.com/IntelPython/mkl_random.git'


# install this package first
# install on-policy package
WANDB_CREDENTIALS_PATH='/home/memole/projects/def-gdumas85/memole/wandb_credentials.txt'
export WANDB_API_KEY=$(cat $WANDB_CREDENTIALS_PATH)
cd /home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/
pip install -e .
#install starcraft
#mkdir 3rdparty
export SC2PATH="/home/memole/projects/def-gdumas85/memole/MPPO-ATTENTIOAN/3rdparty/StarCraftII"

#cd 3rdparty
#echo 'SC2PATH is set to '$SC2PATH
#wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
#unzip -P iagreetotheeula SC2.4.10.zip
#rm -rf SC2.4.10.zip
#export MAP_DIR="/content/drive/MyDrive/MPPO-ATTENTIOAN/3rdparty/StarCraftII/Maps/"
#echo 'MAP_DIR is set to '$MAP_DIR
#mkdir -p $MAP_DIR
#cd ..
#wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
#unzip SMAC_Maps.zip
#mv SMAC_Maps $MAP_DIR
#rm -rf SMAC_Maps.zip

#Hanabi

echo "Install Hanabi...."
#cd $DIR/onpolicy/envs/hanabi/
#cmake -B _build -S .
#cmake --build _build
#python -c 'import pyhanabi'

# install on-policy package
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
##install this package first
##football environment
#python3 -m pip install --upgrade pip setuptools psutil wheel
#pip install --no-index --no-cache-dir gfootball

# install melting pot
#pip install dm-env
#pip install pygame
##install DeepMind Lab2D https://github.com/deepmind/lab2d
#wget https://files.pythonhosted.org/packages/4b/31/884879224de4627b5d45b307cec8f4cd1e60db9aa61871e4aa2518c6584b/dmlab2d-1.0.0_dev.10-cp310-cp310-manylinux_2_31_x86_64.whl -O dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl
#setrpaths.sh --path dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl 
#pip install dmlab2d-1.0.0_dev.10-cp310-cp310-linux_x86_64.whl 
#pip install --no-index libcst
#git clone -b main https://github.com/deepmind/meltingpot
#cd meltingpot
#pip install --editable .[dev]
#pip install  --no-index --no-cache-dir dm-acme
#pip install -U pip
#pip install "ray[cpp]" --no-index
#python -c 'from ray import cpp'
wandb login $WANDB_API_KEY
echo "Start running the train meltingpot script ..."


cd $DIR/onpolicy/scripts/train

CUDA_VISIBLE_DEVICES=0,1 python train_mpe.py --use_valuenorm --use_popart --env_name "MPE" --algorithm_name "mappo" --experiment_name "check" \
    --scenario_name "simple_spread" --num_agents 2 --num_landmarks 3 --seed 1 --use_render True --use_attention True \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 20 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb --user_name "zsheikhb" --wandb_name "zsheikhb" --share_policy

cd $DIR/onpolicy/scripts/render   
CUDA_VISIBLE_DEVICES=0 python render_mpe.py --save_gifs --share_policy --env_name "MPE" --algorithm_name "mappo" \
    --experiment_name "check" --scenario_name "simple_spread" --num_agents 2 --num_landmarks 3 --seed 1 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 25 \
    --model_dir "mpe_spread" --save_gifs True

#test meltingpot environment 
#CUDA_VISIBLE_DEVICES=0,1 /home/memole/MAPPO/bin/python3 -m meltingpot.examples.pettingzoo.sb3_train
#CUDA_VISIBLE_DEVICES=0,1 python train_meltingpot.py --use_valuenorm --use_popart --env_name "Meltingpot" --algorithm_name "mappo" --experiment_name "check" \
#    --substrate_name "territory__rooms" --num_agents 9 --seed 1 --use_render --n_rollout_threads 16\
#    --use_wandb --user_name "zsheikhb" --wandb_name "zsheikhb" --share_policy False

