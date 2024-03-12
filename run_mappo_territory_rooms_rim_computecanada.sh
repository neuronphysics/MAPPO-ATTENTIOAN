#!/bin/bash
#SBATCH --job-name=MAPPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G                                        # Ask for 10 GB of RAM
#SBATCH --time=4-23:59:59  
#SBATCH --account=def-gdumas85
#SBATCH --output=/home/memole/projects/def-gdumas85/memole/SkillDynamics/MPPO-ATTENTIOAN/logs/MAPPO-attention-meltingpot-territory_rooms-seed-1_%N-%j.out
#SBATCH --error=/home/memole/projects/def-gdumas85/memole/SkillDynamics/MPPO-ATTENTIOAN/logs/MAPPO-attention-meltingpot-territory_rooms-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load gcc python/3.10 opencv/4.7 mujoco mpi4py arrow/11
module load scipy-stack
module load rust/1.65.0
DIR=/home/memole/projects/def-gdumas85/memole/SkillDynamics/MPPO-ATTENTIOAN

#virtualenv --no-download --clear /home/memole/MAPPO
source /home/memole/MAPPO/bin/activate


CURRENT_PATH=`pwd`
echo "current path ---> $CURRENT_PATH"
pip install --no-index --upgrade pip
pip install --no-index --upgrade easydict
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
cd /home/memole/projects/def-gdumas85/memole/SkillDynamics/MPPO-ATTENTIOAN
pip install -e .
#install starcraft
#mkdir 3rdparty
#export SC2PATH="/home/memole/projects/def-gdumas85/memole/MARL_SKILL/MPPO-ATTENTIOAN/3rdparty/StarCraftII"

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


seed=$1

echo "seed ---> $seed"



wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42
echo "Start running the train meltingpot script ..."


cd $CURRENT_PATH/onpolicy/scripts/train


CUDA_VISIBLE_DEVICES=0 python train_meltingpot.py --use_valuenorm False --use_popart True --env_name "Meltingpot" \
    --algorithm_name "mappo" --experiment_name "check" \
    --substrate_name "territory__rooms" --num_agents 9 --seed $seed --lr 0.0001 --n_rollout_threads 2 \
    --use_wandb False --user_name "zsheikhb" --wandb_name "zsheikhb" --share_policy False --use_centralized_V False --use_attention True \
    --skill_dim 64  --num_training_skill_dynamics 1 --entropy_coef 0.006 --attention_module "RIM" --rim_num_units 6 --rim_topk 4 --hidden_size 96 \
    --num_env_steps 4000000 --log_interval 1 --episode_length 1000 --skill_discriminator_lr 0.00001 --dynamics_lr 0.00001 --coefficient_skill_return 0.005
