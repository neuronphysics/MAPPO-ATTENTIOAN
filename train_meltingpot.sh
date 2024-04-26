#!/bin/bash
conda init bash

conda shell.bash activate marl
wandb login a2a1bab96ebbc3869c65e3632485e02fcae9cc42
conda activate marl


general_dir="/home/zsheikhb/MARL"

environment=$1
seed=$2
module=$3
hidden=$4
units=$5
topk=$6
skill=$7
optimizer=$8
run=$9

# Default values in case the environment does not match
agents=0
substrate=""
episode_length=0
bottom=0
sup=0

# Set agents and substrate name based on the environment
# Set agents and substrate name based on the environment
case "$environment" in
    "HARVEST")
        echo "RUNNING HARVEST"
        agents=16
        substrate="allelopathic_harvest__open"
        episode_length=2000
        bottom=32
        sup=24
        ;;
    "TERRITORY_O" | "TERRITORY_R" | "TERRITORY_I")
        case "$environment" in
            "TERRITORY_O")
                echo "RUNNING territory__open"
                substrate="territory__open"
                agents=9
                bottom=18  # Adjust these values as necessary for the environment
                sup=15
                ;;
            "TERRITORY_R")
                echo "RUNNING territory__rooms"
                substrate="territory__rooms"
                agents=9
                bottom=18  # Adjust these values as necessary for the environment
                sup=15
                ;;
            "TERRITORY_I")
                echo "RUNNING territory__inside_out"
                substrate="territory__inside_out"
                agents=5
                bottom=10  # Adjust these values as necessary for the environment
                sup=8
                ;;
        esac
        episode_length=1000
        ;;
    "PREDATOR_RF" | "PREDATOR_O" | "PREDATOR_AH" | "PREDATOR_OR")
        case "$environment" in
            "PREDATOR_RF")
                echo "RUNNING predator_prey__random_forest"
                substrate="predator_prey__random_forest"
                agents=12
                bottom=24  # Adjust these values as necessary for the environment
                sup=18
                ;;
            "PREDATOR_O")
                echo "RUNNING predator_prey__open"
                substrate="predator_prey__open"
                agents=12
                bottom=24  # Adjust these values as necessary for the environment
                sup=18
                ;;
            "PREDATOR_AH")
                echo "RUNNING predator_prey__alley_hunt"
                substrate="predator_prey__alley_hunt"
                agents=10
                bottom=20  # Adjust these values as necessary for the environment
                sup=15
                ;;
            "PREDATOR_OR")
                echo "RUNNING predator_prey__orchard"
                substrate="predator_prey__orchard"
                agents=12
                bottom=24  # Adjust these values as necessary for the environment
                sup=18
                ;;
        esac
        episode_length=1000
        ;;
    "CHICKEN_A" | "CHICKEN_R")
        case "$environment" in
            "CHICKEN_A")
                echo "RUNNING chicken_in_the_matrix__arena"
                substrate="chicken_in_the_matrix__arena"
                agents=8
                bottom=16  # Adjust these values as necessary for the environment
                sup=12
                ;;
            "CHICKEN_R")
                echo "RUNNING chicken_in_the_matrix__repeated"
                substrate="chicken_in_the_matrix__repeated"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
        esac
        episode_length=1000
        ;;
    "COOKING_ASY" | "COOKING_CRAMPED" | "COOKING_RING" | "COOKING_CIRCUIT" | "COOKING_FORCED" | "COOKING_EIGHT" | "COOKING_CROWDED")
        case "$environment" in
            "COOKING_ASY")
                echo "RUNNING collaborative_cooking__asymmetric"
                substrate="collaborative_cooking__asymmetric"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_CRAMPED")
                echo "RUNNING collaborative_cooking__cramped"
                substrate="collaborative_cooking__cramped"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_RING")
                echo "RUNNING collaborative_cooking__ring"
                substrate="collaborative_cooking__ring"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_CIRCUIT")
                echo "RUNNING collaborative_cooking__circuit"
                substrate="collaborative_cooking__circuit"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_FORCED")
                echo "RUNNING collaborative_cooking__forced"
                substrate="collaborative_cooking__forced"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_EIGHT")
                echo "RUNNING collaborative_cooking__figure_eight"
                substrate="collaborative_cooking__figure_eight"
                agents=6
                bottom=12  # Adjust these values as necessary for the environment
                sup=9
                ;;
            "COOKING_CROWDED")
                echo "RUNNING collaborative_cooking__crowded"
                substrate="collaborative_cooking__crowded"
                agents=9
                bottom=18  # Adjust these values as necessary for the environment
                sup=14
                ;;
        esac
        episode_length=1000
        ;;
    "SCISSORS_A" | "SCISSORS_R" | "SCISSORS_O")
        case "$environment" in
            "SCISSORS_A")
                echo "RUNNING running_with_scissors_in_the_matrix__arena"
                substrate="running_with_scissors_in_the_matrix__arena"
                agents=8
                bottom=16  # Adjust these values as necessary for the environment
                sup=12
                ;;
            "SCISSORS_R")
                echo "RUNNING running_with_scissors_in_the_matrix__repeated"
                substrate="running_with_scissors_in_the_matrix__repeated"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "SCISSORS_O")
                echo "RUNNING running_with_scissors_in_the_matrix__one_shot"
                substrate="running_with_scissors_in_the_matrix__one_shot"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
        esac
        episode_length=1000
        ;;
    "STAG_HUNT")
        echo "RUNNING stag_hunt_in_the_matrix__arena"
        substrate="stag_hunt_in_the_matrix__arena"
        agents=8
        episode_length=1000
        bottom=16  # Adjust these values as necessary for the environment
        sup=12
        ;;
    "CLEAN")
        echo "RUNNING clean_up"
        agents=7
        substrate="clean_up"
        episode_length=1000
        bottom=14
        sup=11
        ;;
    "CHEMISTRY")
        echo "RUNNING chemistry"
        agents=8
        substrate="chemistry__three_metabolic_cycles_with_plentiful_distractors"
        episode_length=1000
        bottom=16
        sup=12
        ;;
    "STRAVINSKY")
        echo "RUNNING bach or stravinsky"
        agents=8
        substrate="bach_or_stravinsky_in_the_matrix__arena"
        episode_length=1000
        bottom=16
        sup=12
        ;;
    "COOKING")
        echo "RUNNING coolaborative cooking"
        agents=2
        substrate="collaborative_cooking__cramped"
        episode_length=1000
        bottom=4
        sup=3
        ;;
    "PRISONERS")
        echo "RUNNING prisoners_dilemma_in_the_matrix__arena"
        agents=8
        substrate="prisoners_dilemma_in_the_matrix__arena"
        episode_length=1000
        bottom=16
        sup=12
        ;;
    *)
        echo "Unknown environment: $environment"
        exit 1
        ;;
esac

echo "Agents: $agents"
echo "Substrate: $substrate"
echo "Environment: $environment"
echo "Seed: $seed"
echo "Module: $module"
echo "Hidden: $hidden"
echo "Units: $units"
echo "using: $skills"
echo "Optimizer: $optimizer"
echo "Run number: $run"



if [ "$skill" = "SKILLS" ]; then
    export PYTHONPATH="$PYTHONPATH:/home/zsheikhb/MARL/master_skills"
    echo "Using skill"
    # Execute the program based on the module
    if [ "$module" = "RIM" ]; then
        echo "Executing program for RIM"
        CUDA_VISIBLE_DEVICES=0,1 python /home/zsheikhb/MARL/master_skills/onpolicy/scripts/train/train_meltingpot.py --bottom_up_form_num_of_objects ${bottom} --sup_attention_num_keypoints ${sup} --rim_num_units ${units} --rim_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot
        " --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --num_bands_positional_encoding 32 --skill_dim 128 --num_training_skill_dynamics 1 --entropy_coef 0.004 --skill_discriminator_lr 0.00001 --coefficient_skill_return 0.005 
    elif [ "$module" = "SCOFF" ]; then
        echo "Executing program for SCOFF"
        CUDA_VISIBLE_DEVICES=0,1 python /home/zsheikhb/MARL/master_skills/onpolicy/scripts/train/train_meltingpot.py --bottom_up_form_num_of_objects ${bottom} --sup_attention_num_keypoints ${sup} --scoff_num_units ${units} --scoff_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --num_bands_positional_encoding 32 --skill_dim 128 --num_training_skill_dynamics 1 --entropy_coef 0.004 --skill_discriminator_lr 0.00001 --coefficient_skill_return 0.005 
    elif [ "$module" = "LSTM" ]; then
        echo "Executing program for LSTM"
        CUDA_VISIBLE_DEVICES=0,1 python /home/zsheikhb/MARL/master_skills/onpolicy/scripts/train/train_meltingpot.py --bottom_up_form_num_of_objects ${bottom} --sup_attention_num_keypoints ${sup} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention False --use_naive_recurrent_policy True --use_recurrent_policy True --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --num_bands_positional_encoding 32 --skill_dim 128 --num_training_skill_dynamics 1 --entropy_coef 0.004 --skill_discriminator_lr 0.00001 --coefficient_skill_return 0.005 
    else
        echo "Module is neither RIM nor SCOFF, nor LSTM"
    fi

else
    export PYTHONPATH="$PYTHONPATH:/home/zsheikhb/MARL/master"

    echo "Not using skill"

    if ls $general_dir/master/onpolicy/scripts/results/Meltingpot/collaborative_cooking__circuit_0/$substrate/mappo/check/run$run/models/*.pt 1> /dev/null 2>&1; then
        echo "Pre-trained Model exists"
        load_model=True
        path_dir=$general_dir/master/onpolicy/scripts/results/Meltingpot/collaborative_cooking__circuit_0/$substrate/mappo/check/run$run/models
    else
        echo "Pre-trained Model does not exist"
        load_model="False"
        path_dir=None
    fi


    # Execute the program based on the module
    if [ "$module" = "RIM" ]; then
        echo "Executing program for RIM"
        CUDA_VISIBLE_DEVICES=0,1 python $general_dir/master/onpolicy/scripts/train/train_meltingpot.py --load_model ${load_model} --model_dir ${path_dir}  --run_num ${run} --optimizer ${optimizer} --rim_num_units ${units} --rim_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --entropy_coef 0.004 
    elif [ "$module" = "SCOFF" ]; then
        echo "Executing program for SCOFF"
        CUDA_VISIBLE_DEVICES=0,1 python $general_dir/master/onpolicy/scripts/train/train_meltingpot.py --load_model ${load_model} --model_dir ${path_dir} --run_num ${run} --optimizer ${optimizer} --scoff_num_units ${units} --scoff_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --entropy_coef 0.004 
    elif [ "$module" = "LSTM" ]; then
        echo "Executing program for LSTM"
        CUDA_VISIBLE_DEVICES=0,1 python $general_dir/master/onpolicy/scripts/train/train_meltingpot.py --load_model ${load_model} --model_dir ${path_dir} --run_num ${run} --optimizer ${optimizer} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads 1 --use_wandb True --share_policy False --use_centralized_V False --use_attention False --use_naive_recurrent_policy True --use_recurrent_policy True --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo --lr 0.00002 --max_grad_norm 0.2 --entropy_coef 0.004 
    else
        echo "Module is neither RIM nor SCOFF, nor LSTM"
    fi
fi