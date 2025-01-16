#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --output=slurm_output_L-MACs_e10_50-20-250_1-16_%A.out




module load 2022
module load 2023
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0


source /gpfs/work3/0/prjs1059/tf13/bin/activate



XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME python /gpfs/work3/0/prjs1059/SolarML/main.py \
--appName 'speech' \
--lambda_factor 0.0 \
--population_size 20\ 
--sample_size 10\  
--epoch_param 20\
--rounds 50  \
--profile_methods 'L-MACs'\
--peak_mem_bound 1000000\   
--model_size_bound 1000000\
--error_bound 0.5\
--e_bound 100000000000 --res_version '_50-20-250_L-MACs_e_10000000_1-16_' \
# --parser_load '/home/hliu1/prjs1059/SolarML/artifacts/kws/L-MACs_search_speech0.0__50-20-250_L-MACs_e_10_1-13__agingevosearch_state.pickle'

# --parser_load '/home/hliu1/prjs1059/SolarML/artifacts/kws/MACs_search_speech0.0__50-20-250_MACs_e_10_11-22__agingevosearch_state.pickle' 
# --parser_load '/gpfs/work3/0/prjs1059/SolarML/artifacts/kws/nnMeter_search_speech0.0__100-20-300_nnMeter_20_11-14__agingevosearch_state.pickle' 
# '/gpfs/work3/0/prjs1059/SolarML/artifacts/kws/search_speech0.0_a_agingevosearch_state.pickle' 'MACs'
#--population_size 50 --sample_size 20  --rounds 250  \


# XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME python /gpfs/work3/0/prjs1059/SolarML/main.py --appName 'speech' --lambda_factor 0.0 --population_size 10 --sample_size 5 --epoch_param 1 --rounds 20  --profile_methods 'L-MACs' --peak_mem_bound 10000000000  --model_size_bound 10000000000 --error_bound 0.8 --e_bound 100000000000 --res_version '_50-20-250_L-MACs_e_10_1-2_' --parser_load '/home/hliu1/prjs1059/SolarML/artifacts/kws/L-MACs_search_speech0.0__50-20-250_L-MACs_e_10_1-13__agingevosearch_state.pickle'