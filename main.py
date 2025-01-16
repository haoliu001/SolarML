
import argparse
import logging
from constants.const import globalVar
from search.AES import AgingEvoSearch
import tensorflow as tf # type: ignore


def main():

    parser_save_every =1
    parser_seed = 0

    parser = argparse.ArgumentParser(description="params")
    parser.add_argument('--appName', type=str, required=True, help="targeted application (solar or speech)")
    parser.add_argument('--lambda_factor', type = float,default=0.0, help="if tadeoff in loss fn acc-lambda*(e-e_min)/(e_max-e_min)")
    parser.add_argument('--epoch_param', type = int, default=20, help="how many epoches")
    parser.add_argument('--error_bound',type = float, default=0.25, help="it is 1-accuracy")
    parser.add_argument('--peak_mem_bound', type = int,default=100000, help="")
    parser.add_argument('--model_size_bound', type = int,default=100000, help="it represents the model size (Bytes)")
    parser.add_argument('--parser_load', default=None, help='we can load previous running recordings')
    parser.add_argument('--res_version', default='a', help='version')
    parser.add_argument('--MAX_CONV_BLOCKS', type = int,default=10, help='the max number of conv blocks in model search space')
    parser.add_argument('--MAX_LAYERS_PER_CONV_BLOCK', default=4, help='the max number of layers in model blocks')
    parser.add_argument('--MAX_DENSE_BLOCKS', type = int,default=3, help='the max number of dense blocks in model search space')
    parser.add_argument('--turns', type = int,default=20, help='the frequency of executing evolution once')
    parser.add_argument('--start_pruning_param',type = int, default=3, help='parameter of pruning')
    parser.add_argument('--finish_pruning_param', type = int,default=18, help='parameter of pruning')
    parser.add_argument('--search_algorithm', default=AgingEvoSearch, help='supported searching algorithm (only support AgingEvoSearch )')
    parser.add_argument('--population_size', type= int, default=100, help='')
    parser.add_argument('--sample_size', type= int,default=50, help='')
    parser.add_argument('--rounds',type= int, default=200, help='')
    parser.add_argument('--profile_methods',type= str, default='L-MACs', help='prediction methods for energy')
    parser.add_argument('--profile_sample',type= bool, default=False, help='if has records')
    parser.add_argument('--e_bound',type= int, default=1000000, help='energy bounds units mW*s')
    parser.add_argument('--min_sense1',type= int, default=0, help='min of sensing param 1')
    parser.add_argument('--max_sense1',type= int, default=0, help='max of sensing param 1')
    parser.add_argument('--min_sense2',type= int, default=0, help='min of sensing param 2')
    parser.add_argument('--max_sense2',type= int, default=0, help='max of sensing param 2')
    parser.add_argument('--min_sense3',type= int, default=0, help='min of sensing param 3')
    parser.add_argument('--max_sense3',type= int, default=0, help='max of sensing param 3')
    
    
    args = parser.parse_args()
    
    globalVar.appName = args.appName
    globalVar.lambda_factor = args.lambda_factor
    globalVar.epoch_param = args.epoch_param
    globalVar.error_bound = args.error_bound
    globalVar.peak_mem_bound = args.peak_mem_bound
    globalVar.model_size_bound = args.model_size_bound
    globalVar.parser_load = args.parser_load
    globalVar.res_version = args.res_version
    globalVar.MAX_CONV_BLOCKS = args.MAX_CONV_BLOCKS
    globalVar.MAX_LAYERS_PER_CONV_BLOCK = args.MAX_LAYERS_PER_CONV_BLOCK
    globalVar.MAX_DENSE_BLOCKS = args.MAX_DENSE_BLOCKS
    globalVar.turns = args.turns
    globalVar.start_pruning_param = args.start_pruning_param
    globalVar.finish_pruning_param =args.finish_pruning_param
    globalVar.search_algorithm = args.search_algorithm
    globalVar.population_size = args.population_size
    globalVar.sample_size = args.sample_size
    globalVar.rounds = args.rounds
    globalVar.profile_methods = args.profile_methods
    globalVar.profile_sample = args.profile_sample
    globalVar.e_bound = args.e_bound
    


    globalVar.loadArgus()

    

    algo = globalVar.search_algorithm
    search_space = globalVar.search_config.search_space
    dataset = globalVar.training_config.dataset # type: ignore
    search_space.input_shape = dataset.input_shape
    search_space.num_classes = dataset.num_classes

    
    search = algo(experiment_name=globalVar.exp_name,
                  search_config=globalVar.search_config,
                  training_config=globalVar.training_config,
                  bound_config=globalVar.bound_config)
    search.search(load_from=globalVar.parser_load, save_every=parser_save_every)
    search.print_log()


if __name__ == "__main__":
    main()