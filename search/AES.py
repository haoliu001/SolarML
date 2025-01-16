
import copy
import logging
import os
from pathlib import Path
import pickle
import subprocess
from typing import List
from config.cfg import AgingEvoConfig, BoundConfig, TrainingConfig # type: ignore
from searchspace.cnnarchitecture import CnnArchitecture # type: ignore
from searchspace.schema import get_schema # type: ignore
from trainer.dClass import ArchitecturePoint, EvaluatedPoint # type: ignore
from trainer.gpu_trainer import GPUTrainer # type: ignore
from trainer.model_trainer import ModelTrainer # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from constants.const import globalVar


class AgingEvoSearch:
    def __init__(self,
                 experiment_name: str,
                 search_config: AgingEvoConfig,
                 training_config: TrainingConfig,
                 bound_config: BoundConfig):
        self.log = logging.getLogger(name=f"AgingEvoSearch [{experiment_name}]")
        self.config = search_config
        self.trainer = ModelTrainer(training_config)

        self.root_dir = Path(search_config.checkpoint_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.pruning = training_config.pruning

        self.constraint_bounds = [bound_config.error_bound,
                                  bound_config.peak_mem_bound,
                                  bound_config.model_size_bound,
                                  bound_config.mac_bound,
                                  bound_config.e_bound] # add e_bound

        self.history: List[EvaluatedPoint] = []
        self.population: List[EvaluatedPoint] = []

        self.population_size = search_config.population_size
        self.initial_population_size = search_config.initial_population_size or self.population_size
        self.rounds = search_config.rounds
        self.sample_size = search_config.sample_size
        num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
        self.max_parallel_evaluations = search_config.max_parallel_evaluations or num_gpus

    def save_state(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.history, f)
        self.log.info(f"Saved {len(self.history)} architectures to {file}.")

    def load_state(self, file):
        with open(file, "rb") as f:
            self.history = pickle.load(f)
            self.history = self.history[0:self.population_size]
        self.population = self.history[-self.population_size:]

    def maybe_save_state(self, save_every):
        if len(self.history) % save_every == 0:
            # orignal save into pickle
            file = self.root_dir / f"{self.experiment_name}_agingevosearch_state.pickle"
            self.save_state(file.as_posix())

    # updated loss function
    def acc_e_loss_fn(self):
      def loss_fn(i:EvaluatedPoint):
        acc = 1 - i.val_error
        e = i.resource_features[3]
        return acc - (globalVar.lambda_factor * (e-globalVar.e_min)/(globalVar.e_max-globalVar.e_min)) 
      return loss_fn

    # prev loss function in uNAS
    def get_mo_fitness_fn(self):
        lambdas = np.random.uniform(low=0.0, high=1.0, size=4)
        def normalise(x, l=0, u=1, cap=10.0):
            return min((x - l) / (u - l), cap)
        def fitness(i: EvaluatedPoint):
            features = [i.val_error] + i.resource_features
            normalised_features = [normalise(f, u=c) / l
                                   for f, c, l in zip(features, self.constraint_bounds, lambdas)
                                   if c is not None]  # bound = None means ignored objective
            return -max(normalised_features) 
        return fitness

    def bounds_log(self, history_size=25):
        def to_feature_vector(i):
            return [i.val_error] + i.resource_features
        within_bounds = \
            [all(o <= b
                 for o, b in zip(to_feature_vector(i), self.constraint_bounds)
                 if b is not None)
             for i in self.history[-history_size:]]
        self.log.info(f"In bounds: {sum(within_bounds)} within "
                      f"the last {len(within_bounds)} architectures.")

    def evolve(self, point: ArchitecturePoint):
        arch = np.random.choice(self.config.search_space.produce_morphs(point.arch))
        sparsity = None
        if self.pruning:
            incr = np.random.normal(loc=0.0, scale=0.05)
            sparsity = np.clip(point.sparsity + incr,
                               self.pruning.min_sparsity, self.pruning.max_sparsity)
        return ArchitecturePoint(arch=arch, sparsity=sparsity)

    def evolve_sense(self, point:ArchitecturePoint, tr):
        
        sense1 = point.arch.architecture['sample'][0] # type: ignore
        sense2 = point.arch.architecture['sample'][1] # type: ignore
        sense3 = point.arch.architecture['sample'][2] # type: ignore

        sense1_list_aux = [sense1-2, sense1, sense1+2]
        sense2_list_aux = [sense2-2, sense2, sense2+2]
        
        sense3_drop = copy.deepcopy(sense3)
        sense3_insert = copy.deepcopy(sense3)
        rand_index = np.random.randint(0, 9)
        while rand_index not in sense3:
            rand_index = np.random.randint(0, 9)
        sense3_drop.remove(rand_index)

        if len(sense3) != 9:
            rand_index = np.random.randint(0, 9)
            while rand_index in sense3:
                rand_index = np.random.randint(0, 9)
            sense3_insert.append(rand_index)
        sense3_list_aux = [sense3_drop, sense3, sense3_insert]

        schema = get_schema()
        min_sense1, max_sense1 = schema[f"sense1"].bounds # type: ignore
        min_sense2, max_sense2 = schema[f"sense2"].bounds # type: ignore
        min_sense3, max_sense3 = schema[f"sense3"].bounds # type: ignore


        sense1_list = [i for i in sense1_list_aux if i>= min_sense1 and i<= max_sense1]
        sense2_list = [i for i in sense2_list_aux if i>= min_sense2 and i<= max_sense2]
        sense3_list = [i for i in sense3_list_aux if len(i)>= min_sense3 and len(i)<= max_sense3]

        param_grid = {'s1': sense1_list, 's2': sense2_list,'s3': sense3_list}
        grid_score_list = []
        grid_candidate_list = []
        for s1 in param_grid["s1"]:
          for s2 in param_grid["s2"]:
              for s3 in param_grid['s3']:
                point.arch.architecture['sample'][0]= s1 # type: ignore
                point.arch.architecture['sample'][1]= s2 # type: ignore
                point.arch.architecture['sample'][2]= s3 # type: ignore
                info = tr.evaluate(point)
                print(" sample_candidate is : {0}, info is {1}".format( point.arch.architecture, info),file=open(globalVar.evolve_name, 'a')) # type: ignore
                acc = 1 - info.test_error
                e = info.resource_features[3]
                metric = acc - globalVar.lambda_factor * e 
                print("acc, e, metric in  mutate_sample is: ",acc,e,metric)
                grid_score_list.append(metric)
                grid_candidate_list.append(copy.deepcopy(point.arch.architecture)) # type: ignore ## see if it really works
                print('rate {0}, am_reso {1}, sp_reso {2}, acc {3}, e {4}, metric {5}'.format(s1, s2, s3, acc, e, metric))
        best_index = np.argmax(np.array(grid_score_list))
        print('best index {0}'.format(best_index))
        point.arch.architecture = grid_candidate_list[best_index] # type: ignore # update architecture, need to check
        info = tr.evaluate(point)
        print(" best sample_candidate is : {0}, info is {1}".format( point.arch.architecture, info),file=open(globalVar.evolve_name, 'a'))  # type: ignore
        return info


    def random_sample(self):
        arch = self.config.search_space.random_architecture()
        print("arch in random_sample() is {0}".format(arch))
        sparsity = None
        if self.pruning:
            sparsity = np.random.uniform(self.pruning.min_sparsity, self.pruning.max_sparsity)
        return ArchitecturePoint(arch=arch, sparsity=sparsity)

    def print_log(self):
        print("history:{0}".format(self.history))


    def show_best_candidate(self, bestArch, Sparsity):
        trainer = self.trainer
        ss = self.config.search_space
        tr = GPUTrainer(ss, trainer)
        bestCan = ArchitecturePoint(arch=CnnArchitecture(bestArch), sparsity=Sparsity)
        info = tr.evaluate(bestCan)
        filename = 'eNAS_lambda' + str(globalVar.lambda_factor) + '_' + globalVar.appName + '_' + globalVar.res_version + '_final_result.txt' 
        print("best candidate is : {0}, info is {1}".format(bestCan.arch.architecture, info), file=open(filename, 'a')) # type: ignore

    def check_load_from(self, load_from: str = None, save_every: int = None): # type: ignore
        if load_from:
            self.load_state(load_from)
            filename = load_from+'history.txt'
            print('len loaded history is {0}; len population is {1}'.format(len(self.history), len(self.population)), file=open(filename, 'a'))
            print('loadded history is {0}; population is {1}'.format(self.history, self.population), file=open(filename, 'a'))
            for candidate in self.population:
                energy_tot = candidate.resource_features[3]
                # global e_min, e_max
                if energy_tot < globalVar.e_min:
                    globalVar.e_min = energy_tot
                if energy_tot > globalVar.e_max:
                    globalVar.e_max =  energy_tot
            np.random.seed(None)
            sample = np.random.choice(self.population, size=self.sample_size)
            for index, element in enumerate(sample):
                value = self.acc_e_loss_fn()(element)
                print('index is {0}, val is {1}, ele is {2}'.format(index, value, element), file=open(filename, 'a'))
            parent = max(sample, key=self.acc_e_loss_fn())
            # index_parent = np.argmax(sample)
            print(' parent architecture is : {0}'.format( parent),file=open(filename, 'a'))



    def search(self, load_from: str = None, save_every: int = None): # type: ignore
        if load_from:
            self.load_state(load_from)
            filename = load_from+'history' + globalVar.appName + '_' + globalVar.res_version + '_'+ str(globalVar.lambda_factor) +'.txt' 
            print('len loaded history is {0}; len population is {1}'.format(len(self.history), len(self.population)), file=open(filename, 'a'))
            print('loadded history is {0}; population is {1}'.format(self.history, self.population), file=open(filename, 'a'))

        trainer = self.trainer
        ss = self.config.search_space
        tr = GPUTrainer(ss, trainer)

        def should_submit_more(cap):
            return (len(self.history) < cap)

        def point_number():
            return len(self.history) + 1

        def s_evolve():
            round = 0
            while len(self.history) < self.rounds:
                    np.random.seed(None)
                    sample = np.random.choice(self.population, size=self.sample_size)
                    parent = max(sample, key=self.acc_e_loss_fn()) 
                    
                    print(("parent architecture is : {0}, info is {1}".format(parent.point.arch.architecture, tr.evaluate(parent.point))),file=open(globalVar.evolve_name,'a'))
                
                    info, qt_size  = tr.evaluate(self.evolve(parent.point))
                    if len(self.history) %  globalVar.turns == 0: # type: ignore
                        info = self.evolve_sense(info.point, tr) # type: ignore
                    
                    energy_tot = info.resource_features[3] # type: ignore
                    features = [info.val_error] + info.resource_features[0:4] # type: ignore
                    if all(o <= b for o, b in zip(features, self.constraint_bounds) if b is not None):
                        round += 1
                        print("round {0} new can is : {1}, info is {2}".format(round, info.point.arch.architecture, info),file=open(globalVar.evolve_name, 'a')) # type: ignore
                        self.population.append(info) # type: ignore

                        while len(self.population) > self.population_size:
                            self.population.pop(0)
                        self.history.append(info) # type: ignore
                        self.maybe_save_state(save_every)
                        self.bounds_log()


        def tf2cc(index):
            source_file = 'model_aux.tflite'
            mid_file = 'mcuMeter_res1/tflite_v2/kws_m4_'+str(index)+'.tflite'
            end_file = 'mcuMeter_res1/ccFiles_v2/kws_m4_'+str(index)+'.cc'
            command = "xxd -i "+source_file + " > "+end_file
            subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            os.rename(source_file, mid_file)

        turn = 0
        while len(self.history) < self.initial_population_size:
                rand_sample = self.random_sample()   
                info, qt_size = tr.evaluate(rand_sample)

                energy_tot = info.resource_features[3]
                features = [info.val_error] + info.resource_features[0:4]
                print('features: ', features)
                print('constraints: ', self.constraint_bounds)
                if all(o <= b for o, b in zip(features, self.constraint_bounds) if b is not None):
                    self.population.append(info)
                    self.history.append(info)
                    self.maybe_save_state(save_every)
                    # resource_features = [peak_memory_usage(rg), model_size(rg, sparse=unstructured_sparsity),
                        #  inference_latency(rg, compute_weight=1, mem_access_weight=0),energy_tot]
                    print("turn {0} ok in constraints is : {1}, energy is {2}".format(turn, features, energy_tot),file=open(globalVar.search_name, 'a'))
                    turn += 1
                    if energy_tot < globalVar.e_min:
                        globalVar.e_min = energy_tot
                    if energy_tot > globalVar.e_max:
                        globalVar.e_max =  energy_tot

        print("e_max = {0}".format(globalVar.e_max),file=open(globalVar.search_name,'a'))
        print("e_min = {0}".format(globalVar.e_min),file=open(globalVar.search_name,'a'))
        print("history = {0}".format(self.history),file=open(globalVar.search_name,'a'))
        print("population = {0}".format(self.population),file=open(globalVar.search_name,'a'))
        
        s_evolve()