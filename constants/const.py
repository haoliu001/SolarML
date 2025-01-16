from typing import Any
import tensorflow as tf # type: ignore
import tensorflow_addons as tfa # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler # type: ignore

# from config.cfg import TrainingConfig, BoundConfig, AgingEvoConfig, PruningConfig # type: ignore


class globalVar:
    num_class_digits = 11
    num_class_letters = 27
    num_class_unlocking = 31
    x_train=None
    y1_train=None
    y2_train=None
    y3_train= x_val= y1_val=y2_val=y3_val=x_test= y1_test= y2_test= y3_test=None
    e_max = 0.0
    e_min = 1000000.0
    
    appName = ''
    lambda_factor = 0.0
    epoch_param = 0
    error_bound = 0.0
    peak_mem_bound = 0
    model_size_bound = 0
    parser_load = ''
    res_version = ''
    MAX_CONV_BLOCKS = 0
    MAX_LAYERS_PER_CONV_BLOCK = 0
    MAX_DENSE_BLOCKS = 0
    turns = 300
    start_pruning_param = 0
    finish_pruning_param = 0
    task_type = ''
    search_algorithm = None
    _SCHEMA = None
    MAX_SENSE1 = MAX_SENSE2 = MAX_SENSE3 = 0
    search_name = evolve_name = ''
    population_size = 0
    sample_size = 0
    rounds = 150
    global_cato = 0
    exp_name = ''
    profile_methods = ''
    profile_sample = False
    e_bound = 20


    training_config = None
    search_config = None
    bound_config = None

    
    @classmethod
    def loadArgus(cls):
        from config.cfg import TrainingConfig
        def lr_schedule(epoch):
            if 0 <= epoch < 90:
                return 0.01
            if 90 <= epoch < 105:
                return 0.005
            return 0.001
        if cls.appName == 'solar':
            if cls.MAX_SENSE1 == 0 | cls.MAX_SENSE2 == 0 | cls.MAX_SENSE3 == 0:
                cls.MAX_SENSE1 = 100
                cls.MAX_SENSE2 = 12
                cls.MAX_SENSE3 = 9
            cls.global_cato = 0
            from data.solarData import SolarData
            cls.training_config = TrainingConfig(
                dataset=SolarData(classCato=cls.global_cato),
                # dataset=SpeechCommands("/datasets/speech_commands_v0.02"),
                epochs=cls.epoch_param,
                batch_size=32, #512 for speech command
                # optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.005, momentum=0.9, weight_decay=4e-5),
                optimizer=lambda: tf.keras.optimizers.Adam(learning_rate=0.00001),
                callbacks=lambda: [tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=3)],
            )
        elif cls.appName == 'speech':
            from data.speechcommands import SpeechCommands # type: ignore
            if cls.MAX_SENSE1 == 0 | cls.MAX_SENSE2 == 0 | cls.MAX_SENSE3 == 0:
                cls.MAX_SENSE1 = 30  # winS
                cls.MAX_SENSE2 = 30  # winD
                cls.MAX_SENSE3 = 40  # features
            cls.training_config = TrainingConfig(
                dataset=SpeechCommands("/gpfs/work3/0/prjs1059/BF_solarcell_EAI/2nd_year/inNAS-layerData-M4/datasets/speech_commands_v0.02"),
                epochs=cls.epoch_param,
                batch_size=512,
                optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.005, momentum=0.9, weight_decay=4e-5),
                callbacks=lambda: []
            )
        elif cls.appName == 'cifar10':
            from data.cifar10 import CIFAR10
            cls.MAX_SENSE1 = 100
            cls.MAX_SENSE2 = 12
            cls.MAX_SENSE3 = 9
            cls.training_config = TrainingConfig(dataset=CIFAR10(),optimizer=lambda: tfa.optimizers.SGDW(learning_rate=0.01, momentum=0.9, weight_decay=1e-5),batch_size=128,epochs=130, callbacks=lambda: [LearningRateScheduler(lr_schedule)])
        else:
            cls.training_config = None
            cls.MAX_SENSE1 = 0
            cls.MAX_SENSE2 = 0
            cls.MAX_SENSE3 = 0
            assert('can not find corresponding app')


        cls.search_name = cls.profile_methods+ '_'+ cls.appName + cls.task_type+ str(cls.lambda_factor) + "_search_" + cls.res_version+'.txt'
        cls.evolve_name = cls.profile_methods+ '_'+cls.appName + cls.task_type+ str(cls.lambda_factor) + "_evolve_" + cls.res_version+'.txt'
        cls.exp_name = cls.profile_methods+ '_'+"search_" +cls.appName + cls.task_type+ str(cls.lambda_factor) + "_" + cls.res_version
        from searchspace.cnnSS import CnnSearchSpace # type: ignore
        from config.cfg import BoundConfig, AgingEvoConfig, PruningConfig # type: ignore

        cls.search_config = AgingEvoConfig(
            search_space=CnnSearchSpace(), # type: ignore
            checkpoint_dir="artifacts/kws",
            population_size= cls.population_size, 
            sample_size= cls.sample_size,
            initial_population_size = cls.population_size, 
            rounds=cls.rounds
        )

        cls.bound_config = BoundConfig(
            error_bound=cls.error_bound,
            peak_mem_bound=cls.peak_mem_bound,
            model_size_bound=cls.model_size_bound,
            mac_bound=30000000,
            e_bound = cls.e_bound
        )

        cls.training_config.pruning = PruningConfig( # type: ignore
            structured=False,
            start_pruning_at_epoch=cls.start_pruning_param,
            finish_pruning_by_epoch=cls.finish_pruning_param, #18
            min_sparsity=0.05, # 0.05
            max_sparsity=0.8 # 0.8
        )