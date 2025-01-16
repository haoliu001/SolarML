import logging
import math
from constants.const import globalVar
import numpy as np # type: ignore
from utils.process import try_count_flops, quantised_accuracy, peak_memory_usage, model_size, inference_latency
from .dClass import EvaluatedPoint # type: ignore
from constants.const import globalVar
# from mcuMeter_pred.mcuMeter_pred import ModelHandler
# from nnMeter_pred.nnMeter_pred import generate_latency_energy_for_tflite_nnMeter # type: ignore


class GPUTrainer:
    def __init__(self, search_space, trainer):
        self.trainer = trainer
        self.ss = search_space
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    def evaluate(self, point, saveH5MACs=False, index=''):
        log = logging.getLogger("Worker")
        data = self.trainer.dataset
        arch = point.arch
        data.update_dataset(arch.architecture)
        model = self.ss.to_keras_model(arch, data.input_shape, data.num_classes)
        if saveH5MACs == True:
            flops,dense_flops,depth_flops,maxpool_flops,BS_flops,conv_flops = try_count_flops(model)
            model.save('/home/hliu1/prjs1059/MCU-meter-NAS/dataset/L-Data-1000/h5/'+index)
            print('model index: ', index)
            return flops
        results = self.trainer.train_and_eval(model, sparsity=point.sparsity)
        val_error, test_error= results["val_error"], results["test_error"]
        rg = self.ss.to_resource_graph(arch, data.input_shape, data.num_classes,
                                       pruned_weights=results["pruned_weights"])
        unstructured_sparsity = self.trainer.config.pruning and \
                                not self.trainer.config.pruning.structured
        
        

        model_quantized = self.ss.to_keras_model(arch, data.input_shape, data.num_classes)
        qt_size = quantised_accuracy(model_quantized, data,1,5,1)

        if globalVar.profile_methods == 'L-MACs':
          flops,dense_flops,depth_flops,maxpool_flops,BS_flops,conv_flops = try_count_flops(model) # type: ignore
          energy_MACs = 0.00178851 * dense_flops -0.00095607 * depth_flops + 0.0094064 * maxpool_flops + 0.02781019 *BS_flops + 0.00085691 * conv_flops + 83.47847021437997
        elif globalVar.profile_methods == 'MACs':
          flops,dense_flops,depth_flops,maxpool_flops,BS_flops,conv_flops = try_count_flops(model) # type: ignore
          energy_MACs = (0.0354989 * flops + 33754.26025285)/10000
        else:
          energy_MACs = 0
        
        if globalVar.appName == 'solar':
          train_aux = data.train_dataset()[0]
          shape = np.array(train_aux).shape
          print("rate*delay is {0}, channels is {1}".format(shape[1],shape[2]))
          if np.max(np.array(train_aux)) <= 256:
            reso_aux = 8
          else :
            reso_aux = 10
          rate_aux = float(shape[1])/6
          energy_sample = rate_aux * 17.14187 + float(shape[2]) * 97.1332 - reso_aux * 20.3761 + 9481.0205
          # print("Rate: %d, energy: %f" % (rate_aux, energy_sample))
        elif globalVar.appName == 'speech':
          rate = arch.architecture['sample'][0]
          reso = arch.architecture['sample'][1]
          chan = arch.architecture['sample'][2]

          feature_size = math.floor((1000-reso)/(rate))+1

          if reso > 32:
            sampleSize = 1024
          elif reso > 16:
            sampleSize = 512
          elif reso > 8:
            sampleSize = 256
          elif reso > 4:
            sampleSize = 128
          elif reso > 2:
            sampleSize = 64
          else: 
            sampleSize = 0
          energy_sample = chan * 5.1246115 + reso * 20.29759982 - rate * 29.00385246 - feature_size * 0.14318719 + sampleSize*5.62146279 + 875.1937633603479
          print("Rate: %d, energy: %f" % (rate, energy_sample))
        elif globalVar.appName == 'cifar10':
          energy_sample = 100
        else:
          energy_sample = 0
          print('no appName match')

        if globalVar.profile_sample == False:
          energy_sample = 0  
        energy_tot = energy_MACs + energy_sample
        #Debug 1
        resource_features = [peak_memory_usage(rg), model_size(rg, sparse=unstructured_sparsity),
                             inference_latency(rg, compute_weight=1, mem_access_weight=0),energy_tot]
        resource_features[1] = qt_size # we change it into quantized size, which is more hardware-wise
        log.info(f"Training complete: val_error={val_error:.4f}, test_error={test_error:.4f}, "
                 f"resource_features={resource_features}.")
        return EvaluatedPoint(point=point,
                              val_error=val_error, test_error=test_error,
                              resource_features=resource_features), qt_size

