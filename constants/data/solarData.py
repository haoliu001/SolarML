from .dataset import Dataset # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
import pandas as pd
from tensorflow.keras.utils import to_categorical # type: ignore
from constants.const import globalVar
from sklearn.model_selection import train_test_split # type: ignore
from scipy import signal # type: ignore

class SolarData(Dataset):
    def __init__(self,classCato):
      self.classCato = classCato
      def create_frame(file_name):
        data_frame = []
        for i, file in enumerate(file_name):
          data = pd.read_csv(file,names=['sc1','sc2','sc3','sc4','sc5','sc6','sc7','sc8','sc9'])
          data_frame.append(data)
        data_frame = pd.concat(data_frame, join = 'outer')
        data_frame_reshape = np.reshape((data_frame).values, (-1,1200,9,1))
        return data_frame_reshape

      def create_label_frame(file_name):
        label_frame = pd.DataFrame(columns = ['label'])
        for i, file in enumerate(file_name):
          data = pd.read_csv(file, names = ['label'])
          label_frame = pd.concat([label_frame, data], axis = 0, ignore_index = True)
        label_frame_reshape = np.array(label_frame.values)
        return label_frame_reshape

      def get_data_files():
        user_names = [ 'user1','user2','user3','user4','user5','user6','user7','user8','user9','user10']
        pre_fileName = '/home/hliu1/prjs1059/SolarML/Data_new/data_'
        label1 = '_label1'
        types_list = ['_digits.csv']
        file_data = []
        file_label1 = []
        for types in types_list:
          for i in range(len(user_names)):
              filename = pre_fileName + user_names[i] + types
              labelname1 = pre_fileName + user_names[i]+label1 + types
              file_data.append(filename)
              file_label1.append(labelname1)
        return file_data, file_label1

      def create_frame_data_label(file_data, file_label1):
        data_combined = create_frame(file_data)
        data_3D = data_combined
        label_combined1 = create_label_frame(file_label1)
        data_y1 = to_categorical(label_combined1,num_classes=globalVar.num_class_digits)

        x_train1,x_test,y1_train1,y1_test = train_test_split(data_3D,data_y1,test_size=0.2,shuffle=True)
        x_train,x_val,y1_train,y1_val = train_test_split(x_train1, y1_train1,test_size=0.2,shuffle=True)
        return x_train, y1_train, x_val, y1_val, x_test, y1_test

      def preprocess_data():
        file_data, file_label1= get_data_files()
        No_sensors = ['sc1','sc2','sc3','sc4','sc5','sc6','sc7','sc8','sc9']
        num_sensors = len(No_sensors)
        x_train, y1_train, x_val, y1_val, x_test, y1_test = create_frame_data_label(file_data, file_label1)
        return x_train, y1_train, x_val, y1_val, x_test, y1_test

      self.x_train, self.y1_train, self.x_val, self.y1_val, self.x_test, self.y1_test = preprocess_data()
      globalVar.x_train = self.x_train
      globalVar.x_val = self.x_val
      globalVar.x_test = self.x_test
      globalVar.y1_train = self.y1_train
      globalVar.y1_val = self.y1_val
      globalVar.y1_test = self.y1_test

    def train_dataset(self) :
        return [self.x_train,self.y1_train]

    def validation_dataset(self):
        return [self.x_val,self.y1_val]

    def test_dataset(self):
        return[self.x_test,self.y1_test]

    # line 7
    def update_dataset(self,model_list):

        def adjust_dataset_rate(rate):
            print("shape in update_dataset",np.array(globalVar.x_train).shape,np.array(globalVar.x_val).shape,np.array(globalVar.x_test).shape)
            max_val = np.max(globalVar.x_train)
            data_list_new = []
            for data_aux in [globalVar.x_train, globalVar.x_val, globalVar.x_test]:
              data_new = []
              for index in range(len(data_aux)): # type: ignore
                data_new.append(signal.resample(data_aux[index], (rate*6))) # type: ignore
              data_list_new.append(data_new)
            self.x_train = np.clip(data_list_new[0], None, max_val)
            self.x_val = np.clip(data_list_new[1], None, max_val)
            self.x_test= np.clip(data_list_new[2], None, max_val)
            print("shape aft adjust in rate: {0}".format(np.array(self.x_train).shape))

        def adjust_dataset_resolution(reso):
            data_list_reso = []
            times = 0
            if reso == 8:
              times = 0.25
            elif reso == 10:
              times = 1
            elif reso == 12:
              times = 4
            for data_aux in [self.x_train, self.x_val, self.x_test]:
              data_new = []
              for index in range(len(data_aux)):
                data_new.append((data_aux[index]*times))
              data_list_reso.append(data_new)
            self.x_train = data_list_reso[0]
            self.x_val = data_list_reso[1]
            self.x_test = data_list_reso[2]
            print("shape aft adjust in reso: {0}".format(np.array(self.x_train).shape))

        def adjust_dataset_channel(chan):
            list_cha = chan
            data_list_cha = []
            for data_aux in [self.x_train, self.x_val, self.x_test]:
              arr_aux = np.array(data_aux)
              data_list_cha.append(arr_aux[:,:,list_cha,:])
            self.x_train = data_list_cha[0]
            self.x_val = data_list_cha[1]
            self.x_test = data_list_cha[2]
            print("shape aft adjust in chan: {0}".format(np.array(self.x_train).shape))


        rate = model_list['sample'][0]
        reso = model_list['sample'][1]
        chan = model_list['sample'][2]
        print("rate is {0}, reso is {1}, chan is {2}".format(rate,reso,chan))
        adjust_dataset_rate(rate)
        adjust_dataset_resolution(reso)
        adjust_dataset_channel(chan)

    @property
    def num_classes(self):
        class_list=[11,27,31]
        return class_list[self.classCato]
    @property
    def input_shape(self):
        return (self.x_train.shape[1],self.x_train.shape[2],self.x_train.shape[3])
