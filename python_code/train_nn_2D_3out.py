import tensorflow as tf
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
import random
import sys
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
import os
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.platform import tf_logging as logging
from tensorflow.keras.metrics import AUC
import time
import math
import json


# Weights and Biases related imports
import wandb
from wandb.keras import WandbMetricsLogger




pr_metric = AUC(curve='PR', num_thresholds=100) # The higher the threshold value, the more accurate it is calculated.


#np.random.seed(42)
#random.seed(42)


#  CHANNELS
# channel{1}={'EKG','ECGL','ECG','ECGLECGR'};%                               ECG
# channel{2}={'ABDO','ABDOMINAL','ABD','ABDORES','ABDOMEN'}; %               ABDOMINAL
# channel{3}={'THOR','THORACIC','CHEST','THORRES','THORAX'}; %               CHEST
# channel{4}={'FLOW','AUX','CANNULAFLOW','NASALFLOW','NEWAIR','AIRFLOW'}; %        NASSAL
# channel{5}={'SPO2','SAO2','SPO2'}; %         
# channel{6}={'RRI}; %     
# channel{7}={'SPO2'}; %   delay 10  
# channel{8}={'SPO2'}; %   delay 15
# channel{8}={'SPO2'}; %   delay 20 
# channel{8}={'SPO2'}; %   delay 25  



# from numba import jit
def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print(" ================================ Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.==========================================")
    else:
        print("Toc: start time not set")
        

class ReduceLRBacktrack(ReduceLROnPlateau):
    def __init__(self, best_path, *args, **kwargs):
        super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
        self.best_path = best_path
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                             self.monitor, ','.join(list(logs.keys())))
        if not self.monitor_op(current, self.best): # not new best
            if not self.in_cooldown(): # and we're not in cooldown
                if self.wait+1 >= self.patience: # going to reduce lr
                    # load best model so far
                    print("Backtracking to best model before reducting LR")
                    self.model.load_weights(self.best_path)
        super().on_epoch_end(epoch, logs) # actually reduce LR
        

def find_and_select_indices_sleep(labels):
    twos_indices = [i for i, label in enumerate(labels) if label == 2] #sleep - ahi
    ones_indices = [i for i, label in enumerate(labels) if label == 1] #sleep - noahi
    zeros_indices = [i for i, label in enumerate(labels) if label == 0] #awake
    num_twos = min(len(twos_indices),64)
    num_ones = len(ones_indices)
    num_zeros = len(zeros_indices)
    #print(num_zeros, num_ones, num_twos)
    if num_twos > 0:
        num_selected_ones = min(num_twos, num_ones)
        num_selected_zeros = min(num_twos, num_zeros)
        selected_twos_indices = random.sample(twos_indices, num_twos)
        selected_ones_indices = random.sample(ones_indices, num_selected_ones)
        selected_zeros_indices = random.sample(zeros_indices, num_selected_zeros)
        selected_indices = selected_twos_indices + selected_ones_indices + selected_zeros_indices
    else:
        selected_ones_indices = random.sample(ones_indices, 5)
        selected_zeros_indices = random.sample(min(num_zeros, 5))
        selected_indices = selected_ones_indices + selected_zeros_indices
    
    sorted_indices = sorted(selected_indices)
    return sorted_indices



def get_steps(val_f,batch_size):
    count=0
    for current_file in val_f:
        try:
            with h5py.File(current_file, 'r') as hf:
                y=hf["y"][:,0]
            
            count += len(y)
        except:
            pass
        
    
    total_samp=round(count/batch_size)     
    return total_samp


def generator_sleep(files,batch_size,channels_id): 
    last_file=1
    random.shuffle(files)
    while True:
        if not(last_file):          
            for current_file in files:
                print(current_file)
                toc()
                tic()
                try:
                    with h5py.File(current_file, 'r') as hf:
                        y=hf["y"][:,0]
                        s=hf["sleep_label"][:,0]
                        label = y+s
                        # Find 20 negative samples per positive
                        inx=find_and_select_indices_sleep(label)
                        print(len(inx))
                        label=label[inx]
                        label = to_categorical(label[:])
                        x = []
                        for chan in channels_id:
                            print(chan)
                            x.append(hf["x"+chan][inx,:])
                        
                    
                    x= tf.convert_to_tensor(x)
                    #x= tf.transpose(x, [1, 2, 0, 3])
                    x= tf.transpose(x, [1, 2, 3, 0])
                    label= tf.convert_to_tensor(label) 
                    # Create a permutation of indices
                    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
                    shuffled_indices = tf.random.shuffle(indices)
                    # Shuffle tensors
                    x = tf.gather(x, shuffled_indices)
                    label = tf.gather(label, shuffled_indices)
                    if len(y)<batch_size:
                        yield x[:], label[:] 
                    else:
                        for ii in range(math.ceil(len(label)/batch_size)):
                            if (ii+1)*batch_size >= len(label):
                                yield  x[ii*batch_size:], label[ii*batch_size:] 
                                break
                            else:
                                yield x[ii*batch_size:(ii+1)*batch_size], label[ii*batch_size:(ii+1)*batch_size] 
                except:
                    pass
            last_file=1
        else:
            last_file=0 

         

def generator_sleep_dict(files_dict,batch_size,channels_id,src): 
    while True:
        current_file, data = random.choice(list(files_dict.items()))
        print(current_file)
        toc()
        tic()
        try:
            n_select = data["n_select"]
            inx_select = random.sample(data["awake"], n_select) + random.sample(data["sleep"], n_select) + random.sample(data["ahi"], n_select)
            print(len(inx_select))
            inx_select = sorted(inx_select)
            x = []
            with h5py.File(src+"DATA/"+current_file, 'r') as hf:
                y = tf.convert_to_tensor(to_categorical(hf["label_y_s"][inx_select,0]))
                #for chan in channels_id:
                #    print(chan)
                #    x.append(hf["x"+chan][inx_select,:])
                x = hf["X"][:,inx_select,:]
            x= tf.convert_to_tensor(x[channels_id,:,:])
            x= tf.transpose(x, [1, 2, 3, 0])
            # Create a permutation of indices
            indices = np.arange(n_select)
            shuffled_indices = tf.random.shuffle(indices)
            # Shuffle tensors
            x = tf.gather(x, shuffled_indices)
            y = tf.gather(y, shuffled_indices)
            if len(y)<batch_size:
                yield x[:], y[:] 
            else:
                for ii in range(math.ceil(len(y)/batch_size)):
                    if (ii+1)*batch_size >= len(y):
                        yield  x[ii*batch_size:], y[ii*batch_size:] 
                        break
                    else:
                        yield x[ii*batch_size:(ii+1)*batch_size], y[ii*batch_size:(ii+1)*batch_size] 
        except Exception as e: 
            print(e)
            pass

        

def get_optimizer(lr=1e-5, optimizer="adam"):
    "Select optmizer between adam and sgd with momentum"
    if optimizer.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if optimizer.lower() == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.1)

def get_model_sleep(fs, tw, n_channels, lr=1e-5, optimizer='adam'):
    dim=fs * tw
    model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_tensor=tf.keras.layers.Input(shape=(dim, 1, n_channels))
    )
    output = model.layers[-1].output
    output= tf.keras.layers.GlobalAveragePooling2D()(output)
    output= tf.keras.layers.Dropout(0.2)(output)
    output=tf.keras.layers.Dense(3, activation="softmax")(output)
    model_out = tf.keras.models.Model(inputs=model.input, outputs=output) 
    
    model_out.compile(optimizer=get_optimizer(lr, optimizer), loss='categorical_crossentropy',metrics=[pr_metric, "accuracy"])    
    return model_out



if __name__ == '__main__':
    wandb.login()
    ## INPUTS
    # # Parameters
    batch_size = int(sys.argv[1]) 
    fold = int(sys.argv[2])
    channels =  sys.argv[3]  #Needs to be a vector now '[2,3]'
    tw = int(sys.argv[4] )
    
    new_model = int(sys.argv[5])
    src = sys.argv[6] # src  = "/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/"

    epochs=100
    lr=1e-5
    # fs_c= list([0, 64, 64, 64, 64, 0, 64, 64, 64, 64])
    fs = 64 #[fs_c[int(x)] for x in channels_id][0]  # They are all the same now

    # GET FILES
    # JSON file
    file_train = src+"SPLIT/dict_TRAIN_split"+str(fold)+".json"
    with open(file_train, "r") as f:
        files_dict_train = json.loads(f.read())
    
    file_val = src+"SPLIT/dict_VAL_split"+str(fold)+".json"
    with open(file_val, "r") as f:
        files_dict_val = json.loads(f.read())
    
    
    """
    train_files = src+"SPLIT/TRAIN_split"+str(fold)+".txt"
    with open(train_files, 'r') as f:
        files = f.readlines()
    
    train_f = np.array([src+"DATA/"+file[:-1] for file in files])
     
    val_files = src+"SPLIT/VAL_split"+str(fold)+".txt"
    with open(val_files, 'r') as f:
        files = f.readlines()
        
    val_f = np.array([src+"DATA/"+file[:-1] for file in files])
    """

    # WEIGHTS FOLDER
    src_w=src+"WEIGHTS_3out_X_"+str(fold)+"/"
    try:
        os.mkdir(src_w)    
    except:
        pass
        

    # Get Steps for training  (Mini-Batch)
    #val_s = get_steps(val_f,batch_size)
    #train_s= val_s/5 # if bigger it takes too long to do the epoch. train_s = val_s*2
    #val_s = val_s/10
    train_s = 1000
    val_s = 250

    
    
    print("FOLD " + str(fold) + " CHANNELS " + channels + " TRAIN_STEPS " + str(train_s) + " VAL_STEPS " + str(val_s) )
    print("TRAIN _ VAL _ TEST size=" + str(len(files_dict_train))  +" " +  str(len(files_dict_val))     )

    channels_id = channels[1:-1].split(',')
    f_w= "WEIGHTS_eff_ahi_2D_"
    weights_name = src_w+f_w+"F"+str(fold)+"_Ch_"+"_".join(channels_id)+".hdf5"
    print(weights_name)
    
    if '5' in channels_id:
        channels_id.extend(['7', '8', '9', '10'])
    
    
    # TRAINING CALLBACKS
    # Initialize a W&B run
    
    configs = dict(
    num_classes = 3,
    batch_size = batch_size,
    image_size = fs,
    image_channels = len(channels_id),
    learning_rate = lr,
    epochs = epochs
    )
    
    
    run = wandb.init(
        project = "OSA",
        config = configs,
        group = "Trial_"+str(fold)+"_newSF",
        name = "Trial_"+str(fold)+"_ch_"+"_".join(channels_id)+"_X",
    )
    
    reduce_lr = ReduceLRBacktrack(best_path=weights_name, monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6) 
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True)
    checkpoint = ModelCheckpoint(weights_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [reduce_lr, early_stop, checkpoint, WandbMetricsLogger(log_freq=10)] 
    
    
    
    if new_model==1:
        # GET MODEL
        print("NEW MODEL WITH RAMDOM WEIGHTS")
        model= get_model_sleep(fs, tw, len(channels_id), lr)
        
    
    elif new_model ==0:
        # LOAD MODEL
        try:
            print("LOADING OLD MODEL WITH WEIHTS IN: "+ weights_name)
            model = tf.keras.models.load_model(weights_name)
            model.compile(Adam(learning_rate=lr), loss='categorical_crossentropy',metrics=[pr_metric, "accuracy"])
        except:
            print("OLD MODEL NOT FOUND")
            print("NEW MODEL WITH RAMDOM WEIGHTS")
            model= get_model_sleep(fs, tw, len(channels_id), lr)
            

    else:
        print("CHOOSE IF STARTING NEW MODEL (1) OR OLD (0)")
    
    ch_new_indx = [10, 10, 0, 1, 2, 3, 10, 4, 5, 6, 7]   
    ch_new_indx = [ch_new_indx[int(i)] for i in channels_id]
    
    model.fit(
        generator_sleep_dict(files_dict_train,batch_size,ch_new_indx ,src),
        steps_per_epoch=train_s,
        epochs=epochs,
        shuffle=False,
        validation_data=generator_sleep_dict(files_dict_val,batch_size,ch_new_indx ,src),
        validation_steps=val_s,
        callbacks=callbacks_list,
        verbose=2
    )

    



