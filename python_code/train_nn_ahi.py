import tensorflow as tf
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam #, AdamW
import random
import sys
import numpy as np
import pandas as pd
import math
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
import os
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.platform import tf_logging as logging
from tensorflow.keras.metrics import AUC
pr_metric = AUC(curve='PR', num_thresholds=100) # The higher the threshold value, the more accurate it is calculated.
import json
import time

#from tensorflow.keras import mixed_precision

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

#np.random.seed(42)
#random.seed(42)


# Weights and Biases related imports
import wandb
from wandb.keras import WandbMetricsLogger

#  LABELS
#  AHI - 2
#  NOT AHI - 1
# AWAKE - 0


#  CHANNELS
# channels{2}={'ABDO','ABDOMINAL','ABD','ABDORES','ABDOMEN'}; %               ABDOMINAL
# channels{3}={'THOR','THORACIC','CHEST','THORRES','THORAX'}; %               CHEST
# channels{4}={'FLOW','AUX','CANNULAFLOW','NASALFLOW','NEWAIR', 'AIRFLOW' }; %        NASSAL
# channels{5}={'SPO2','SAO2','SPO2'}; %                                       O2
# channel{7}={'SPO2'}; %   delay 10  
# channel{8}={'SPO2'}; %   delay 15
# channel{8}={'SPO2'}; %   delay 20 
# channel{8}={'SPO2'}; %   delay 25 

#min_frequency{2}=32;
#min_frequency{3}=32;
#min_frequency{4}=64;
#min_frequency{5}=1;
#min_frequency{7}=1;
#min_frequency{8}=1;
#min_frequency{9}=1;
#min_frequency{10}=1;

fs_c= list([0, 32, 32, 64, 1, 0, 1, 1, 1, 1])
fs_c= list([64, 64, 64, 64, 64, 64, 64, 64, 64, 64])

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



def find_and_select_indices(labels):
    ones_indices = [i for i, label in enumerate(labels) if label == 1]
    zeros_indices = [i for i, label in enumerate(labels) if label == 0]
    num_ones = len(ones_indices)
    num_zeros = len(zeros_indices)
    num_selected_zeros = min(20 * num_ones, num_zeros)
    selected_zeros_indices = random.sample(zeros_indices, num_selected_zeros)
    selected_indices = ones_indices + selected_zeros_indices
    sorted_indices = sorted(selected_indices)
    return sorted_indices


def generator_sleep_dict(files_dict,batch_size,chan,src): 
    while True:
        current_file, data = random.choice(list(files_dict.items()))
        print(current_file)
        toc()
        tic()
        try:
            n_select = data["n_select"]
            inx_select = random.sample(data["sleep"], n_select) + random.sample(data["ahi"], n_select)
            print(len(inx_select))
            inx_select = sorted(inx_select)
            x = []
            with h5py.File(src+"DATA_5s/"+current_file, 'r') as hf:
                y = tf.convert_to_tensor(to_categorical(hf["y"][inx_select,0]))
                x = tf.convert_to_tensor(hf["x"+str(chan)][inx_select,:] )
                
            # Create a permutation of indices
            indices = np.arange(n_select)
            shuffled_indices = tf.random.shuffle(indices)
            # Shuffle tensors
            x = tf.gather(x, shuffled_indices)
            y = tf.gather(y, shuffled_indices)
            
            if tf.math.is_nan(x[0,0]):
                print(current_file)
                continue
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
        

def generator(files,batch_size,chan): 
    last_file=1
    random.shuffle(files)
    while True:
        if not(last_file):          
            for current_file in files:
                try:
                    with h5py.File(current_file, 'r') as hf:
                            y=hf["y"][:,0]
                            
                            # Find 20 negative samples per positive
                            inx=find_and_select_indices(y)
                            y=hf["y"][inx,0]
                            
                            if not any(y):
                                continue
                            
                            y = to_categorical(y[:])
                            x= tf.convert_to_tensor(hf["x"+str(chan)][inx,:] )
                            if tf.math.is_nan(x[0,0]):
                                print(current_file)
                                continue
                            
                            y= tf.convert_to_tensor( y ) 
                            # Create a permutation of indices
                            indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
                            shuffled_indices = tf.random.shuffle(indices)
                            # Shuffle tensors
                            x = tf.gather(x, shuffled_indices)
                            y = tf.gather(y, shuffled_indices)
      
                            if len(y)<batch_size:
                                yield x[:], y[:] 
                            else:
                                for ii in range( math.floor(len(y)/batch_size) + 1):
                                    if (ii+1)*batch_size >= len(y):
                                        yield  x[ii*batch_size:], y[ii*batch_size:] 
                                        break
                                    else:
                                        yield x[ii*batch_size:(ii+1)*batch_size], y[ii*batch_size:(ii+1)*batch_size] 
                except:
                    pass
            last_file=1
        else:
            last_file=0 
            



def get_model(fs,tw,lr):
    dim=fs * tw
    model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_tensor=tf.keras.layers.Input(shape=(dim, 1,1))
    )
    output = model.layers[-1].output
    output= tf.keras.layers.GlobalAveragePooling2D()(output)
    output= tf.keras.layers.Dropout(0.2)(output)
    output=tf.keras.layers.Dense(2, activation="softmax")(output)
    model_out = tf.keras.models.Model(inputs=model.input, outputs=output) 
    model_out.compile(Adam(learning_rate=lr), loss='categorical_crossentropy',metrics=[pr_metric, "accuracy"])
    return model_out



def get_steps(val_f,batch_size):
    count=0
    for file_input in val_f:
        try:
            with h5py.File(file_input, 'r') as hf:
                y=hf["y"][:,0]
                count += len(y[y>=0])
        except:
            pass
    total_samp=round(count/batch_size)     
    return total_samp






if __name__ == '__main__':
    ## INPUTS
    # # Parameters
    batch_size=int(sys.argv[1] ) 
    fold=int(sys.argv[2]  )
    chan=sys.argv[3] 
    tw=int(sys.argv[4] )
    new_model= int(sys.argv[5] )
    src = sys.argv[6]
    
    epochs=100
    lr=1e-5
    
    
    # GET FILES 
    file_train = src+"SPLIT/dict_TRAIN_split"+str(fold)+".json"
    with open(file_train, "r") as f:
        files_dict_train = json.loads(f.read())
    
    file_val = src+"SPLIT/dict_VAL_split"+str(fold)+".json"
    with open(file_val, "r") as f:
        files_dict_val = json.loads(f.read())
 
    # WEIGHTS FOLDER
    src_w=src+"WEIGHTS_2out_ahi_"+str(fold)+"/"
    try:
        os.mkdir(src_w)    
    except:
        pass
    

    # Get Steps for training  (Mini-Batch)
    train_s = 1024
    val_s = 256
        
    print(train_s,val_s )
    print("FOLD " + str(fold) + " CHANNEL " + str(chan ) + " VAL_STEPS " + str(val_s) )
    print("TRAIN _ VAL _ TEST size=" + str(len(files_dict_train))  +" " +  str(len(files_dict_val))     )


 
    chan=int(chan)
    fs=fs_c[chan-1]

    f_w= "WEIGHTS_eff_ahi_1D_"
    
    
    # TRAINING CALLBACKS
    # Initialize a W&B run
    
    configs = dict(
    num_classes = 2,
    batch_size = batch_size,
    image_size = fs,
    image_channels = 1,
    learning_rate = lr,
    epochs = epochs
    )
    
    
    run = wandb.init(
        project = "OSA",
        config = configs,
        group = "Trial_"+str(fold)+"_ahiONLY",
        name = "Trial_"+str(fold)+"_ahi_"+str(tw)+"_ch_"+str(chan)+"_r3",
    )
    
    #tf.keras.backend.clear_session()
    
    # TRAINING CALLBACKS
    reduce_lr = ReduceLRBacktrack(best_path=src_w+f_w+str(fold)+"_"+str(chan)+".hdf5", monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6) 
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True)
    checkpoint = ModelCheckpoint(src_w+f_w+str(fold)+"_"+str(chan)+".hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [reduce_lr, early_stop, checkpoint, WandbMetricsLogger(log_freq=10)] 
    
    
    if new_model==1:
        # GET MODEL
        print("NEW MODEL WITH RAMDOM WEIGHTS")
        model= get_model(fs,tw,lr )
        
        
    
    elif new_model ==0:
        # LOAD MODEL
        try:
            print("LOADING OLD MODEL WITH WEIHTS IN: "+ src_w+f_w+str(fold)+"_"+str(chan)+".hdf5")
            model = tf.keras.models.load_model(src_w+f_w+str(fold)+"_"+str(chan)+".hdf5")
            model.compile(Adam(learning_rate=lr), loss='categorical_crossentropy',metrics=[pr_metric, "accuracy"])
        except:
            print("OLD MODEL NOT FOUND")
            print("NEW MODEL WITH RAMDOM WEIGHTS")
            model= get_model(fs,tw,lr )
        
        
    else:
        print("CHOOSE IF STARTING NEW MODEL (1) OR OLD (0)")
    
    model.fit(
        generator_sleep_dict(files_dict_train,batch_size,chan,src),
        steps_per_epoch=train_s,
        epochs=epochs,
        shuffle=False,
        validation_data=generator_sleep_dict(files_dict_val,batch_size,chan,src),
        validation_steps=val_s,
        callbacks=callbacks_list,
        verbose=2)

    



