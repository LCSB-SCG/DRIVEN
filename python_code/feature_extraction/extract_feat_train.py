import tensorflow as tf
import h5py
import random
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import os
import time
import random
import time
import glob
import shutil
import json
#np.random.seed(42) 
#random.seed(42)


def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print(" ================================ Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.==========================================")
    else:
        print("Toc: start time not set")



def EXTRACT_FEATURES(fold, src_data, src_result, models, channels_id, current_file, dim, inx_select): 
    file_out = src_result+current_file
    n_f_d=str(np.char.replace(file_out,'hdf5','delete')) 
    if  os.path.exists( file_out ) or os.path.exists( n_f_d ):
        return
    
    fd = h5py.File(n_f_d, mode='w') 
    fd = open(n_f_d, "w")
    fd.close()   
    f1 = h5py.File(file_out, mode='w')
    f1.create_dataset("x", (1, dim*len(models) ), np.float32,maxshape=(None,dim*len(models)))
    f1.create_dataset("y", (1,2), np.float32,maxshape=(None,2))
    indx=0
    batch_size = 2
    n_feat=len(models)
    try:
        tic()
        with h5py.File(src_data+current_file, 'r') as hf:
            y=hf["y"][inx_select,0]
            if not any(y):
                print("Not any positive in " + current_file)
                os.remove(n_f_d)
                return
            
            y = to_categorical(y[:])
            feats=np.zeros(shape=[len(y),dim*len(models)])
            for id_chan in range(n_feat):
                x = tf.convert_to_tensor(hf["x" + channels_id[id_chan]][inx_select, :])
                num_samples = x.shape[0]
                num_batches = num_samples // batch_size
                cnn_output = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    batch_x = x[start_idx:end_idx]
                    batch_output = np.array(eval("models[" + str(id_chan) + "].predict(batch_x)"))
                    cnn_output.append(batch_output)
                
                # If there are remaining samples that don't fit in a complete batch, process them separately
                if num_samples % batch_size != 0:
                    remaining_samples = x[num_batches * batch_size:]
                    remaining_output = np.array(eval("models[" + str(id_chan) + "].predict(remaining_samples)"))
                    cnn_output.append(remaining_output)
                
                cnn_output = np.concatenate(cnn_output, axis=0)           
                #########################
                # cnn_output=np.array( eval("model"+str(lll)+".predict(x)")   )
                feats[list(range(len(y))), (id_chan)*dim:(id_chan+1)*dim] = cnn_output[:]
            
          
        kk = list(range(len(y)))    
        f1["x"].resize(len(y),axis=0)
        f1["y"].resize(len(y),axis=0)
        f1["x"][kk, ...] = feats[:]
        f1["y"][kk, ...] = y[:]
        print( current_file + "_Finished_TOTAL_SAMPLES: " +str(len(y)))
        os.remove(n_f_d)
        toc()
    except Exception as e:
        print(" ERROR: " + current_file + " Message: " + str(e))
        return
    
    return



def EXTRACT_FEATURES_SLEEP(fold, src_data, src_result, models, channels_id, current_file, dim, inx_select): 
    file_out = src_result+current_file
    n_f_d=str(np.char.replace(file_out,'hdf5','delete')) 
    if  os.path.exists( file_out ) or os.path.exists( n_f_d ):
        return
    
    fd = h5py.File(n_f_d, mode='w') 
    fd = open(n_f_d, "w")
    fd.close()   
    f1 = h5py.File(file_out, mode='w')
    f1.create_dataset("x", (1, dim*len(models) ), np.float32,maxshape=(None,dim*len(models)))
    f1.create_dataset("sleep_label", (1,2), np.float32,maxshape=(None,2))
    indx=0
    batch_size = 2
    n_feat=len(models)
    try:
        tic()
        with h5py.File(src_data+current_file, 'r') as hf:
            y=hf["sleep_label"][inx_select,0]
            if not any(y):
                print("Not any positive in " + current_file)
                os.remove(n_f_d)
                return
            
            y = to_categorical(y[:])
            feats=np.zeros(shape=[len(y),dim*len(models)])
            for id_chan in range(n_feat):
                x = tf.convert_to_tensor(hf["x" + channels_id[id_chan]][inx_select, :])
                num_samples = x.shape[0]
                num_batches = num_samples // batch_size
                cnn_output = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = (i + 1) * batch_size
                    batch_x = x[start_idx:end_idx]
                    batch_output = np.array(eval("models[" + str(id_chan) + "].predict(batch_x)"))
                    cnn_output.append(batch_output)
                
                # If there are remaining samples that don't fit in a complete batch, process them separately
                if num_samples % batch_size != 0:
                    remaining_samples = x[num_batches * batch_size:]
                    remaining_output = np.array(eval("models[" + str(id_chan) + "].predict(remaining_samples)"))
                    cnn_output.append(remaining_output)
                
                cnn_output = np.concatenate(cnn_output, axis=0)           
                #########################
                # cnn_output=np.array( eval("model"+str(lll)+".predict(x)")   )
                feats[list(range(len(y))), (id_chan)*dim:(id_chan+1)*dim] = cnn_output[:]
            
          
        kk = list(range(len(y)))    
        f1["x"].resize(len(y),axis=0)
        f1["sleep_label"].resize(len(y),axis=0)
        f1["x"][kk, ...] = feats[:]
        f1["sleep_label"][kk, ...] = y[:]
        print( current_file + "_Finished_TOTAL_SAMPLES: " +str(len(y)))
        os.remove(n_f_d)
        toc()
    except Exception as e:
        print(" ERROR: " + current_file + " Message: " + str(e))
        return
    
    return

if __name__ == '__main__':
    ## INPUTS
    # # Parameters
    
    fold=int(sys.argv[1])
    src=sys.argv[2]
    predict =  sys.argv[3] #ahi or sleep
    last_layer = int(sys.argv[4])
    stride = str(sys.argv[5])

    channels_id = ["2", "3", "4", "5", "7", "8", "9", "10"]

    f_w= "WEIGHTS_eff_"+predict+"_1D_"
    
    if last_layer == 3:
        s_w= "FEAT_DATA_"+str(fold)+"_"+predict+"/"
    
    elif last_layer == 1:
        s_w= "FEAT_DATA_"+str(fold)+"_"+predict+"_2Out/"
        
        
    src_data=src+"DATA_"+stride+"s/"


    src_w_old=src+"WEIGHTS_2out_"+predict+"_"+str(fold)+"/"
    src_w = src+"WEIGHTS_2out_"+predict+"_"+str(fold)+"_all/"
   
    try:
        shutil.copytree(src_w_old, src_w)
    except:
        pass
    
    src_o=src+s_w

    file_train = src+"SPLIT/dict_TRAIN_split"+str(fold)+".json"
    with open(file_train, "r") as f:
        files_dict_train = json.loads(f.read())
    
    train_f_all = list(files_dict_train.keys())

    print("TOTAL_FILES: " + str(len(train_f_all)))
    try:
        os.mkdir(src_o)    
    except:
        pass        
    # last_layer=-2
    #if last_layer == 3:
        #last_layer=-3 # PART THAT CHANGES A LOT STUFFFFF 
    # # LOAD MODELS
    ## 1
    models = []
    for model_id in channels_id:
        model_name =src_w+f_w+str(fold)+"_"+model_id+".hdf5"
        model = tf.keras.models.load_model(model_name)
        if last_layer == 3:
            model= Model(inputs=model.input, outputs=model.layers[-last_layer].output)
        
        models.append(model)
        

    dim=models[-1].layers[-1].output_shape[-1]
    print("DIMMMMM")
    print(dim)
    
    # ## TRAINING
    src_result= src_o+'TRAINING/'
    try:
        os.mkdir(src_result)    
    except:
        pass 
     
    total_created_files = glob.glob(src_result + "*.hdf5")           
    while len(total_created_files)<len(train_f_all):
        total_created_files = glob.glob(src_result + "*.hdf5") 
        current_file = random.choice(train_f_all)
        data = files_dict_train[current_file]
        
        if predict == "ahi":
            inx_select = random.sample(data["sleep"], data["n_select"]) + random.sample(data["ahi"], data["n_select"])
            inx_select = sorted(inx_select)    
            EXTRACT_FEATURES(fold, src_data, src_result, models, channels_id, current_file, dim, inx_select)
        elif predict == "sleep":
            inx_select =random.sample(data["awake"], n_select) + random.sample(data["sleep"], int(n_select/2)) + random.sample(data["ahi"],  int(n_select/2))
            inx_select = sorted(inx_select)    
            EXTRACT_FEATURES_SLEEP(fold, src_data, src_result, models, channels_id, current_file, dim, inx_select)
        else: 
            print("Choose ahi or sleep")
            
    
    ## VALIDATION
    
    file_val = src+"SPLIT/dict_VAL_split"+str(fold)+".json"
    with open(file_val, "r") as f:
        files_dict_val= json.loads(f.read())
    
    val_f_all = list(files_dict_val.keys())
    print("TOTAL_VAL_FILES: " + str(len(val_f_all)))
    
    src_result= src_o+'VALIDATION/'
    try:
        os.mkdir(src_result)    
    except:
        pass  

  
    total_created_files= glob.glob(src_result + "*.hdf5")           
    while len(total_created_files)<len(val_f_all):
        total_created_files= glob.glob(src_result+"*.hdf5")  
        current_file = random.choice(val_f_all)
        data = files_dict_val[current_file]
        
        if predict == "ahi":
            inx_select = random.sample(data["sleep"], data["n_select"]) + random.sample(data["ahi"], data["n_select"])
            inx_select = sorted(inx_select)    
            EXTRACT_FEATURES(fold, src_data, src_result, models, channels_id, current_file, dim, inx_select)
        
        elif predict == "sleep":
            inx_select =random.sample(data["awake"], n_select) + random.sample(data["sleep"], int(n_select/2)) + random.sample(data["ahi"],  int(n_select/2))
            inx_select = sorted(inx_select)    
            EXTRACT_FEATURES_SLEEP(fold, src_data, src_result, models, channels_id, current_file, dim, inx_select)
    
        else: 
            print("Choose ahi or sleep")
    print("__________________ OVER _________________" )
