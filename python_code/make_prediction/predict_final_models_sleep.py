import tensorflow as tf
import h5py
import random
import sys
import numpy as np
import pandas as pd
import glob
from tensorflow.keras.models import Model
import os
import joblib
import time
import math 
from itertools import combinations


def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print(" ================================ Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.==========================================")
    else:
        print("Toc: start time not set")
        
     
def model_1_sleep2(src_data, src_features, src_result, current_file, sensors): 
    with h5py.File(src_data+current_file, 'r') as hf:
        y_real = np.array(hf["y"][:,0] ) 
        y_sleep = np.array(hf["sleep_label"][:,0] )
    
    #feats_id = 1 #  GET FEATURES ID in order
    with h5py.File(src_features+current_file, 'r') as hf:
        y_sleep1 = np.array(hf["sleep_label"][:,1] ) 
        x_all = np.array(hf["x"][:,:])
        #print(x_all.shape) #[inx_feat, :])
    
    #print(y_real1)
    #print(y_real)
    if not np.array_equal(y_sleep1, y_sleep): #y_real1 != y_real:
        print("NOT THE SAME")
        print(current_file)
        return
    
    for i in range(len(sensors)):
        #print(sens[i])
        sleep_pred = x_all[:,i*2:i*2+2]
        #print(y_pred)
        sleep_pred = sleep_pred[:,1]
        #print(y_pred.shape)
        src_result_model_file = src_result+str(sensors[i])+"_sleep/"+current_file
        f1 = h5py.File(src_result_model_file, mode='w')
        f1.create_dataset("sleep_pred", data=sleep_pred)
        f1.create_dataset("y_sleep", data=y_sleep) 
        f1.close()
    return

                    
def model_2_sleep2(model_lgb, src_data, src_features, src_result, current_file, inx_feat):
    with h5py.File(src_data+current_file, 'r') as hf:
        y_real = np.array(hf["y"][:,0] ) 
        y_sleep = np.array(hf["sleep_label"][:,0] )
    
    #feats_id = 1 #  GET FEATURES ID in order
    with h5py.File(src_features+current_file, 'r') as hf:
        y_sleep1 = np.array(hf["sleep_label"][:,1] ) 
        x_all = np.array(hf["x"][:,inx_feat])
        #print(x_all.shape) #[inx_feat, :])
    
    #print(y_real1)
    #print(y_real)
    if not np.array_equal(y_sleep1, y_sleep): #y_real1 != y_real:
        print("NOT THE SAME")
        print(current_file)
        return
    
    sleep_pred = model_lgb.predict_proba(x_all)
    sleep_pred = sleep_pred[:,1]
    #y_pred.extend(pred[:,1])
    
    f1 = h5py.File(src_result+current_file, mode='w') 
    f1.create_dataset("sleep_pred", data=sleep_pred)
    f1.create_dataset("y_sleep", data=y_sleep) 
    f1.close()
    return



def model_4_sleep(model, src_data, src_result, current_file, sensors_id):
    f1 = h5py.File(src_result+current_file, mode='w') 
    y_pred = [] 
    x_all=[]
    with h5py.File(src_data+current_file, 'r') as hf:
        y_real = np.array(hf["y"][:,0] ) 
        x_all = np.array(hf["X"][sensors_id, :])
        y_sleep = np.array(hf["sleep_label"][:,0] )
    
    if len(x_all[0])==0:
        print("ERROR len < 0" + current_file)
        return
    
    #print(x_all[0].shape)
    n_windows=len(x_all[0])
    #print(n_windows)
    batch_size=2
    for w in range( math.ceil( n_windows/batch_size ) ):
        if (w+1)*batch_size>=n_windows:
            w_limint=n_windows
            new_batch_size = n_windows-w*batch_size
        else:
            w_limint=(w+1)*batch_size
        
        x= tf.convert_to_tensor(x_all[:,w*batch_size:w_limint,:,:]) #ch, batch, ws*sf,1
        #x = tf.transpose(x, [1, 2, 0, 3]) #batch, ws*sf, ch, 1
        x= tf.transpose(x, [1, 2, 3, 0])
        pred = model.predict(x)
        y_pred.extend(pred [:,1])
    
    f1.create_dataset("y_pred", data=y_pred)
    f1.create_dataset("y_real", data=y_real)
    f1.create_dataset("y_sleep", data=y_sleep) 
    f1.close()
    return


if __name__ == '__main__':
    ## INPUTS

    fold=int(sys.argv[1])   
    src=sys.argv[2] # src = "/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/"
    data_set=sys.argv[3]
    model_type_eval = int(sys.argv[4])  
    st="40"
    
    test_files = src+"SPLIT/"+data_set+"_split"+str(fold)+".txt"
    with open(test_files, 'r') as f:
        test_f = f.readlines()

    test_f = np.array([file[:-1] for file in test_f])
    print(len(test_f))
        
    src_models = src+"WEIGHTS_2out_sleep_"+str(fold)+"_all/"
    
    ###################################################################################
    # MODEL 1
    if model_type_eval == 1:
        
        sensors=[2, 3, 4, 5, 7, 8, 9, 10]


        src_data = src+"DATA_"+st+"s/"
        src_features = src+"PREDICTIONS/FEATURES/"+data_set+"/"+st+"s/FEAT_DATA_"+str(fold)+"_sleep_2Out/"
        src_result=src+"PREDICTIONS/"+data_set+"/"+st+"s/Model1_"+str(fold)+"_Ch_"

        for i in range(len(sensors)):
            try:
                src_result_m=src+"PREDICTIONS/"+data_set+"/"+st+"s/Model1_"+str(fold)+"_Ch_"+str(sensors[i])+"_sleep"
                os.mkdir(src_result_m)    
            except:
                pass 
                
                
        np.random.shuffle(test_f)
        current_file=test_f[0]
        for current_file in test_f:
            #print(current_file)
            #tic()
            try:
                model_1_sleep2(src_data, src_features, src_result, current_file, sensors)
            except Exception as e:
                print("ERROR: " + current_file + " Message: " + str(e))
                print(current_file)    
            #toc()
            
        print(" ================================= OVER =====================================")  

            
    
    ###################################################################################
    # MODEL 2 and 3
    if model_type_eval == 2:
        
        dnn_features = 1280
        #sensors = [[2, 3, 4, 5],[7, 8, 9, 10]  ]
        #sensors_comb = [[2, 3, 4, 5], [2, 3], [2, 5], [2, 5], [5]]  
        sens=[[2, 3, 4, 5]] #ABDO - THOR - FLOW - SPO2 -  - SPO2+delays...
        
        comb_3 = list(map(list,list(combinations(sens[0], 3) )))
        comb_2 = list(map(list,list(combinations(sens[0], 2) )))
        comb_1 = list(map(list,list(combinations(sens[0], 1) )))

        sens.extend(comb_3)
        sens.extend(comb_2)
        sens.extend(comb_1)
        
        current_sens = sens[0]
        
        sens=[[4]]
        for current_sens in sens:
    
            inx_feat=np.array([] ,dtype=int)  
            
            for l in current_sens:
                inx_feat= np.append(inx_feat, list(range((l-2)*dnn_features, (l-1)*dnn_features)))  
                
            if 5 in current_sens: #spo2
                for spo in [4, 5, 6, 7]:
                    inx_feat= np.append(inx_feat, list(range(spo*dnn_features, (spo+1)*dnn_features)))  
              
            sen_names = "_".join(map(str,current_sens))
            model_name = src_models+"LGBD_MODEL_fsleep_"+str(sen_names)+"_sleep.pkl"
            model_lgb = joblib.load(model_name)

            src_data = src+"DATA_"+st+"s/"
            src_features = src+"PREDICTIONS/FEATURES/"+data_set+"/"+st+"s/FEAT_DATA_"+str(fold)+"_sleep/"
            src_result=src+"PREDICTIONS/"+data_set+"/"+st+"s/sleep/Model2_"+str(fold)+"_Ch_"+sen_names+"_sleep/"
            
            
            try:
                os.mkdir(src_result)    
            except:
                pass        
            
            #np.random.shuffle(test_f)
            current_file=test_f[0]
            for current_file in test_f:
                #print(current_file)
                n_f_o=src_result+current_file
                #n_f_d=str(np.char.replace(n_f_o, 'hdf5','delete'))
                #if  os.path.exists( n_f_o ) or os.path.exists( n_f_d ):
                #    continue
                #fd = open(n_f_d, "w")
                #fd.close()
                try:  
                    #tic()
                    model_2_sleep2(model_lgb, src_data, src_features, src_result, current_file, inx_feat)
                    #os.remove(n_f_d)
                    #toc()
                except Exception as e:
                    print("ERROR: " + current_file + " Message: " + str(e))
                    print(current_file)

            print("DONE")
        print(" ================================= OVER =====================================")
    
    ###################################################################################    
    if model_type_eval == 3:
        
        dnn_features = 2
        #sensors = [[2, 3, 4, 5],[7, 8, 9, 10]  ]
        #sensors_comb = [[2, 3, 4, 5], [2, 3], [2, 5], [2, 5], [5]]  
        sens=[[2, 3, 4, 5]] #ABDO - THOR - FLOW - SPO2 -  - SPO2+delays...
        
        comb_3 = list(map(list,list(combinations(sens[0], 3) )))
        comb_2 = list(map(list,list(combinations(sens[0], 2) )))
        comb_1 = list(map(list,list(combinations(sens[0], 1) )))

        sens.extend(comb_3)
        sens.extend(comb_2)
        sens.extend(comb_1)
        
        current_sens = sens[0]
        for current_sens in sens:
    
            inx_feat=np.array([] ,dtype=int)  
            
            for l in current_sens:
                inx_feat= np.append(inx_feat, list(range((l-2)*dnn_features, (l-1)*dnn_features)))  
                
            if 5 in current_sens: #spo2
                for spo in [4, 5, 6, 7]:
                    inx_feat= np.append(inx_feat, list(range(spo*dnn_features, (spo+1)*dnn_features)))  
              
            sen_names = "_".join(map(str,current_sens))
            model_name = src_models+"LGBD_MODEL_fsleep_"+str(sen_names)+"_m_sleep.pkl"
            model_lgb = joblib.load(model_name)

            src_data = src+"DATA_"+st+"s/"
            src_features = src+"PREDICTIONS/FEATURES/"+data_set+"/"+st+"s/FEAT_DATA_"+str(fold)+"_sleep_2Out/"
            src_result=src+"PREDICTIONS/"+data_set+"/"+st+"s/Model3_"+str(fold)+"_Ch_"+sen_names+"_sleep/"

            
            try:
                os.mkdir(src_result)    
            except:
                pass        
            
            np.random.shuffle(test_f)
            
            for current_file in test_f:
                #print(current_file)
                #tic()
                n_f_o=src_result+current_file
                #n_f_d=str(np.char.replace(n_f_o, 'hdf5','delete'))
                #if  os.path.exists( n_f_o ) or os.path.exists( n_f_d ):
                #    continue
                #fd = open(n_f_d, "w")
                #fd.close()  
                try:
                    #tic()
                    model_2_sleep2(model_lgb, src_data, src_features, src_result, current_file, inx_feat)
                    #os.remove(n_f_d)
                    #toc()
                except Exception as e:
                    print("ERROR: " + current_file + " Message: " + str(e))
                    print(current_file)

            print("DONE")
        print(" ================================= OVER =====================================")       
    
    
    ###################################################################################
    # MODEL 4
    if model_type_eval == 4:
        sensors_comb = [[2, 3, 4, 5], [2, 3], [2, 5], [3, 5], [5]]  
        sensors_comb = [[3,5]]  
        for sensors in sensors_comb:
            f_w= "WEIGHTS_eff_ahi_2D_F"
            model_name =src_models+f_w+str(fold)+"_Ch_"+"_".join([str(ch) for ch in sensors])+".hdf5"
            model = tf.keras.models.load_model(model_name)
            src_data=src+"DATA/"
            src_result=src+"PREDICTIONS/"+data_set+"/Model4_"+str(fold)+"_Ch_"+"_".join([str(ch) for ch in sensors])+"/"
            
            if 5 in sensors:
                sensors.extend([7, 8, 9, 10])
            
            ch_new_indx = [10, 10, 0, 1, 2, 3, 10, 4, 5, 6, 7]   
            sensors_id = [ch_new_indx[int(i)] for i in sensors]
        
            try:
                os.mkdir(src_result)    
            except:
                pass        
            
            np.random.shuffle(test_f)
            
            for current_file in test_f:
                #print(current_file)
                tic()
                n_f_o=src_result+current_file
                n_f_d=str(np.char.replace(n_f_o, 'hdf5','delete'))
                if  os.path.exists( n_f_o ) or os.path.exists( n_f_d ):
                    continue
                fd = open(n_f_d, "w")
                fd.close()  
                model_4_sleep(model, src_data, src_result, current_file, sensors_id)
                os.remove(n_f_d)
                toc()

            print("DONE")
        print(" ================================= OVER =====================================")
    
"""    
    
src= "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/PREDICTIONS/TEST/"    
model = "Model4_1_Ch_2_3/"
file  = "mesa-sleep-3173.hdf5"  

   
with h5py.File(src+model+file, 'r') as hf:
    y_real = np.array(hf["y_real"][:] ) 
    y_pred = np.array(hf["y_pred"][:] )
    y_sleep = np.array(hf["y_sleep"][:] )
    

print(len(y_real), y_real, y_pred, y_sleep)

"""

    
    
