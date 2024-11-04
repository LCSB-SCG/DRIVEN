"""
DRIVEN is © 2024, University of Luxembourg

DRIVEN is published and distributed under the Academic Software License v1.0 (ASL). 

DRIVEN is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the ASL for more details. 

You should have received a copy of the ASL along with this program; if not, write to LCSB-innovation@uni.lu.  It is also published at https://turbogap.fi/wiki/index.php/Academic_Software_Licence.

You may contact the original licensor at LCSB-innovation@uni.lu.
"""

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
        
     
def model_1_ahi(src_data, src_features, src_result, current_file, sens): 
    with h5py.File(src_data+current_file, 'r') as hf:
        y_real = np.array(hf["y"][:,0] ) 
        y_sleep = np.array(hf["sleep_label"][:,0] )
    
    #feats_id = 1 #  GET FEATURES ID in order
    with h5py.File(src_features+current_file, 'r') as hf:
        y_real1 = np.array(hf["y"][:,1] ) 
        x_all = np.array(hf["x"][:,:])
        #print(x_all.shape) #[inx_feat, :])
    
    if not np.array_equal(y_real1, y_real): #y_real1 != y_real:
        print("NOT THE SAME")
        print(current_file)
        return
    
    for i in range(len(sens)):
        y_pred = x_all[:,i*2:i*2+2]
        y_pred = y_pred[:,1]
        
        src_result_model_file = src_result+str(sens[i])+"/"+current_file
        f1 = h5py.File(src_result_model_file, mode='w')
        f1.create_dataset("y_pred", data=y_pred)
        f1.create_dataset("y_real", data=y_real)
        f1.create_dataset("y_sleep", data=y_sleep) 
        f1.close()
    return


def model_1_sleep(src_data, src_features, src_result, current_file, sensors): 
    with h5py.File(src_data+current_file, 'r') as hf:
        y_real = np.array(hf["y"][:,0] ) 
        y_sleep = np.array(hf["sleep_label"][:,0] )
    
    #feats_id = 1 #  GET FEATURES ID in order
    with h5py.File(src_features+current_file, 'r') as hf:
        y_sleep1 = np.array(hf["sleep_label"][:,1] ) 
        x_all = np.array(hf["x"][:,:])
    
    if not np.array_equal(y_sleep1, y_sleep): #y_real1 != y_real:
        print("NOT THE SAME")
        print(current_file)
        return
    
    for i in range(len(sensors)):
        sleep_pred = x_all[:,i*2:i*2+2]
        sleep_pred = sleep_pred[:,1]
        
        src_result_model_file = src_result+str(sensors[i])+"_sleep/"+current_file
        f1 = h5py.File(src_result_model_file, mode='w')
        f1.create_dataset("sleep_pred", data=sleep_pred)
        f1.create_dataset("y_sleep", data=y_sleep) 
        f1.close()
    return


                    
def model_2_ahi(model_lgb, src_data, src_features, src_result, current_file, inx_feat):
    with h5py.File(src_data+current_file, 'r') as hf:
        y_real = np.array(hf["y"][:,0] ) 
        y_sleep = np.array(hf["sleep_label"][:,0] )
    
    #feats_id = 1 #  GET FEATURES ID in order
    with h5py.File(src_features+current_file, 'r') as hf:
        y_real1 = np.array(hf["y"][:,1] ) 
        x_all = np.array(hf["x"][:,inx_feat])
    
    if not np.array_equal(y_real1, y_real): #y_real1 != y_real:
        print("NOT THE SAME")
        print(current_file)
        return
    y_pred = model_lgb.predict_proba(x_all)
    y_pred =y_pred[:,1]
    
    f1 = h5py.File(src_result+current_file, mode='w') 
    f1.create_dataset("y_pred", data=y_pred)
    f1.create_dataset("y_real", data=y_real)
    f1.create_dataset("y_sleep", data=y_sleep) 
    f1.close()
    return


def model_2_sleep(model_lgb, src_data, src_features, src_result, current_file, inx_feat):
    with h5py.File(src_data+current_file, 'r') as hf:
        y_real = np.array(hf["y"][:,0] ) 
        y_sleep = np.array(hf["sleep_label"][:,0] )
    
    #feats_id = 1 #  GET FEATURES ID in order
    with h5py.File(src_features+current_file, 'r') as hf:
        y_sleep1 = np.array(hf["sleep_label"][:,1] ) 
        x_all = np.array(hf["x"][:,inx_feat])
    
    if not np.array_equal(y_sleep1, y_sleep): #y_real1 != y_real:
        print("NOT THE SAME")
        print(current_file)
        return
    
    sleep_pred = model_lgb.predict_proba(x_all)
    sleep_pred = sleep_pred[:,1]
    
    f1 = h5py.File(src_result+current_file, mode='w') 
    f1.create_dataset("sleep_pred", data=sleep_pred)
    f1.create_dataset("y_sleep", data=y_sleep) 
    f1.close()
    return


if __name__ == '__main__':
    ## INPUTS

    fold=int(sys.argv[1])   
    src=sys.argv[2] # src = "/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/"
    data_set=sys.argv[3]
    model_type_eval = int(sys.argv[4]) 
    stride = str(sys.argv[5]) 
    predict =  sys.argv[6] #ahi or sleep
    
    test_files = src+"SPLIT/"+data_set+"_split"+str(fold)+".txt"
    with open(test_files, 'r') as f:
        test_f = f.readlines()

    test_f = np.array([file[:-1] for file in test_f])
    print(len(test_f))
        
    src_models = src+"WEIGHTS_2out_"+predict+"_"+str(fold)+"_all/"
    
    ###################################################################################
    # MODEL 1
    if model_type_eval == 1:
        
        sensors=[2, 3, 4, 5, 7, 8, 9, 10]
        
        src_data = src+"DATA_"+stride+"s/"
        src_features = src+"PREDICTIONS/FEATURES/"+data_set+"/"+stride+"s/FEAT_DATA_"+str(fold)+"_"+predict+"_2Out/"
        src_result= src+"PREDICTIONS/"+data_set+"/"+stride+"s/Model1_"+str(fold)+"_Ch_"
        
        
        for i in range(len(sensors)):
            try:
                src_result_m = src_result+str(sensors[i])
                os.mkdir(src_result_m)    
            except:
                pass 
        
        
        np.random.shuffle(test_f)
        for current_file in test_f:
            #print(current_file)
            n_f_o=src_result+current_file
            n_f_d=str(np.char.replace(n_f_o, 'hdf5','delete'))
            if os.path.exists( n_f_o ) or os.path.exists( n_f_d ):
                    continue
            #tic()
            try:
                if predict == "ahi":
                    model_1_ahi(src_data, src_features, src_result, current_file, sensors)
                elif predict == "sleep":
                    model_1_sleep(src_data, src_features, src_result, current_file, sensors)
                else: 
                    print("Choose ahi or sleep")
            except Exception as e:
                print("ERROR: " + current_file + " Message: " + str(e))  
            #toc()

            
    
    ###################################################################################
    # MODEL 2 and 3
    if model_type_eval == 2:
        
        dnn_features = 1280
        sens=[[2, 3, 4, 5]] #ABDO - THOR - FLOW - SPO2 -  - SPO2+delays...
        
        comb_3 = list(map(list,list(combinations(sens[0], 3) )))
        comb_2 = list(map(list,list(combinations(sens[0], 2) )))
        comb_1 = list(map(list,list(combinations(sens[0], 1) )))

        sens.extend(comb_3)
        sens.extend(comb_2)
        sens.extend(comb_1)
        
        
        for current_sens in sens:
    
            inx_feat=np.array([] ,dtype=int)  
            
            for l in current_sens:
                inx_feat= np.append(inx_feat, list(range((l-2)*dnn_features, (l-1)*dnn_features)))  
                
            if 5 in current_sens: #spo2
                for spo in [4, 5, 6, 7]:
                    inx_feat= np.append(inx_feat, list(range(spo*dnn_features, (spo+1)*dnn_features)))  
              
            sen_names = "_".join(map(str,current_sens))
            model_name = src_models+"LGBD_MODEL_f"+predict+"_"+str(sen_names)+".pkl"
            model_lgb = joblib.load(model_name)

            src_data = src+"DATA_"+stride+"s/"
            src_features = src+"PREDICTIONS/FEATURES/"+data_set+"/"+stride+"s/FEAT_DATA_"+str(fold)+"_"+predict+"/"
            src_result=src+"PREDICTIONS/"+data_set+"/"+stride+"s/Model2_"+str(fold)+"_Ch_"+sen_names+"/"
            
            
            try:
                os.mkdir(src_result)    
            except:
                pass        
            
            np.random.shuffle(test_f)
            
            for current_file in test_f:
                #print(current_file)
                n_f_o=src_result+current_file
                n_f_d=str(np.char.replace(n_f_o, 'hdf5','delete'))
                if  os.path.exists( n_f_o ) or os.path.exists( n_f_d ):
                    continue
                fd = open(n_f_d, "w")
                fd.close()
                try:  
                    tic()
                    if predict == "ahi":
                        model_2_ahi(model_lgb, src_data, src_features, src_result, current_file, inx_feat)
                    elif predict == "sleep":
                        model_2_sleep(model_lgb, src_data, src_features, src_result, current_file, inx_feat)
                    else: 
                        print("Choose ahi or sleep")
                
                    os.remove(n_f_d)
                    toc()
                except Exception as e:
                    print("ERROR: " + current_file + " Message: " + str(e))

            print("DONE")
        print(" ================================= OVER =====================================")
    
    ###################################################################################    
    if model_type_eval == 3:
        
        dnn_features = 2
        sens=[[2, 3, 4, 5]] #ABDO - THOR - FLOW - SPO2 -  - SPO2+delays...
        
        comb_3 = list(map(list,list(combinations(sens[0], 3) )))
        comb_2 = list(map(list,list(combinations(sens[0], 2) )))
        comb_1 = list(map(list,list(combinations(sens[0], 1) )))

        sens.extend(comb_3)
        sens.extend(comb_2)
        sens.extend(comb_1)
        
        #sens=[[2, 5]]
        for current_sens in sens:
    
            inx_feat=np.array([] ,dtype=int)  
            
            for l in current_sens:
                inx_feat= np.append(inx_feat, list(range((l-2)*dnn_features, (l-1)*dnn_features)))  
                
            if 5 in current_sens: #spo2
                for spo in [4, 5, 6, 7]:
                    inx_feat= np.append(inx_feat, list(range(spo*dnn_features, (spo+1)*dnn_features)))  
              
            sen_names = "_".join(map(str,current_sens))
            model_name = src_models+"LGBD_MODEL_f"+predict+"_"+str(sen_names)+"_m.pkl"
            model_lgb = joblib.load(model_name)

            src_data = src+"DATA_"+stride+"s/"
            src_features = src+"PREDICTIONS/FEATURES/"+data_set+"/"+stride+"s/FEAT_DATA_"+str(fold)+"_"+predict+"_2Out/"
            src_result=src+"PREDICTIONS/"+data_set+"/"+stride+"s/Model3_"+str(fold)+"_Ch_"+sen_names+"/"
            
            try:
                os.mkdir(src_result)    
            except:
                pass        
            
            np.random.shuffle(test_f)
            
            for current_file in test_f:
                #print(current_file)
                n_f_o=src_result+current_file
                n_f_d=str(np.char.replace(n_f_o, 'hdf5','delete'))
                if  os.path.exists( n_f_o ) or os.path.exists( n_f_d ):
                    continue
                fd = open(n_f_d, "w")
                fd.close()  
                try:
                    tic()
                    if predict == "ahi":
                        model_2_ahi(model_lgb, src_data, src_features, src_result, current_file, inx_feat)
                    elif predict == "sleep":
                        model_2_sleep(model_lgb, src_data, src_features, src_result, current_file, inx_feat)
                    else: 
                        print("Choose ahi or sleep")
                        
                    os.remove(n_f_d)
                    toc()
                except Exception as e:
                    print("ERROR: " + current_file + " Message: " + str(e))

            print("DONE")
        print(" ================================= OVER =====================================")       
    
    
   
