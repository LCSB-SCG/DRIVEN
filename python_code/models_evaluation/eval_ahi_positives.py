import numpy as np
import h5py
import os 

import math
from sklearn import metrics
import sys
import time
import pandas as pd
import scipy.io
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        time_end = (time.time() - startTime_for_tictoc)
    
    else:
        print("Toc: start time not set")
        time_end = 0
    return time_end
        
def calculate_ahi(y_pred, ws, st):
    num_hours = len(y_pred)*st / 3600 
    factor = 60/ws*2 #N of datapoints per min
    y_pred_binary =  y_pred > 0.5
    ahi = sum(y_pred_binary)/num_hours/factor
    ahi_class = calculate_ahi_class(ahi)   
    return ahi, ahi_class

"""
def calculate_ahi(y_pred, y_real, threshold, ws, st):
    num_hours = len(y_pred)*st / 3600 
    factor = 60/ws*2 #N of datapoints per min
    y_pred_binary =  y_pred > threshold
    tn, fp, fn, tp = conf_matrix(y_real, y_pred_binary)
    #print(y_pred_binary)
    ahi = sum(y_pred_binary)/num_hours/factor
    ahi_class = calculate_ahi_class(ahi)   
    return tn, fp, fn, tp, ahi, ahi_class
"""

def conf_matrix(y_real, y_pred_binary):
    try:
        tn, fp, fn, tp = metrics.confusion_matrix(y_real, y_pred_binary).ravel()
    except:
        print(y_real)
        print(y_pred_binary)
        print(metrics.confusion_matrix(y_real, y_pred_binary))
        tn = 0
        fp = 0
        fn = 0
        tp = 0
    return tn, fp, fn, tp
        
        
    
    
    
def prob_moving_average(y_pred, n_past):
    y_pred_i = np.insert(y_pred, 0, 0)
    cumsum_vec = np.cumsum(y_pred_i) 
    y_pred_ma = (cumsum_vec[n_past:] - cumsum_vec[:-n_past]) /n_past
    for i in range(n_past-1):
        y_pred_ma = np.insert(y_pred_ma, i, np.average(y_pred[:i+1]))
    return y_pred_ma
    

def calculate_ahi_class(ahi):
    if ahi < 5:
        ahi_class=0
    elif ahi < 15:
        ahi_class = 1
    elif ahi < 30:
        ahi_class = 2
    else:
        ahi_class = 3
          
    return ahi_class



"""
file = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/PREDICTIONS/VAL/Model3_1_Ch_2_3_4_5/mros-visit1-aa0744.hdf5"
with h5py.File(file, 'r') as f:
    y_real = np.array(f["y_real"][:]) 
    y_pred = np.array(f["y_pred"][:])
    y_sleep = np.array(f["y_sleep"][:]) 

y_all = np.append([y_real], [y_pred], axis=0)
y_all2 = np.append(y_all, [y_sleep], axis=0)

y_pred_ma = prob_moving_average(y_pred, n_past)
y_all3 = np.append(y_all2, [y_pred_ma], axis=0)

np.savetxt("/scratch/users/mretamales/OSA_scratch/new_pipeline/values_st15.csv", y_all3, delimiter=",")

ws = 30
st=15
n_past=2
threshold=0.67
calculate_ahi_moving_average(y_pred, threshold, ws, st, n_past)


for n_past in range(1,10):
    for threshold in [0.5, 0.6, 0.7]:
        calculate_ahi_moving_average(y_pred, threshold, ws, st, n_past)

"""
if __name__ == '__main__':
    
    # python eval_predictions_ahiclass.py "/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/" 80 "VAL"

    ##  CHANNELS
    # 1 ECG
    # 2 ABDOMINAL
    # 3 CHEST
    # 4 NASSAL
    # 5 SPO2    
    # 6 RRI
    # 7 SPO2 + 10
    # 8 SPO2 + 15
    # 9 SPO2 + 20
    # 10 SPO2 + 25
    
    src = sys.argv[1]  
    ws = int(sys.argv[2])
    st = int(sys.argv[3])
    data_set = sys.argv[4]
    
    output_dir = "/work/projects/heart_project/OSA_MW/7_EVAL_MODELS/MOVING_AVERAGE/"
    
    try:
        os.mkdir(output_dir)
    except:
        pass 
    
    dir_ahi = "/work/projects/heart_project/OSA_MW/LABELS_MAT_AHI/"
    data = src+"DATA_5s"
    files_pred = os.listdir(data)


    
    results_name = output_dir+"EVAL"
    
    ahi_reals = []
    ahi_real_classes = []
    
    patients = []
    
    positives = [] 
    totals = []
    positive_sleeps = []
    total_sleeps = []
    
    current_file=files_pred[0]
    for current_file in files_pred:
        patient = current_file.split(".")[0]
        try:
            with h5py.File(data+"/"+current_file, 'r') as f:
            #with h5py.File(file, 'r') as f:
                y_real = np.array(f["y"][:]) 
                y_sleep = np.array(f["sleep_label"][:])
        except:
            print(current_file)
            continue
        
        
        if not len(y_real) == len(y_sleep) or len(y_real) < 1:
            print(current_file)
            continue
        
        #GET REAL ONE AHI
        current_file_name = (current_file.split("/")[-1]).split(".")[0]
        ahi_real_f = (scipy.io.loadmat(dir_ahi+current_file_name+"-label.mat"))["ahi_c"][0][0]
        ahi_real_class = calculate_ahi_class(ahi_real_f)
            
        
        y_real_binary =  sum(y_real > 0.5)
        len_y_real = len(y_real)
        
        # Take out the sleeping parts
        indx_s = np.nonzero(y_sleep)
        y_real_sleep = np.array(y_real)[indx_s]
        
        y_real_binary_sleep  =  sum(y_real_sleep  > 0.5)
        len_y_real_sleep  = len(y_real_sleep )
        
        patients.append(patient)
        ahi_reals.append(ahi_real_f)
        ahi_real_classes.append(ahi_real_class)
        positives.append(y_real_binary) 
        totals.append(len_y_real)
        positive_sleeps.append(y_real_binary_sleep)
        total_sleeps.append(len_y_real_sleep)
        
    colnames = ["patients", "ahi_reals", "ahi_real_classes", "positives", "total", "positives_sleep", "total_sleep"]

    ahi_results = pd.concat([pd.DataFrame(patients), pd.DataFrame(ahi_reals), pd.DataFrame(ahi_real_classes), 
                             pd.DataFrame(positives), pd.DataFrame(totals), pd.DataFrame(positive_sleeps),                     
                            pd.DataFrame(total_sleeps)], axis=1)            
    
    #print(patients)
    #print(ahi_results)
    ahi_results.columns = colnames

    with open(results_name+"_ahi_scores_"+str(ws)+"_5s.csv", "w") as f:
        ahi_results.to_csv(f)
        

        
        