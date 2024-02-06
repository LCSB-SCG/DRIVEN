import numpy as np
import h5py
import os 
import sys
import pandas as pd
import scipy.io



  

def calculate_ahi(y_predicted, thresholds_try, ws, st):
    num_hours = len(y_predicted)*st / 3600 
    factor = 60/ws*2 #N of datapoints per min
    ahi = []
    ahi_class_l = []
    for threshold in thresholds_try:
        y_pred_binary = sum( np.array(y_predicted) > threshold)
        Predicted_ph = y_pred_binary/num_hours/factor
        ahi.append(Predicted_ph)
        ahi_class = calculate_ahi_class(Predicted_ph)
        ahi_class_l.append(ahi_class)
        
    return ahi, ahi_class_l

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
np.savetxt("/scratch/users/mretamales/OSA_scratch/new_pipeline/values_st15.csv", y_all2, delimiter=",")


"""


if __name__ == '__main__':
    
    
    thresholds_try = [0.5]
    
    
    
    dir_ahi = "/work/projects/heart_project/OSA_MW/LABELS_MAT_AHI/"
    src = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/"
    data_dir = src+"DATA/"
    output_dir = src+"EVAL_AHI/"
    try:
        os.mkdir(output_dir)
    except:
        pass 
    ws = 30
    st = 15
    
        
    files_pred = os.listdir(data_dir)
    
    patient = []
    ahi_scores = []
    ahi_classes = []
    ahi_sleep_scores = []
    ahi_sleep_classes = []
    ahi_real_classes = []
    ahi_real = []
    positives = []
    total = []
    positives_s = []
    total_s = []
        
        
    for current_file in files_pred[:3000]:
        #print(current_file)
        try:
            with h5py.File(data_dir+current_file, 'r') as f:
            #with h5py.File(file, 'r') as f:
                y_real = np.array(f["y"][:]) 
                y_sleep = np.array(f["sleep_label"][:])
        except:
            print("ERROR READING")
            print(current_file)
            continue
        
        if not len(y_real) == len(y_sleep):
            print("Y NOT S")
            print(current_file)
            continue
        
        total.append(len(y_real)) 
        positives.append(sum(y_real>0.5))
        #print(total, positives)
        
        try: 
            ahi, ahi_class = calculate_ahi(y_real, thresholds_try, ws, st)
        except:
            ahi = 0
            ahi_class = 0
        
        ahi_scores.append(ahi)
        ahi_classes.append(ahi_class)
        #print(ahi_scores, ahi_classes)   
         
        try:
            # Take out the sleeping parts
            indx_s = np.nonzero(y_sleep)
            #print(indx_s[0])
            y_real_sleep = np.array(y_real)[indx_s]
            total_s.append(len(y_real_sleep)) 
            positives_s.append(sum(y_real_sleep>0.5)) 
            ahi_sleep, ahi_class_sleep = calculate_ahi(y_real_sleep, thresholds_try, ws, st)
        
        except:
            ahi_sleep = 0
            ahi_class_sleep = 0
            
        ahi_sleep_scores.append(ahi_sleep)
        ahi_sleep_classes.append(ahi_class_sleep)
            
        #GET file
        current_file_name = (current_file.split("/")[-1]).split(".")[0]
        patient.append(current_file_name)
        ahi_real_f = (scipy.io.loadmat(dir_ahi+current_file_name+"-label.mat"))["ahi_c"][0][0]
        ahi_real_class = calculate_ahi_class(ahi_real_f)
        ahi_real_classes.append(ahi_real_class)
        ahi_real.append(ahi_real_f)
        #print(ahi)
    
    
    colnames = ["patient", "ahi_real", "ahi_real_classes", "positives", "total", "ahi_scores", "ahi_classes", "positives_s", "total_s", "ahi_sleep_scores", "ahi_sleep_classes"] 

    ahi_dataframe = pd.concat([pd.DataFrame(patient), pd.DataFrame(ahi_real), pd.DataFrame(ahi_real_classes), pd.DataFrame(positives), pd.DataFrame(total), pd.DataFrame(ahi_scores), pd.DataFrame(ahi_classes), pd.DataFrame(positives_s), pd.DataFrame(total_s), pd.DataFrame(ahi_sleep_scores), pd.DataFrame(ahi_sleep_classes)], axis=1)
    

    ahi_dataframe.columns = colnames
    with open(output_dir+"P_ahi_scores.csv", "w") as f:
        ahi_dataframe.to_csv(f)
        
   

    


