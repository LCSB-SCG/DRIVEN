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

import json
 
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
        


def calculate_ahi_factor(y_pred, thresholds_try, factor):
    ahi_l = []
    ahi_class_l = []
    for threshold in thresholds_try:
        y_pred_binary =  y_pred > threshold
        ahi = (sum(y_pred_binary)/len(y_pred_binary) * factor)
        ahi_l.append(ahi)
        ahi_class = calculate_ahi_class(ahi) 
        ahi_class_l.append(ahi_class)  
    return ahi_l, ahi_class_l



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
    st = int(ws/2)
    data_set = sys.argv[3]
    
    
    thresholds_try = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    output_dir = src+"EVAL_MODELS/ALL_MODELS/"
    
    if ws == 80:
        factor = 52
    elif ws == 30:
        factor = 77
    elif ws == 10:
        factor = 114
        
    try:
        os.mkdir(output_dir)
    except:
        pass 
    
    dir_ahi = "/work/projects/heart_project/OSA_MW/LABELS_MAT_AHI/"
    dir_models = src+"PREDICTIONS/"+data_set+"/"+str(st)+"s"

    
    models_resuls = [ f.path for f in os.scandir(dir_models) if f.is_dir() ]
    #print(models_resuls)
    
    model_names = []
    folds = []
    channels_save = []
    auroc_scores = []
    aupr_scores = []
    auroc_scores_sleep = []
    aupr_scores_sleep = []
    times_end = []
    ahi_sq_errs = []
    ahi_f1_scores = []
    ahi_sq_errs_sleep = []
    ahi_f1_scores_sleep = []
    
    acc_s = []
    recall_s = []
    spec_s = []
    pres_s = []
    f1_s = []
        
        
    for model in models_resuls:
        
        print(model)
        files_pred = os.listdir(model)
        print(len(files_pred))
        
        model_name_split = (model.split("/")[-1]).split("_")
        #print(model_name_split)
        model_name = model_name_split[0]
        if not model_name.startswith("Model") or model.endswith("sleep"):
            continue
        
        
        fold = model_name_split[1]
        model_names.append(model_name)
        folds.append(fold)
        channels = "_".join(model_name_split[3:])
        channels_save.append(channels)
        
        results_name = model_name+"_f"+fold+"_ch_"+channels+"_"
        
        try:
            factor_file = src+"EVAL_MODELS/MOVING_AVERAGE/"+channels+"/best_ahis/"+results_name+"_thresholds_VAL.txt"
            print(factor_file)
            with open(factor_file, "r") as f:
                lines = f.readlines()

            factor = float(lines[6].split(",")[1][1:6])
            print(factor)
        
        except: 
            print("No old factor")
            if ws == 80:
                factor = 52
            elif ws == 30:
                factor = 77
            elif ws == 10:
                factor = 114
            print(factor)
            pass
        
        
        
        y_reals = []
        y_preds = []
        y_sleeps =[]
        ahi_scores = []
        ahi_classes = []
        ahi_real = []
        ahi_real_classes = []
        ahi_sleep_scores = []
        ahi_sleep_classes = []
        
        for current_file in files_pred:
            try:
                with h5py.File(model+"/"+current_file, 'r') as f:
                    y_real = np.array(f["y_real"][:]) 
                    y_pred = np.array(f["y_pred"][:])
                    y_sleep = np.array(f["y_sleep"][:])
            except:
                print("CANT READ "+current_file)
                continue
            
            if not len(y_real) == len(y_pred) or not len(y_real) == len(y_sleep):
                print("WEIRD lengths "+current_file)
                continue
                
            y_reals.extend(y_real)
            y_preds.extend(y_pred)
            y_sleeps.extend(y_sleep)
            ahi, ahi_class = calculate_ahi_factor(y_pred, thresholds_try, factor)
            ahi_scores.append(ahi)
            ahi_classes.append(ahi_class)
            # Take out the sleeping parts
            indx_s = np.nonzero(y_sleep)
            #print(indx_s[0])
            y_pred_sleep = np.array(y_pred)[indx_s]
            ahi_sleep, ahi_class_sleep = calculate_ahi_factor(y_pred_sleep, thresholds_try, factor)
            ahi_sleep_scores.append(ahi_sleep)
            ahi_sleep_classes.append(ahi_class_sleep)
            #GET file
            current_file_name = (current_file.split("/")[-1]).split(".")[0]
            ahi_real_f = (scipy.io.loadmat(dir_ahi+current_file_name+"-label.mat"))["ahi_c"][0][0]
            #ahi_real_f = mat["ahi_c"][0][0]
            ahi_real_class = calculate_ahi_class(ahi_real_f)
            ahi_real_classes.append(ahi_real_class)
            ahi_real.append(ahi_real_f)
            #print(ahi)
        
        
        colnames = ["AHI_real"] +  ["AHI_T_"+str(i) for i in thresholds_try] +  ["AHI_T_sleep_"+str(i) for i in thresholds_try] + ["AHI_CLASS_R"] + ["AHI_Class_T_"+str(i) for i in thresholds_try] + ["AHI_Class_T_sleep_"+str(i) for i in thresholds_try]
        ahi_dataframe = pd.concat([pd.DataFrame(ahi_real), pd.DataFrame(ahi_scores), pd.DataFrame(ahi_sleep_scores), pd.DataFrame(ahi_real_classes), pd.DataFrame(ahi_classes), pd.DataFrame(ahi_sleep_classes)], axis=1)
        ahi_dataframe.columns = colnames
        with open(results_name+"_ahi_scores"+data_set+".csv", "w") as f:
            ahi_dataframe.to_csv(f)
         
        
        # Take out the sleeping parts
        indx_s = np.nonzero(y_sleeps)
        #print(indx_s[0])
        y_preds_sleep = np.array(y_preds)[indx_s]
        y_reals_sleep = np.array(y_reals)[indx_s]
            
            
        ahi_real = np.array(ahi_real)
        f1s = []
        sq_errs = []
        f1s_sleep = []
        sq_errs_sleep = []
        
        acc_pred = []
        recall_pred = []
        spec_pred = []
        pres_pred = []
        f1_pred = []
            
        y_preds = np.array(y_preds) 
        for i in range(len(thresholds_try)):
            thr = thresholds_try[i]
            #print(ahi_scores)
            ahi_sc_thr = np.array([ahi_scores[patient][i] for patient in range(len(ahi_scores))])
            #print(ahi_sc_thr)
            sq_err = np.sqrt(sum((ahi_real - ahi_sc_thr)**2) / len(ahi_scores))
            sq_errs.append(sq_err)
            ahi_class_thr = np.array([ahi_classes[patient][i] for patient in range(len(ahi_scores))])
            f1 = metrics.f1_score(ahi_real_classes, ahi_class_thr, average='weighted')
            f1s.append(f1)
            ahi_sc_sleep_thr = np.array([ahi_sleep_scores[patient][i] for patient in range(len(ahi_sleep_scores))])
            sq_err_sleep = np.sqrt(sum((ahi_real - ahi_sc_sleep_thr)**2) / len(ahi_sleep_scores))
            sq_errs_sleep.append(sq_err_sleep)
            ahi_class_thr_sleep = np.array([ahi_sleep_classes[patient][i] for patient in range(len(ahi_sleep_scores))])
            f1 = metrics.f1_score(ahi_real_classes, ahi_class_thr_sleep, average='weighted')
            f1s_sleep.append(f1)
            
            acc_pred.append(metrics.accuracy_score(y_reals, y_preds>thr))
            recall_pred.append(metrics.recall_score(y_reals, y_preds>thr))
            spec_pred.append(metrics.recall_score(y_reals, y_preds>thr, pos_label=0))
            pres_pred.append(metrics.precision_score(y_reals, y_preds>thr))
            f1_pred.append(metrics.f1_score(y_reals, y_preds>thr))
        
            
        print(f1s_sleep)
        ahi_sq_errs.append(sq_errs)
        ahi_f1_scores.append(f1s)
        ahi_sq_errs_sleep.append(sq_errs_sleep)
        ahi_f1_scores_sleep.append(f1s_sleep)
        auroc = metrics.roc_auc_score(y_reals, y_preds)
        aupr = metrics.average_precision_score(y_reals, y_preds)
        auroc_scores.append(auroc)
        aupr_scores.append(aupr)
        
        auroc_s = metrics.roc_auc_score(y_reals_sleep, y_preds_sleep)
        aupr_s = metrics.average_precision_score(y_reals_sleep, y_preds_sleep)
        auroc_scores_sleep.append(auroc_s)
        aupr_scores_sleep.append(aupr_s)
        
        print(auroc, aupr)
        print(auroc_s, aupr_s)
        
        acc_s.append(acc_pred)
        recall_s.append(recall_pred)
        spec_s.append(spec_pred)
        pres_s.append(pres_pred)
        f1_s.append(f1_pred)
        
        
    
    
    print(auroc_scores)
    print(aupr_scores)
    bin_result = pd.DataFrame.from_dict({"Name": model_names, 
                                         "Fold": folds,
                                         "Channel": channels_save,
                                         "AUROC": auroc_scores,
                                         "AUPR": aupr_scores,
                                         "AUROC_S": auroc_scores_sleep,
                                         "AUPR_S": aupr_scores_sleep})
    colnames = ["acc_T_"+str(i) for i in thresholds_try] + ["rec_T_"+str(i) for i in thresholds_try] + ["spec_T_"+str(i) for i in thresholds_try] + ["pres_T_"+str(i) for i in thresholds_try] + ["f1_b_T_"+str(i) for i in thresholds_try]
    bin_scores = pd.concat([pd.DataFrame(acc_s), pd.DataFrame(recall_s), pd.DataFrame(spec_s), pd.DataFrame(pres_s), pd.DataFrame(f1_s)], axis=1)
    bin_scores.columns = colnames
    colnames = ["sq_err_T_"+str(i) for i in thresholds_try] + ["f1_T_"+str(i) for i in thresholds_try] + ["sq_err_sleep_T_"+str(i) for i in thresholds_try] + ["f1_sleep_T_"+str(i) for i in thresholds_try]
    ahi_results = pd.concat([pd.DataFrame(ahi_sq_errs), pd.DataFrame(ahi_f1_scores), pd.DataFrame(ahi_sq_errs_sleep), pd.DataFrame(ahi_f1_scores_sleep)], axis=1)
    ahi_results.columns = colnames
    results = pd.concat([ bin_result, bin_scores, ahi_results ], axis = 1)
    
    with open(output_dir+data_set+"_scores.csv", "w") as f:
        results.to_csv(f)    

    