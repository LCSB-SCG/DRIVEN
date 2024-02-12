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




if __name__ == '__main__':

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
    
    src = sys.argv[1]  # 
    ws = int(sys.argv[2])
    st = int(ws/2)
    data_set = sys.argv[3]
    
    output_dir = src+"EVAL_MODELS/SLEEP/"
    
    try:
        os.mkdir(output_dir)
    except:
        pass 
    
    curves_dir = output_dir+"curves/"
    
    try:
        os.mkdir(curves_dir)
    except:
        pass 
    
    dir_models = src+"PREDICTIONS/"+data_set+"/"+str(st)+"s/sleep/"

    
    models_results = [ f.path for f in os.scandir(dir_models) if f.is_dir() ]

    
    model_names = []
    folds = []
    channels_save = []
    auroc_scores = []
    aupr_scores = []

        
    model = models_results[0]    
    for model in models_results:
        
        
        print(model)
        files_pred = os.listdir(model)
        #print(len(files_pred))
        model_name_short = (model.split("/")[-1])
        model_name_split = model_name_short.split("_")
        #print(model_name_split)
        model_name = model_name_split[0]
        if not model_name.startswith("Model"):
            continue
        
        fold = model_name_split[1]
        model_names.append(model_name)
        folds.append(fold)
        channels = "_".join(model_name_split[3:])
        channels_save.append(channels)
        
        #results_name = output_dir+model_name+"_"+fold+"_"+channels

        sleep_preds = []
        y_sleeps =[]
        
        current_file = files_pred[0]
        for current_file in files_pred:
            #print(current_file)
            try:
                with h5py.File(model+"/"+current_file, 'r') as f:
                    sleep_pred = np.array(f["sleep_pred"][:]) 
                    y_sleep = np.array(f["y_sleep"][:])
            except:
                print(current_file)
                continue
            
            if not len(sleep_pred ) == len(y_sleep):
                print(current_file)
                continue
                
            sleep_preds.extend(sleep_pred)
            y_sleeps.extend(y_sleep)
            
        auroc = metrics.roc_auc_score(y_sleeps, sleep_preds)
        aupr = metrics.average_precision_score(y_sleeps, sleep_preds)
        auroc_scores.append(auroc)
        aupr_scores.append(aupr)

        print(auroc, aupr)
        
        fpr, tpr, _ = metrics.roc_curve(y_sleeps, sleep_preds)
        pre_arr, rec_arr, _ = metrics.precision_recall_curve(y_sleeps, sleep_preds)
        
        dict_curves = {"Model": model_name,"Fold": fold,"Channel": channels,
                       "fpr_roc": fpr.tolist(), "tpr_roc": tpr.tolist(),  "auroc": auroc, 
                       "pre_pr": pre_arr.tolist(), "rec_pr": rec_arr.tolist(), "aupr": aupr
                    }
        
        
        with open(curves_dir+model_name_short+"_"+data_set+"_metrics.txt", "w", ) as fp:
            # Load the dictionary from the file
            json.dump(dict_curves, fp)

        
        
    bin_results = pd.DataFrame.from_dict({"Name": model_names, 
                                         "Fold": folds,
                                         "Channel": channels_save,
                                         "AUROC": auroc_scores,
                                         "AUPR": aupr_scores})
    
    
    with open(output_dir+data_set+"_scores.csv", "w") as f:
        bin_results.to_csv(f)    

    