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

import matplotlib
import matplotlib.pyplot as plt
import json
import matplotlib.pylab as pylab

from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning)


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
  

      

def calculate_ahi(y_pred, y_real, threshold, ws, st, factor):
    num_hours = len(y_pred)*st / 3600 
    dp_permin = 60/ws*(ws/st) #N of datapoints per min
    y_pred_binary =  y_pred > threshold
    tn, fp, fn, tp = conf_matrix(y_real, y_pred_binary)
    #print(y_pred_binary)
    #ahi = sum(y_pred_binary)/num_hours/dp_permin
    ahi = sum(y_pred_binary)/len(y_pred_binary) * factor
    ahi_class = calculate_ahi_class(ahi)   
    return tn, fp, fn, tp, ahi, ahi_class



def conf_matrix(y_real, y_pred_binary):
    try:
        tn, fp, fn, tp = metrics.confusion_matrix(y_real, y_pred_binary).ravel()
    except:
        #print(y_real)
        #print(y_pred_binary)
        #print(metrics.confusion_matrix(y_real, y_pred_binary))
        tn = sum(y_real==0)
        fp = 0
        fn = 0
        tp = sum(y_real==1)
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

    params = {'legend.fontsize': 'large',
            #'figure.figsize': (15, 5),
            'axes.labelsize': 'large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'medium',
            'ytick.labelsize':'medium'}
    pylab.rcParams.update(params)
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
    
    src = sys.argv[1]  # src = "/work/projects/heart_project/OSA_MW/all_10_ws_10648_files_ahi_sleep_newSF/"
    ws = int(sys.argv[2])
    st = int(sys.argv[3])
    data_set = sys.argv[4]
    fold = str(sys.argv[5])
    model_type = str(sys.argv[6])
    channels = str(sys.argv[7])
    
    
    ch_names = ["Abdominal", "Thoracic", "Flow", "SpO2"]
    channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])

    #thr_val = float(sys.argv[8])
    if ws == 80:
        factor = 52
    elif ws == 30:
        factor = 77
    elif ws == 10:
        factor = 114
        
        
    
    output_dir = src+"EVAL_MODELS/MOVING_AVERAGE/"+channels+"/"
    
    try:
        os.mkdir(output_dir)
        os.mkdir(output_dir+"curves/")
        os.mkdir(output_dir+"figures/")
        os.mkdir(output_dir+"ahi_scores/")
        os.mkdir(output_dir+"best_ahis/")
    except:
        pass 
    
    
    dir_ahi = "/work/projects/heart_project/OSA_MW/LABELS_MAT_AHI/"
    model = src+"PREDICTIONS/"+data_set+"/"+str(st)+"s/Model"+model_type+"_"+fold+"_Ch_"+channels
    
        
    print(model)
    files_pred = os.listdir(model)
    print(len(files_pred))
    
    model_name_split = (model.split("/")[-1]).split("_")
    print(model_name_split)
    model_name = model_name_split[0]
    
    #fold = model_name_split[1]
    #channels = "_".join(model_name_split[3:])
    
    results_name = model_name+"_f"+fold+"_ch_"+channels+"_"
    
    if data_set == "TEST":
        
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "r") as f:
            lines = f.readlines()
        
        threshold_best_f1_classif = float((lines[0].split(","))[1])
        threshold_crossing_pr = float((lines[3].split(","))[1])
        threshold_best_f1_ahi = float((lines[6].split(","))[1])
        factor = float(lines[9].split(",")[1][1:6])
        factor_crosspr = float(lines[12].split(",")[1][1:6])
    
    
    print("Factor: "+str(factor))
    
    
    y_reals = []
    y_preds = []
    y_sleeps =[]
    y_preds_ma = [[],[],[],[],[]]
    
    patient_one = []
    ahi_real_one = []
    ahi_real_classes_one = []
    positives_real_one = []
    total_real_one = []
    positives_real_sleep_one = []
    total_real_sleep_one = []
    
    patients = []
    ahi_reals = []
    ahi_real_classes = []
    thresholds = []
    n_pasts = []
    tns = []
    fps = []
    fns = []
    tps = []
    positives = [] 
    totals = []
    per_positives = []
    acs = []
    prs = []
    recs = []
    f1s = []
    
    ahi_scores = []
    ahi_classes = []
    
    tn_sleeps = []
    fp_sleeps = []
    fn_sleeps = []
    tp_sleeps = []
    positive_sleeps = []
    total_sleeps = []
    per_positive_sleeps = []
    ac_sleeps = []
    pr_sleeps = []
    rec_sleeps = []
    f1_sleeps = []
    
    ahi_sleep_scores = []
    ahi_sleep_classes = []
    

    current_file = files_pred[0]
    for current_file in files_pred:
        patient = current_file.split(".")[0]
        try:
            with h5py.File(model+"/"+current_file, 'r') as f:
            #with h5py.File(file, 'r') as f:
                y_real = np.array(f["y_real"][:]) 
                y_pred = np.array(f["y_pred"][:])
                y_sleep = np.array(f["y_sleep"][:])
        except:
            print("Could not read "+current_file)
            continue
        
        
        if not len(y_real) == len(y_pred) or not len(y_real) == len(y_sleep) or len(y_real)<1:
            print("Someting weird with the length of "+current_file)
            continue
        
        #GET REAL ONE AHI
        current_file_name = (current_file.split("/")[-1]).split(".")[0]
        ahi_real_f = (scipy.io.loadmat(dir_ahi+current_file_name+"-label.mat"))["ahi_c"][0][0]
        ahi_real_class = calculate_ahi_class(ahi_real_f)
            
        y_reals.extend(y_real)
        y_preds.extend(y_pred)
        y_sleeps.extend(y_sleep)
        
        y_pred_binary =  sum(y_real > 0.5)
        len_y_pred = len(y_real)
        
        # Take out the sleeping parts
        indx_s = np.nonzero(y_sleep)
        #print(indx_s[0])
        y_pred_sleep = np.array(y_pred)[indx_s]
        y_real_sleep = np.array(y_real)[indx_s]
        
        y_pred_binary_sleep  =  sum(y_real_sleep  > 0.5)
        len_y_pred_sleep  = len(y_real_sleep )
        
        patient_one.append(patient)
        ahi_real_one.append(ahi_real_f)
        ahi_real_classes_one.append(ahi_real_class)
        positives_real_one.append(y_pred_binary) 
        total_real_one.append(len_y_pred)
        positives_real_sleep_one.append(y_pred_binary_sleep)
        total_real_sleep_one.append(len_y_pred_sleep)
        
        
        
        for n_past in range(1,2):
            y_pred_ma = prob_moving_average(y_pred, n_past)
            y_preds_ma[n_past-1].extend(y_pred_ma)
            y_pred_ma_sleep = np.array(y_pred_ma)[indx_s]
            for threshold_it in range(40,90):
                threshold = threshold_it/100
                tn, fp, fn, tp, ahi, ahi_class = calculate_ahi(y_pred_ma, y_real, threshold, ws, st, factor)
                tn_sleep, fp_sleep, fn_sleep, tp_sleep, ahi_sleep, ahi_class_sleep = calculate_ahi(y_pred_ma_sleep, y_real_sleep, threshold, ws, st, factor)
                
                
                patients.append(patient)
                ahi_reals.append(ahi_real_f)
                ahi_real_classes.append(ahi_real_class)
                thresholds.append(threshold )
                n_pasts.append(n_past)
                tns.append(tn) 
                fps.append(fp)
                fns.append(fn)
                tps.append(tp)
                
                per_positives.append((fp+tp)/(tn+fp+fn+tp))
                acs.append((tn+tp)/(tn+fp+fn+tp))
                prs.append(tp/(tp+fp))
                recs.append(tp/(tp+fn))
                f1s.append(2*tp/(2*tp+fp+fn))
    
                ahi_scores.append(ahi)
                ahi_classes.append(ahi_class)
                
                tn_sleeps.append(tn_sleep) 
                fp_sleeps.append(fp_sleep)
                fn_sleeps.append(fn_sleep)
                tp_sleeps.append(tp_sleep)

                per_positive_sleeps.append((fp_sleep+tp_sleep)/(tn_sleep+fp_sleep+fn_sleep+tp_sleep))
                ac_sleeps.append((tn_sleep+tp_sleep)/(tn_sleep+fp_sleep+fn_sleep+tp_sleep))
                pr_sleeps.append(tp_sleep/(tp_sleep+fp_sleep))
                rec_sleeps.append(tp_sleep/(tp_sleep+fn_sleep))
                f1_sleeps.append(2*tp_sleep/(2*tp_sleep+fp_sleep+fn_sleep))
                
                ahi_sleep_scores.append(ahi_sleep)
                ahi_sleep_classes.append(ahi_class_sleep)
                
                
        
        
    colnames = ["patients", "ahi_reals", "ahi_real_classes", "thresholds", "n_pasts", 
                "tns", "fps", "fns", "tps", "per_positives",
                "acs",  "prs",  "recs",  "f1s",
                "ahi_scores", "ahi_classes", "tn_sleeps", "fp_sleeps", "fn_sleeps", "tp_sleeps",
                "per_positive_sleeps", "ac_sleeps", "pr_sleeps", "rec_sleeps",  "f1_sleeps",
                "ahi_sleep_scores", "ahi_sleep_classes"]
    
    ahi_results_patients = pd.concat([pd.DataFrame(patients), pd.DataFrame(ahi_reals), pd.DataFrame(ahi_real_classes),
                                pd.DataFrame(thresholds), pd.DataFrame(n_pasts), pd.DataFrame(tns), pd.DataFrame(fps), 
                                pd.DataFrame(fns), pd.DataFrame(tps), pd.DataFrame(per_positives), 
                                pd.DataFrame(acs),  pd.DataFrame(prs),  pd.DataFrame(recs),  pd.DataFrame(f1s),
                                pd.DataFrame(ahi_scores), pd.DataFrame(ahi_classes),
                                pd.DataFrame(tn_sleeps), pd.DataFrame(fp_sleeps), pd.DataFrame(fn_sleeps), pd.DataFrame(tp_sleeps),
                                pd.DataFrame(per_positive_sleeps), 
                                pd.DataFrame(ac_sleeps),  pd.DataFrame(pr_sleeps),  pd.DataFrame(rec_sleeps),  pd.DataFrame(f1_sleeps),
                                pd.DataFrame(ahi_sleep_scores), pd.DataFrame(ahi_sleep_classes)], axis=1)        
                
                
                    
    ahi_results_patients.columns = colnames

    #with open(output_dir+"ahi_scores/"+results_name+"_ahi_scores_patients_"+data_set+".csv", "w") as f:
    #    ahi_results_patients.to_csv(f)
        

    
    #y_reals.extend(y_real)
    #y_preds.extend(y_pred)
    #y_sleeps.extend(y_sleep)
    
    indx_s = np.nonzero(y_sleeps)
    #print(indx_s[0])
    y_preds_sleep = np.array(y_preds)[indx_s]
    #y_preds_ma_sleep = [np.array(y_preds)[indx_s]
    y_reals_sleep = np.array(y_reals)[indx_s]
    
    thresholds = []
    tns_all = []
    fps_all = []
    fns_all = []
    tps_all = []
    acs_all = []
    prs_all = []
    recs_all = []
    f1s_all = []
    tns_sleep_all = []
    fps_sleep_all = []
    fns_sleep_all = []
    tps_sleep_all = []
    acs_sleep_all = []
    prs_sleep_all = []
    recs_sleep_all = []
    f1s_sleep_all = []
    
    pred_positives = []
    dif_pred_real_positives = []
    
    f1_patients_ws = []
    f1_patients_ms = []
    
    for threshold_it in range(0,100):
        threshold = threshold_it/100
        thresholds.append(threshold)
        
        y_preds_binary =  np.array(y_preds) > threshold
        try:
            tn, fp, fn, tp = metrics.confusion_matrix(y_reals, y_preds_binary).ravel()
        except:
            #print(y_reals)
            #print(y_preds_binary)
            #print(metrics.confusion_matrix(y_reals, y_preds_binary))
            tn=sum(y_reals==0)
            fp=0
            fn=0
            tp=sum(y_reals==1)
    
    
        tns_all.append(tn) 
        fps_all.append(fp)
        fns_all.append(fn)
        tps_all.append(tp)

        acs_all.append((tn+tp)/(tn+fp+fn+tp))
        prs_all.append(tp/(tp+fp))
        recs_all.append(tp/(tp+fn))
        f1s_all.append(2*tp/(2*tp+fp+fn))
        
        
        y_preds_sleep_binary =  y_preds_sleep > threshold
        
        try:
            tn_sleep, fp_sleep, fn_sleep, tp_sleep = metrics.confusion_matrix(y_reals_sleep, y_preds_sleep_binary).ravel()
        except:
            #print(y_reals_sleep)
            #print(y_preds_sleep_binary )
            #print(metrics.confusion_matrix(y_reals_sleep, y_preds_sleep_binary ))
            tn_sleep=sum(y_reals_sleep==0)
            fp_sleep=0
            fn_sleep=0
            tp_sleep=sum(y_reals_sleep==1)
             
        
        tns_sleep_all.append(tn_sleep) 
        fps_sleep_all.append(fp_sleep)
        fns_sleep_all.append(fn_sleep)
        tps_sleep_all.append(tp_sleep)

        acs_sleep_all.append((tn_sleep+tp_sleep)/(tn_sleep+fp_sleep+fn_sleep+tp_sleep))
        prs_sleep_all.append(tp_sleep/(tp_sleep+fp_sleep))
        recs_sleep_all.append(tp_sleep/(tp_sleep+fn_sleep))
        f1s_sleep_all.append(2*tp_sleep/(2*tp_sleep+fp_sleep+fn_sleep))
        
        
        pred_positives.append(fp_sleep+tp_sleep)
        dif_pred_real_positives.append(abs((fp_sleep+tp_sleep)-(fn_sleep+tp_sleep)))
        
        per_selected = ahi_results_patients[ahi_results_patients.thresholds == threshold]
        per_selected = per_selected[per_selected.n_pasts == 1]
        
        f1_patients_w = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_sleep_classes, average="weighted")
        f1_patients_m = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_sleep_classes, average="micro")
        
        f1_patients_ws.append(f1_patients_w)
        f1_patients_ms.append(f1_patients_m)
        
        
        
    
    colnames = ["thresholds", "tns", "fps", "fns", "tps",
                "acs",  "prs",  "recs",  "f1s",
                "tn_sleeps", "fp_sleeps", "fn_sleeps", "tp_sleeps",
                "ac_sleeps", "pr_sleeps", "rec_sleeps",  "f1_sleeps", 
                "pred_positives", "dif_pred_real_positives", 
                "f1_patients_ws", "f1_patients_ms"]
    
    classif_results = pd.concat([pd.DataFrame(thresholds), pd.DataFrame(tns_all), pd.DataFrame(fps_all), 
                             pd.DataFrame(fns_all), pd.DataFrame(tps_all), pd.DataFrame(acs_all),  pd.DataFrame(prs_all),  
                             pd.DataFrame(recs_all),  pd.DataFrame(f1s_all),
                             pd.DataFrame(tns_sleep_all), pd.DataFrame(fps_sleep_all), pd.DataFrame(fns_sleep_all), pd.DataFrame(tps_sleep_all),
                             pd.DataFrame(acs_sleep_all),  pd.DataFrame(prs_sleep_all),  pd.DataFrame(recs_sleep_all),  pd.DataFrame(f1s_sleep_all),
                             pd.DataFrame(pred_positives), pd.DataFrame(dif_pred_real_positives),
                             pd.DataFrame(f1_patients_ws), pd.DataFrame(f1_patients_ms)], axis=1)        
                
                
                    
    classif_results.columns = colnames

    #with open(output_dir+"ahi_scores/"+results_name+"_ahi_scores_classification_"+data_set+".csv", "w") as f:
    #    classif_results.to_csv(f)
    
    
    # PLOT CLASSIFICATION FIGURE and SAVE RESULTS OF PLOTS FOR LATER 
    fig_new = False
    if fig_new:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        fpr, tpr, _ = metrics.roc_curve(y_reals_sleep, y_preds_sleep)
        auroc = metrics.roc_auc_score(y_reals_sleep, y_preds_sleep)

        axs[0].plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
        axs[0].set_title("ROC Curve")
        axs[0].set_xlabel("FPR")
        axs[0].set_ylabel("TPR")
        axs[0].legend()
        #plt.savefig(files_out+"roc_model_2_"+ch_names+".png")

        pre_arr, rec_arr, _ = metrics.precision_recall_curve(y_reals_sleep, y_preds_sleep)
        aupr = metrics.average_precision_score(y_reals_sleep, y_preds_sleep)
        
        axs[1].plot(rec_arr, pre_arr, label=f'AUPR = {aupr:.2f}')
        axs[1].set_title("PR Curve")
        axs[1].set_xlabel("Recall")
        axs[1].set_ylabel("Precision")
        axs[1].legend()
        #plt.savefig(files_out+"pr_model_2_"+ch_names+".png")
        
        
        axs[2].plot(thresholds, acs_sleep_all, label="Accuracy")
        axs[2].plot(thresholds, prs_sleep_all, label="Precision")
        axs[2].plot(thresholds, recs_sleep_all, label="Recall")
        axs[2].plot(thresholds, f1s_sleep_all, label="F1-Classification")
        #plt.plot(thresholds, f1_patients_ws, label="F1-AHI")
        axs[2].set_title("Performance metrics")
        axs[2].set_xlabel("Threshold")
        axs[2].set_label("Score")
        axs[2].legend()
        axs[2].grid()
        axs[2].set_xlim([0.4, 0.9]) 
        axs[2].set_ylim([0, 1]) 

        axs[2].set_yticks(np.arange(0, 1, 0.05))
        axs[2].set_xticks(np.arange(0.4, 0.9, 0.05))
        #fig.tight_layout()
        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.suptitle(data_set+" set. Window Size: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name, fontsize=20)
        #fig.savefig(output_dir+"figures/"+results_name+"_ahi_scores_classification_"+data_set+".png")
        

        dict_curves = {"fpr_roc": fpr.tolist(), "tpr_roc": tpr.tolist(),  "auroc": auroc, "pre_pr": pre_arr.tolist(), 
                    "rec_pr": rec_arr.tolist(), "aupr": aupr, "acc": acs_sleep_all, "prs": prs_sleep_all, 
                    "rec": recs_sleep_all, "f1": f1s_sleep_all, "f1_patients": f1_patients_ws}
        
        #print(dict_curves)   
        #np.save(results_name+"_curves.txt", dict_curves)  
        #with open(output_dir+"curves/"+results_name+"_"+data_set+"_curves.txt", "w", ) as fp:
        with open("/scratch/users/mretamales/OSA_scratch/new_pipeline_curves.txt", "w", ) as fp:
            # Load the dictionary from the file
            json.dump(dict_curves, fp)
    
    
    if True:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        fpr, tpr, _ = metrics.roc_curve(y_reals_sleep, y_preds_sleep)
        auroc = metrics.roc_auc_score(y_reals_sleep, y_preds_sleep)

        axs[0].plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
        axs[0].set_title("ROC Curve")
        axs[0].set_xlabel("FPR")
        axs[0].set_ylabel("TPR")
        axs[0].legend()
        #plt.savefig(files_out+"roc_model_2_"+ch_names+".png")

        pre_arr, rec_arr, _ = metrics.precision_recall_curve(y_reals_sleep, y_preds_sleep)
        aupr = metrics.average_precision_score(y_reals_sleep, y_preds_sleep)
        
        axs[1].plot(rec_arr, pre_arr, label=f'AUPR = {aupr:.2f}')
        axs[1].set_title("PR Curve")
        axs[1].set_xlabel("Recall")
        axs[1].set_ylabel("Precision")
        axs[1].legend()
        #plt.savefig(files_out+"pr_model_2_"+ch_names+".png")
        
        
        axs[2].plot(thresholds, acs_sleep_all, label="Accuracy")
        axs[2].plot(thresholds, prs_sleep_all, label="Precision")
        axs[2].plot(thresholds, recs_sleep_all, label="Recall")
        axs[2].plot(thresholds, f1s_sleep_all, label="F1-Classification")
        #plt.plot(thresholds, f1_patients_ws, label="F1-AHI")
        axs[2].set_title("Performance metrics")
        axs[2].set_xlabel("Threshold")
        axs[2].set_label("Score")
        axs[2].legend()
        axs[2].grid()
        axs[2].set_xlim([0.4, 0.9]) 
        axs[2].set_ylim([0, 1]) 

        axs[2].set_yticks(np.arange(0, 1, 0.05))
        axs[2].set_xticks(np.arange(0.4, 0.9, 0.05))
        #fig.tight_layout()
        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        fig.suptitle(data_set+" set. Window Size: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name, fontsize=20)
        #fig.savefig(output_dir+"figures/"+results_name+"_ahi_scores_classification_"+data_set+".png")
        

        dict_curves = {"fpr_roc": fpr.tolist(), "tpr_roc": tpr.tolist(),  "auroc": auroc, "pre_pr": pre_arr.tolist(), 
                    "rec_pr": rec_arr.tolist(), "aupr": aupr, "acc": acs_sleep_all, "prs": prs_sleep_all, 
                    "rec": recs_sleep_all, "f1": f1s_sleep_all, "f1_patients": f1_patients_ws}
        
        #print(dict_curves)   
        #np.save(results_name+"_curves.txt", dict_curves)  
        with open(output_dir+"curves/"+results_name+"_"+data_set+"_curves.txt", "w", ) as fp:
            # Load the dictionary from the file
            json.dump(dict_curves, fp)
    
    if data_set == "VAL":
        #threshold_best_min = classif_results[abs(classif_results.dif_pred_real_positives) == abs(classif_results.dif_pred_real_positives).min()].iloc[0]['thresholds']
        #threshold_best_acc = classif_results[classif_results.ac_sleeps == (classif_results.ac_sleeps).max()] # .iloc[0]['thresholds']
        
        classif_results_best_f1_classif = classif_results[classif_results.f1_sleeps == (classif_results.f1_sleeps).max()]
        threshold_best_f1_classif = classif_results_best_f1_classif.iloc[0]['thresholds']
        
        classif_results_crossing_pr = classif_results[abs(classif_results.pr_sleeps - classif_results.f1_sleeps) == abs(classif_results.pr_sleeps - classif_results.f1_sleeps).min()]
        threshold_crossing_pr =  classif_results_crossing_pr.iloc[0]['thresholds']
        
        classif_results_best_f1_ahi = classif_results[classif_results.f1_patients_ms == (classif_results.f1_patients_ms).max()]
        threshold_best_f1_ahi =  classif_results_best_f1_ahi.iloc[0]['thresholds']    
                
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "w") as f:
            f.write('F1-CL-Threshold,'+str(threshold_best_f1_classif))
            f.write('\n')
            f.write(classif_results_best_f1_classif.iloc[[0]].to_string())
            f.write('\n')
            f.write('PR-Threshold,'+str(threshold_crossing_pr))
            f.write('\n')
            f.write(classif_results_crossing_pr.iloc[[0]].to_string())
            f.write('\n')
            f.write('F1-AHI-Threshold,'+str(threshold_best_f1_ahi))
            f.write('\n')
            f.write(classif_results_best_f1_ahi.iloc[[0]].to_string())
            
            
    
    elif data_set == "TEST":
        
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "r") as f:
            lines = f.readlines()
        
        threshold_best_f1_classif = float((lines[0].split(","))[1])
        threshold_crossing_pr = float((lines[3].split(","))[1])
        threshold_best_f1_ahi  = float((lines[6].split(","))[1])
        
        classif_results_best_f1_classif = classif_results[classif_results.thresholds == threshold_best_f1_classif]
        classif_results_crossing_pr = classif_results[classif_results.thresholds == threshold_crossing_pr]
        f1_score_plot2 = classif_results_crossing_pr.iloc[0]['f1_patients_ms']
        classif_results_best_f1_ahi = classif_results[classif_results.thresholds == threshold_best_f1_ahi]
        f1_score_plot1 = classif_results_best_f1_ahi.iloc[0]['f1_patients_ms']
        
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "w") as f:
            f.write('F1-Threshold,'+str(threshold_best_f1_classif))
            f.write('\n')
            f.write(classif_results_best_f1_classif.to_string())
            f.write('\n')
            f.write('PR-Threshold,'+str(threshold_crossing_pr))
            f.write('\n')
            f.write(classif_results_crossing_pr.to_string())
            f.write('\n')
            f.write('F1-AHI-Threshold,'+str(threshold_best_f1_ahi))
            f.write('\n')
            f.write(classif_results_best_f1_ahi.to_string())
        

           
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_classif]
    patients_best = patients_best[patients_best.n_pasts==1]
    #with open(output_dir+"best_ahis/"+results_name+"_best_patient_F1scores_"+data_set+".csv", "w") as f:
    #    patients_best.to_csv(f)
        
        
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_ahi]
    patients_best = patients_best[patients_best.n_pasts==1]
    
    
    if data_set == "VAL":
        patients_ahi = (patients_best.ahi_reals).values
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(patients_predpos, patients_ahi)
        
        predictions = reg.predict(patients_predpos)
        patients_best["new_ahi_est"] = predictions
        patients_best["new_ahi_class"] = [calculate_ahi_class(pred) for pred in predictions]
        
        f1_score_plot1 = metrics.f1_score(patients_best["ahi_real_classes"], patients_best["new_ahi_class"], average="micro")
        
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_ahi,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_bestf_f1_ahi,'+str(reg.intercept_))
            
            
    #with open(output_dir+"best_ahis/"+results_name+"_threshold_best_f1_ahi_"+data_set+".csv", "w") as f:
    #    patients_best.to_csv(f)
        
        
    
    
    """
    #plt.figure()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(thresholds, acs_sleep_all, label="Accuracy")
    axs[0].plot(thresholds, prs_sleep_all, label="Precision")
    axs[0].plot(thresholds, recs_sleep_all, label="Recall")
    axs[0].plot(thresholds, f1s_sleep_all, label="F1-Classification")
    axs[0].plot(thresholds, f1_patients_ws, label="F1-AHI")
    #axs[0].plot([threshold_best_f1_classif,threshold_best_f1_classif], [0,1], label = "threshold_best_f1_classif")
    #axs[0].plot([threshold_crossing_pr,threshold_crossing_pr], [0,1], label = "threshold_crossing_pr")
    axs[0].plot([threshold_best_f1_ahi,threshold_best_f1_ahi], [0,1], label = "threshold_best_f1_ahi")
    axs[0].set_title("WS: "+str(ws)+" "+model_name+" fold: "+fold+" ch: "+channels+" dataset: "+data_set)
    axs[0].set_xlabel("Threshold")
    axs[0].set_ylabel("Score")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xlim([0.4, 0.9]) 
    axs[0].set_ylim([0, 1]) 

    axs[0].set_yticks(np.arange(0, 1, 0.05))
    axs[0].set_xticks(np.arange(0.4, 0.9, 0.05))
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    colors = patients_best.ahi_real_classes
    #print(colors)
    colors_scale = ["r", "b", "g", "m"]
    colors_points = [colors_scale[int(c)] for c in colors]
    #cmap = plt.cm.rainbow
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
    
    axs[0].scatter(0,0, c="r", label = "Healthy")
    axs[0].scatter(0,0, c="b", label = "Mild")
    axs[0].scatter(0,0, c="g", label = "Moderate")
    axs[0].scatter(0,0, c="m", label = "Severe")
    axs[0].scatter(patients_best.ahi_reals, patients_best.ahi_sleep_scores, c=colors_points)
    axs[0].plot([0,110],[5,5], 'k--')
    axs[0].plot([0,110],[15,15], 'k--')
    axs[0].plot([0,110],[30,30], 'k--')
    axs[0].plot([0,110],[0,110], 'k-')
    
    axs[0].text(80, 7, "Pred. AHI = 5")
    axs[0].text(80, 17, "Pred. AHI = 15")
    axs[0].text(80, 32, "Pred. AHI = 30")
    axs[0].set_xlim([0, 110]) 
    axs[0].set_ylim([0, 110]) 
    axs[0].legend() 
    axs[0].set_xlabel("Real AHI")
    axs[0].set_ylabel("Predicted AHI")

    cm = metrics.confusion_matrix(patients_best.ahi_real_classes, patients_best.ahi_sleep_classes, labels=[0, 1, 2, 3])
    
    axs[1].matshow(cm, cmap=plt.cm.Blues)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[1].text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if  cm[i, j] > thresh else "black") 

    
    labels = ['Healthy', 'Mild', 'Moderate', 'Severe']
    #f1_score_plot = classif_results_best_f1_ahi.iloc[0]['f1_patients_ms']
    #print(f1_score_plot)
    axs[1].set_title(f'Confusion matrix, F1 = {f1_score_plot1:.2f}')
    axs[1].set_xticklabels([''] + labels)
    axs[1].set_yticklabels([''] + labels)
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.suptitle(data_set+" set. Window Size: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name, fontsize=20)
    #fig.savefig(output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestF1AHI_"+data_set+".png")
    
    
    
    
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_crossing_pr]
    patients_best = patients_best[patients_best.n_pasts==1]
    
    if data_set == "VAL":
        patients_ahi = (patients_best.ahi_reals).values
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(patients_predpos, patients_ahi)
        
        #print(reg.coef_)
        #print(reg.intercept_)
        
        predictions = reg.predict(patients_predpos)
        patients_best["new_ahi_est"] = predictions
        patients_best["new_ahi_class"] = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot2 = metrics.f1_score(patients_best["ahi_real_classes"], patients_best["new_ahi_class"], average="micro")
        
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_best_th_crossing_pr,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_th_crossing_pr,'+str(reg.intercept_))
    
    
    #with open(output_dir+"best_ahis/"+results_name+"_best_patient_PRscores_"+data_set+".csv", "w") as f:
    #    patients_best.to_csv(f)
        
        
    #plt.figure()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(thresholds, acs_sleep_all, label="Accuracy")
    axs[0].plot(thresholds, prs_sleep_all, label="Precision")
    axs[0].plot(thresholds, recs_sleep_all, label="Recall")
    axs[0].plot(thresholds, f1s_sleep_all, label="F1-Classification")
    axs[0].plot(thresholds, f1_patients_ws, label="F1-AHI")
    #axs[0].plot([threshold_best_f1_classif,threshold_best_f1_classif], [0,1], label = "threshold_best_f1_classif")
    axs[0].plot([threshold_crossing_pr,threshold_crossing_pr], [0,1], label = "threshold_crossing_pr")
    #axs[0].plot([threshold_best_f1_ahi,threshold_best_f1_ahi], [0,1], label = "threshold_best_f1_ahi")
    axs[0].set_title("WS: "+str(ws)+" "+model_name+" fold: "+fold+" ch: "+channels+" dataset: "+data_set)
    axs[0].set_xlabel("Threshold")
    axs[0].set_ylabel("Score")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xlim([0.4, 0.9]) 
    axs[0].set_ylim([0, 1]) 

    axs[0].set_yticks(np.arange(0, 1, 0.05))
    axs[0].set_xticks(np.arange(0.4, 0.9, 0.05))
    
    colors = patients_best.ahi_real_classes
    #print(colors)
    colors_scale = ["r", "b", "g", "m"]
    colors_points = [colors_scale[int(c)] for c in colors]
    #cmap = plt.cm.rainbow
    #norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
    
    axs[1].scatter(0,0, c="r", label = "Healthy")
    axs[1].scatter(0,0, c="b", label = "Mild")
    axs[1].scatter(0,0, c="g", label = "Moderate")
    axs[1].scatter(0,0, c="m", label = "Severe")
    axs[1].scatter(patients_best.ahi_reals, patients_best.ahi_sleep_scores, c=colors_points)
    axs[1].plot([0,110],[5,5], 'k--')
    axs[1].plot([0,110],[15,15], 'k--')
    axs[1].plot([0,110],[30,30], 'k--')
    axs[1].plot([0,110],[0,110], 'k-')
    
    axs[1].text(80, 7, "Pred. AHI = 5")
    axs[1].text(80, 17, "Pred. AHI = 15")
    axs[1].text(80, 32, "Pred. AHI = 30")
    axs[1].set_xlim([0, 110]) 
    axs[1].set_ylim([0, 110]) 
    axs[1].legend() 
    axs[1].set_xlabel("Real AHI")
    axs[1].set_ylabel("Predicted AHI")

    cm = metrics.confusion_matrix(patients_best.ahi_real_classes, patients_best.ahi_sleep_classes, labels=[0, 1, 2, 3])
    
    axs[2].matshow(cm, cmap=plt.cm.Blues)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[2].text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if  cm[i, j] > thresh else "black") 

    
    labels = ['Healthy', 'Mild', 'Moderate', 'Severe']
    f1_score_plot = classif_results_crossing_pr.iloc[0]['f1_patients_ms']
    #print(f1_score_plot
    #)
    axs[2].set_title(f'Confusion matrix, F1 = {f1_score_plot2:.2f}')
    axs[2].set_xticklabels([''] + labels)
    axs[2].set_yticklabels([''] + labels)
    axs[2].set_xlabel('Predicted')
    axs[2].set_ylabel('True')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.suptitle(data_set+" set. Window Size: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name, fontsize=16)
    #fig.savefig(output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestPCcross_"+data_set+".png")
   

        
        
    
    
    
    
         
        
        