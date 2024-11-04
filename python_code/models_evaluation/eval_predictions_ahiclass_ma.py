"""
DRIVEN is © 2024, University of Luxembourg

DRIVEN is published and distributed under the Academic Software License v1.0 (ASL). 

DRIVEN is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the ASL for more details. 

You should have received a copy of the ASL along with this program; if not, write to LCSB-innovation@uni.lu.  It is also published at https://turbogap.fi/wiki/index.php/Academic_Software_Licence.

You may contact the original licensor at LCSB-innovation@uni.lu.
"""

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


def plot_roc_pr_metrics(fpr, tpr, auroc, rec_arr, pre_arr, aupr, thrs, acs, prs, recs, f1s, fig_subtitle, fig_saveName):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # ROC
    axs[0].plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
    axs[0].set_title("ROC Curve")
    axs[0].set_xlabel("FPR")
    axs[0].set_ylabel("TPR")
    axs[0].set_xlim([0, 1]) 
    axs[0].set_ylim([0, 1]) 
    axs[0].legend()
    # PR
    axs[1].plot(rec_arr, pre_arr, label=f'AUPR = {aupr:.2f}')
    axs[1].set_title("PR Curve")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_xlim([0, 1]) 
    axs[1].set_ylim([0, 1]) 
    axs[1].legend()
    # Metrics
    axs[2].plot(thrs, acs, label="Accuracy")
    axs[2].plot(thrs, prs, label="Precision")
    axs[2].plot(thrs, recs, label="Recall")
    axs[2].plot(thrs, f1s, label="F1-Classification")
    axs[2].set_title("Performance metrics")
    axs[2].set_xlabel("Threshold")
    axs[2].set_label("Score")
    axs[2].legend()
    axs[2].grid()
    axs[2].set_xlim([0, 1]) 
    axs[2].set_ylim([0, 1]) 
    axs[2].set_yticks(np.arange(0, 1, 0.05))
    axs[2].set_xticks(np.arange(0, 1, 0.05))
    # SAVE
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.suptitle(fig_subtitle, fontsize=20)
    fig.savefig(fig_saveName+".png")
    



def plot_scatter_confusion(ahi_reals, ahi_reals_classes, ahi_pred, ahi_pred_classes, fig_subtitle, fig_saveName):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Scatter plot
    axs[0].scatter(0,0, c="r", label = "Healthy")
    axs[0].scatter(0,0, c="b", label = "Mild")
    axs[0].scatter(0,0, c="g", label = "Moderate")
    axs[0].scatter(0,0, c="m", label = "Severe")
    axs[0].scatter(ahi_reals, ahi_pred, c=colors_points, edgecolors="none", alpha = 0.5)
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
    # Confusion Matrix
    cm = metrics.confusion_matrix(ahi_reals_classes, ahi_pred_classes, labels=[0, 1, 2, 3])
    f1_score_plot = metrics.f1_score(ahi_reals_classes, ahi_pred_classes, average="micro")
    # Plot it
    axs[1].matshow(cm, cmap=plt.cm.Blues)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[1].text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if  cm[i, j] > thresh else "black") 
    
    labels = ['Healthy', 'Mild', 'Moderate', 'Severe']
    axs[1].set_title(f'Confusion matrix, F1 = {f1_score_plot:.2f}')
    axs[1].set_xticklabels([''] + labels)
    axs[1].set_yticklabels([''] + labels)
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')
    #
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.suptitle(fig_subtitle, fontsize=20)
    fig.savefig(fig_saveName+".png")




params = {'legend.fontsize': 'large',
            #'figure.figsize': (15, 5),
            'axes.labelsize': 'large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'medium',
            'ytick.labelsize':'medium'}
pylab.rcParams.update(params)
    
    
    
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
    
    src = sys.argv[1]  # src = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/"
    ws = int(sys.argv[2])
    st = int(sys.argv[3])
    data_set = sys.argv[4]
    fold = str(sys.argv[5])
    model_type = str(sys.argv[6])
    channels = str(sys.argv[7])
    
    threshold_sleep = float(sys.argv[8])
    
    """
    if True:
        src = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/"
        ws = 30
        st = 15
        data_set = "VAL"
        fold = "2"
        model_type = "2"
        channels = "5"
        threshold_sleep = 0.5
    """
    
    inter = True


    ch_names = ["Abdominal", "Thoracic", "Flow", "SpO2"]
    channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])

    #thr_val = float(sys.argv[8])
    if ws == 80:
        factor = 52
    elif ws == 30:
        factor = 77
    elif ws == 10:
        factor = 114


    #output_dir = src+"EVAL_MODELS/MERGE_SLEEP_Ts_"+str(int(threshold_sleep*100))+"_NEW2"
    output_dir = src+"EVAL_MODELS/MERGE_SLEEP_Ts_"+str(int(threshold_sleep*100))+"_RRI_NO3"

    try:
        os.mkdir(output_dir)
    except:
        pass 

    output_dir = output_dir+"/"+channels+"/"

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
    model_sleep = src+"/PREDICTIONS/"+data_set+"/"+str(st)+"s/sleep/Model"+model_type+"_"+fold+"_Ch_"+channels+"_sleep"
    info_apneas =  "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newLabels/DATA_15s_rri"


    print(model)
    files_pred = os.listdir(model)
    print(len(files_pred))

    model_name_split = (model.split("/")[-1]).split("_")
    print(model_name_split)
    model_name = model_name_split[0]

    results_name = model_name+"_f"+fold+"_ch_"+channels+"_"


    # If TEST, read the VALIDATION file and get classification thresholds
    # selected and regresion parameters
    if data_set == "TEST":
        # Read file
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "r") as f:
            lines = f.readlines()
        # Get thresholds
        # No corr
        threshold_best_f1_classif = float((lines[0].split(","))[1])
        threshold_crossing_pr = float((lines[3].split(","))[1])
        threshold_best_f1_ahi = float((lines[6].split(","))[1])
        # Corr ground thruth
        threshold_best_f1_classif_s = float((lines[9].split(","))[1])
        threshold_crossing_pr_s = float((lines[12].split(","))[1])
        threshold_best_f1_ahi_s = float((lines[15].split(","))[1])
        # Corr sleep pred
        threshold_best_f1_classif_sp = float((lines[18].split(","))[1])
        threshold_crossing_pr_sp = float((lines[21].split(","))[1])
        threshold_best_f1_ahi_sp = float((lines[24].split(","))[1])
        # Get regression parameters
        # No corr
        factor_best_f1_classif = float(lines[27].split(",")[1][1:-2])
        inter_best_f1_classif = float(lines[28].split(",")[1])
        factor_best_th_crossing_pr = float(lines[30].split(",")[1][1:-2])
        inter_best_th_crossing_pr = float(lines[31].split(",")[1])
        factor_best_f1_ahi = float(lines[33].split(",")[1][1:-2])
        inter_best_f1_ahi = float(lines[34].split(",")[1])
        # Corr ground thruth
        factor_best_f1_classif_s = float(lines[36].split(",")[1][1:-2])
        inter_best_f1_classif_s = float(lines[37].split(",")[1])
        factor_best_th_crossing_pr_s = float(lines[39].split(",")[1][1:-2])
        inter_best_th_crossing_pr_s = float(lines[40].split(",")[1])
        factor_best_f1_ahi_s = float(lines[42].split(",")[1][1:-2])
        inter_best_f1_ahi_s = float(lines[43].split(",")[1])
        # Corr sleep pred
        factor_best_f1_classif_sp = float(lines[45].split(",")[1][1:-2])
        inter_best_f1_classif_sp = float(lines[46].split(",")[1])
        factor_best_th_crossing_pr_sp = float(lines[48].split(",")[1][1:-2])
        inter_best_th_crossing_pr_sp = float(lines[49].split(",")[1])
        factor_best_f1_ahi_sp = float(lines[51].split(",")[1][1:-2])
        inter_best_f1_ahi_sp = float(lines[52].split(",")[1])
    
    
    print("Factor: "+str(factor))

    ###############################################################################################
    ## Loop through patients and calculate the performance metrics                               ##
    ############################################################################################### 

    # To save all classifications for later    
    y_reals = []
    y_preds = []
    y_sleeps =[]
    y_reals_types = []
    y_sleeps_preds =[]

    # Metrics:
    # 1. Patients data
    patients = []
    ahi_reals = []
    ahi_real_classes = []
    thresholds = []
    # 2.1 Classification data - no sleep correction
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
    # 2.2 AHI prediction - no sleep correction
    ahi_scores = []
    ahi_classes = []
    # 3.1 Classification data - sleep correction - ground truth
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
    # 3.2 AHI prediction - sleep correction - ground truth
    ahi_sleep_scores = []
    ahi_sleep_classes = []
    # 4.1 Classification data - sleep correction - predicted sleep
    tn_sleeps_p = []
    fp_sleeps_p = []
    fn_sleeps_p = []
    tp_sleeps_p = []
    positive_sleeps_p = []
    total_sleeps_p = []
    per_positive_sleeps_p = []
    ac_sleeps_p = []
    pr_sleeps_p = []
    rec_sleeps_p = []
    f1_sleeps_p = []
    # 3.2 AHI prediction - sleep correction - predicted sleep
    ahi_sleep_scores_p = []
    ahi_sleep_classes_p = []

    # Loop through patients
    for current_file in files_pred:
        patient = current_file.split(".")[0]
        # Read AHI_events classification prediction
        try:
            with h5py.File(model+"/"+current_file, 'r') as f:
                y_real = np.array(f["y_real"][:]) 
                y_pred = np.array(f["y_pred"][:])
                y_sleep = np.array(f["y_sleep"][:])
        except:
            print("Could not read "+current_file)
            continue
        # Read sleep predictions
        try:
            with h5py.File(model_sleep+"/"+current_file, 'r') as f:     
                y_sleep2 = np.array(f["y_sleep"][:])
                sleep_pred = np.array(f["sleep_pred"][:])
        except:
            print("Could not read sleep "+current_file)
            continue
        # READ AHI_classif_per apnea type
        try:
            with h5py.File(info_apneas+"/"+current_file, 'r') as f:
                y_apneas = np.array(f["label_y_s"][:]) 
                y_apneas = np.stack(y_apneas, axis=1)[0]
        except:
            print("Could not read "+current_file)
            continue
        
        # Check consistency of channels (just in case)
        if not len(y_real) == len(y_pred) or not len(y_real) == len(y_sleep) or len(y_real)<1 or not all(y_sleep2 == y_sleep):
            print("Someting weird with the length of "+current_file)
            continue
        
        # Check binary AHI-event is same as binarized apneas per type (2, 3, 4)
        if not all(((y_apneas==2)+(y_apneas==3)+(y_apneas==4)) == y_real):
            print("Someting weird with ahi binary and per type "+current_file)
            continue
        
        
        # Get real AHI of the patient
        current_file_name = (current_file.split("/")[-1]).split(".")[0]
        ahi_real_f = (scipy.io.loadmat(dir_ahi+current_file_name+"-label.mat"))["ahi_c"][0][0]
        ahi_real_class = calculate_ahi_class(ahi_real_f)
        # Append all data to the full classification vector (to check classification later)   
        y_reals.extend(y_real)
        y_preds.extend(y_pred)
        y_sleeps.extend(y_sleep)
        y_sleeps_preds.extend(sleep_pred)
        y_reals_types.extend(y_apneas)
        # Take out the awake parts - sleep ground truth
        indx_s = np.nonzero(y_sleep)
        y_pred_sleep = np.array(y_pred)[indx_s]
        y_real_sleep = np.array(y_real)[indx_s]
        # Take out the awake parts - sleep predicted
        indx_sp = np.nonzero( sleep_pred > threshold_sleep)
        y_pred_sleep_pred = np.array(y_pred)[indx_sp]
        y_real_sleep_pred = np.array(y_real)[indx_sp]
        # Loop through thresholds and get all metrics - Per Patient
        for threshold_it in range(0,100):
            threshold = threshold_it/100
            # Patients info
            patients.append(patient)
            ahi_reals.append(ahi_real_f)
            ahi_real_classes.append(ahi_real_class)
            thresholds.append(threshold)
            # Append all to the metrics lists
            # Without sleep correction
            tn, fp, fn, tp, ahi, ahi_class = calculate_ahi(y_pred, y_real, threshold, ws, st, factor)
            # Confusion matrix
            tns.append(tn) 
            fps.append(fp)
            fns.append(fn)
            tps.append(tp)
            # Metrics
            per_positives.append((fp+tp)/(tn+fp+fn+tp))
            acs.append((tn+tp)/(tn+fp+fn+tp))
            prs.append(tp/(tp+fp))
            recs.append(tp/(tp+fn))
            f1s.append(2*tp/(2*tp+fp+fn))
            #  AHIs
            ahi_scores.append(ahi)
            ahi_classes.append(ahi_class)
            del tn, fp, fn, tp, ahi, ahi_class
            # With sleep correction - ground thruth 
            tn_sleep, fp_sleep, fn_sleep, tp_sleep, ahi_sleep, ahi_class_sleep = calculate_ahi(y_pred_sleep, y_real_sleep, threshold, ws, st, factor)
            # Append all to the metrics lists
            # Confusion matrix
            tn_sleeps.append(tn_sleep) 
            fp_sleeps.append(fp_sleep)
            fn_sleeps.append(fn_sleep)
            tp_sleeps.append(tp_sleep)
            # Metrics
            per_positive_sleeps.append((fp_sleep+tp_sleep)/(tn_sleep+fp_sleep+fn_sleep+tp_sleep))
            ac_sleeps.append((tn_sleep+tp_sleep)/(tn_sleep+fp_sleep+fn_sleep+tp_sleep))
            pr_sleeps.append(tp_sleep/(tp_sleep+fp_sleep))
            rec_sleeps.append(tp_sleep/(tp_sleep+fn_sleep))
            f1_sleeps.append(2*tp_sleep/(2*tp_sleep+fp_sleep+fn_sleep))
            #  AHIs
            ahi_sleep_scores.append(ahi_sleep)
            ahi_sleep_classes.append(ahi_class_sleep)
            del tn_sleep, fp_sleep, fn_sleep, tp_sleep, ahi_sleep, ahi_class_sleep
            # With sleep correction - predicted slee
            # Sometimes it fails - as for some thresholds (to high) there are no predicted sleeps 
            try:
                tn_sleep_p, fp_sleep_p, fn_sleep_p, tp_sleep_p, ahi_sleep_p, ahi_class_sleep_p = calculate_ahi(y_pred_sleep_pred, y_real_sleep_pred, threshold, ws, st, factor)
                # Confusion matrix
                tn_sleeps_p.append(tn_sleep_p) 
                fp_sleeps_p.append(fp_sleep_p)
                fn_sleeps_p.append(fn_sleep_p)
                tp_sleeps_p.append(tp_sleep_p)
                # Metrics
                per_positive_sleeps_p.append((fp_sleep_p+tp_sleep_p)/(tn_sleep_p+fp_sleep_p+fn_sleep_p+tp_sleep_p))
                ac_sleeps_p.append((tn_sleep_p+tp_sleep_p)/(tn_sleep_p+fp_sleep_p+fn_sleep_p+tp_sleep_p))
                pr_sleeps_p.append(tp_sleep_p/(tp_sleep_p+fp_sleep_p))
                rec_sleeps_p.append(tp_sleep_p/(tp_sleep_p+fn_sleep_p))
                f1_sleeps_p.append(2*tp_sleep_p/(2*tp_sleep_p+fp_sleep_p+fn_sleep_p))
                #  AHIs
                ahi_sleep_scores_p.append(ahi_sleep_p)
                ahi_sleep_classes_p.append(ahi_class_sleep_p)
                del tn_sleep_p, fp_sleep_p, fn_sleep_p, tp_sleep_p, ahi_sleep_p, ahi_class_sleep_p
            except:
                # Confusion matrix
                tn_sleeps_p.append(None) 
                fp_sleeps_p.append(None)
                fn_sleeps_p.append(None)
                tp_sleeps_p.append(None)
                # Metrics
                per_positive_sleeps_p.append(0)
                ac_sleeps_p.append(None)
                pr_sleeps_p.append(None)
                rec_sleeps_p.append(None)
                f1_sleeps_p.append(None)
                #  AHIs
                ahi_sleep_scores_p.append(0)
                ahi_sleep_classes_p.append(0)
                    
    del y_pred, y_real, y_pred_sleep, y_real_sleep, y_pred_sleep_pred, y_real_sleep_pred         
    #  Save all results in dataframe
    # Column names
    colnames = ["patients", "ahi_reals", "ahi_real_classes", "thresholds", 
                "tns", "fps", "fns", "tps", "per_positives",
                "acs",  "prs",  "recs",  "f1s",
                "ahi_scores", "ahi_classes", "tn_sleeps", "fp_sleeps", "fn_sleeps", "tp_sleeps",
                "per_positive_sleeps", "ac_sleeps", "pr_sleeps", "rec_sleeps",  "f1_sleeps",
                "ahi_sleep_scores", "ahi_sleep_classes",
                "tn_sleeps_p", "fp_sleeps_p", "fn_sleeps_p", "tp_sleeps_p",
                "per_positive_sleeps_p", "ac_sleeps_p", "pr_sleeps_p", "rec_sleeps_p",  "f1_sleeps_p",
                "ahi_sleep_scores_p", "ahi_sleep_classes_p"]
    # All the lists
    ahi_results_patients = pd.concat([pd.DataFrame(patients), pd.DataFrame(ahi_reals), pd.DataFrame(ahi_real_classes),
                                pd.DataFrame(thresholds), pd.DataFrame(tns), pd.DataFrame(fps), 
                                pd.DataFrame(fns), pd.DataFrame(tps), pd.DataFrame(per_positives), 
                                pd.DataFrame(acs),  pd.DataFrame(prs),  pd.DataFrame(recs),  pd.DataFrame(f1s),
                                pd.DataFrame(ahi_scores), pd.DataFrame(ahi_classes),
                                pd.DataFrame(tn_sleeps), pd.DataFrame(fp_sleeps), pd.DataFrame(fn_sleeps), pd.DataFrame(tp_sleeps),
                                pd.DataFrame(per_positive_sleeps), 
                                pd.DataFrame(ac_sleeps),  pd.DataFrame(pr_sleeps),  pd.DataFrame(rec_sleeps),  pd.DataFrame(f1_sleeps),
                                pd.DataFrame(ahi_sleep_scores), pd.DataFrame(ahi_sleep_classes),
                                pd.DataFrame(tn_sleeps_p), pd.DataFrame(fp_sleeps_p), pd.DataFrame(fn_sleeps_p), pd.DataFrame(tp_sleeps_p),
                                pd.DataFrame(per_positive_sleeps_p), 
                                pd.DataFrame(ac_sleeps_p),  pd.DataFrame(pr_sleeps_p),  pd.DataFrame(rec_sleeps_p),  pd.DataFrame(f1_sleeps_p),
                                pd.DataFrame(ahi_sleep_scores_p), pd.DataFrame(ahi_sleep_classes_p)], axis=1)                    
    ahi_results_patients.columns = colnames
    # Save in:
    with open(output_dir+"ahi_scores/"+results_name+"_ahi_scores_patients_"+data_set+".csv", "w") as f:
        ahi_results_patients.to_csv(f)
        

    del patients, ahi_reals, ahi_real_classes, thresholds, tns, fps, fns, tps, positives, totals, per_positives, acs, prs
    del recs, f1s, ahi_scores, ahi_classes, tn_sleeps, fp_sleeps, fn_sleeps, tp_sleeps, positive_sleeps, total_sleeps
    del per_positive_sleeps, ac_sleeps, pr_sleeps, rec_sleeps, f1_sleeps, ahi_sleep_scores, ahi_sleep_classes
    del tn_sleeps_p, fp_sleeps_p, fn_sleeps_p, tp_sleeps_p, positive_sleeps_p, total_sleeps_p, per_positive_sleeps_p
    del ac_sleeps_p, pr_sleeps_p, rec_sleeps_p, f1_sleeps_p, ahi_sleep_scores_p, ahi_sleep_classes_p, current_file, patient, 
    del f, y_sleep, y_sleep2, sleep_pred, current_file_name, ahi_real_f, ahi_real_class, indx_s, indx_sp, threshold_it
    del threshold, colnames

    ###############################################################################################
    ## Get the same performance metrics but considering all patients                             ##
    ############################################################################################### 
    
    # Save them all first in a df for later ... 
    
    colnames = ["y_reals", "y_preds", "y_sleeps", "y_sleeps_preds", "y_reals_types"]
    # All the lists
    clasif_results_total = pd.concat([pd.DataFrame(y_reals),
                                      pd.DataFrame(y_preds),
                                      pd.DataFrame(y_sleeps),
                                      pd.DataFrame(y_sleeps_preds),
                                      pd.DataFrame(y_reals_types)
                                      ], axis=1)   
    
    
    clasif_results_total .columns = colnames
    # Save in:
    with open(output_dir+"curves/classif_all_"+data_set+".csv", "w") as f:
        clasif_results_total.to_csv(f)
        
        
    # Take the awake ones - Sleep ground truth
    indx_s = np.nonzero(y_sleeps)
    y_preds_sleep = np.array(y_preds)[indx_s]
    y_reals_sleep = np.array(y_reals)[indx_s]
    # Take the awake ones - Sleep predicted
    indx_sp = np.nonzero(np.array(y_sleeps_preds) > threshold_sleep)
    y_preds_sleep_pred = np.array(y_preds)[indx_sp]
    y_reals_sleep_pred = np.array(y_reals)[indx_sp]
    # Lists to save metrics
    thresholds = []
    #### No sleep correction
    # Confusion matrix
    tns_all = []
    fps_all = []
    fns_all = []
    tps_all = []
    # Metrics
    acs_all = []
    prs_all = []
    recs_all = []
    f1s_all = []
    # % of precited positives
    pred_positives = []
    dif_pred_real_positives = []
    # f1 scores for AHI classification
    f1_patients_ws = []
    f1_patients_ms = []
    #### Sleep correction - ground truth
    # Confusion Matrix
    tns_sleep_all = []
    fps_sleep_all = []
    fns_sleep_all = []
    tps_sleep_all = []
    # Metrics
    acs_sleep_all = []
    prs_sleep_all = []
    recs_sleep_all = []
    f1s_sleep_all = []
    # % of precited positives
    pred_positives_sleep = []
    dif_pred_real_positives_sleep = []
    # f1 scores for AHI classification
    f1_patients_ws_sleep = []
    f1_patients_ms_sleep = []
    #### Sleep correction - predicted sleep
    # Confusion Matrix
    tns_sleep_pred_all = []
    fps_sleep_pred_all = []
    fns_sleep_pred_all = []
    tps_sleep_pred_all = []
    # Metrics
    acs_sleep_pred_all = []
    prs_sleep_pred_all = []
    recs_sleep_pred_all = []
    f1s_sleep_pred_all = []
    # % of precited positives
    pred_positives_sleep_pred = []
    dif_pred_real_positives_sleep_pred = []
    # f1 scores for AHI classification
    f1_patients_ws_sleep_pred = []
    f1_patients_ms_sleep_pred = []
    # loop though all thresholds 0 to 1
    for threshold_it in range(0,100):
        threshold = threshold_it/100
        thresholds.append(threshold)
        ### No Sleep correction
        # Predicted positives
        y_preds_binary =  np.array(y_preds) > threshold
        # Confusion matrix
        try:
            tn, fp, fn, tp = metrics.confusion_matrix(y_reals, y_preds_binary).ravel()
        except:
            tn=sum(y_reals==0)
            fp=0
            fn=0
            tp=sum(y_reals==1)
        tns_all.append(tn) 
        fps_all.append(fp)
        fns_all.append(fn)
        tps_all.append(tp)
        # Metrics
        acs_all.append((tn+tp)/(tn+fp+fn+tp))
        prs_all.append(tp/(tp+fp))
        recs_all.append(tp/(tp+fn))
        f1s_all.append(2*tp/(2*tp+fp+fn))
        # % of precited positives
        pred_positives.append(fp+tp)
        dif_pred_real_positives.append(abs((fp+tp)-(fn+tp)))
        # f1 scores for AHI classification
        per_selected = ahi_results_patients[ahi_results_patients.thresholds == threshold]
        f1_patients_w = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_classes, average="weighted")
        f1_patients_m = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_classes, average="micro")
        f1_patients_ws.append(f1_patients_w)
        f1_patients_ms.append(f1_patients_m)
        del tn, fp, fn, tp, per_selected, f1_patients_w, f1_patients_m, y_preds_binary
        ### Sleep correction - ground truth
        y_preds_sleep_binary =  y_preds_sleep > threshold
        # Confusion matrix
        try:
            tn_sleep, fp_sleep, fn_sleep, tp_sleep = metrics.confusion_matrix(y_reals_sleep, y_preds_sleep_binary).ravel()
        except:
            tn_sleep=sum(y_reals_sleep==0)
            fp_sleep=0
            fn_sleep=0
            tp_sleep=sum(y_reals_sleep==1)
        tns_sleep_all.append(tn_sleep) 
        fps_sleep_all.append(fp_sleep)
        fns_sleep_all.append(fn_sleep)
        tps_sleep_all.append(tp_sleep)
        # Metrics
        acs_sleep_all.append((tn_sleep+tp_sleep)/(tn_sleep+fp_sleep+fn_sleep+tp_sleep))
        prs_sleep_all.append(tp_sleep/(tp_sleep+fp_sleep))
        recs_sleep_all.append(tp_sleep/(tp_sleep+fn_sleep))
        f1s_sleep_all.append(2*tp_sleep/(2*tp_sleep+fp_sleep+fn_sleep))
        # % of precited positives
        pred_positives_sleep.append(fp_sleep+tp_sleep)
        dif_pred_real_positives_sleep.append(abs((fp_sleep+tp_sleep)-(fn_sleep+tp_sleep)))
        # f1 scores for AHI classification
        per_selected = ahi_results_patients[ahi_results_patients.thresholds == threshold]
        f1_patients_w = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_sleep_classes, average="weighted")
        f1_patients_m = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_sleep_classes, average="micro")
        f1_patients_ws_sleep.append(f1_patients_w)
        f1_patients_ms_sleep.append(f1_patients_m)
        del tn_sleep, fp_sleep, fn_sleep, tp_sleep, per_selected, f1_patients_w, f1_patients_m, y_preds_sleep_binary
        ### Sleep correction - predicted sleep
        y_preds_sleep_pred_binary =  y_preds_sleep_pred > threshold
        # Confusion matrix
        try:
            tn_sleep_p, fp_sleep_p, fn_sleep_p, tp_sleep_p = metrics.confusion_matrix(y_reals_sleep_pred, y_preds_sleep_pred_binary).ravel()
        except:
            tn_sleep_p=sum(y_reals_sleep_pred==0)
            fp_sleep_p=0
            fn_sleep_p=0
            tp_sleep_p=sum(y_reals_sleep_pred==1)
        tns_sleep_pred_all.append(tn_sleep_p) 
        fps_sleep_pred_all.append(fp_sleep_p)
        fns_sleep_pred_all.append(fn_sleep_p)
        tps_sleep_pred_all.append(tp_sleep_p)
        # Metrics
        acs_sleep_pred_all.append((tn_sleep_p+tp_sleep_p)/(tn_sleep_p+fp_sleep_p+fn_sleep_p+tp_sleep_p))
        prs_sleep_pred_all.append(tp_sleep_p/(tp_sleep_p+fp_sleep_p))
        recs_sleep_pred_all.append(tp_sleep_p/(tp_sleep_p+fn_sleep_p))
        f1s_sleep_pred_all.append(2*tp_sleep_p/(2*tp_sleep_p+fp_sleep_p+fn_sleep_p))
        # % of precited positives
        pred_positives_sleep_pred.append(fp_sleep_p+tp_sleep_p)
        dif_pred_real_positives_sleep_pred.append(abs((fp_sleep_p+tp_sleep_p)-(fn_sleep_p+tp_sleep_p)))
        # f1 scores for AHI classification
        per_selected = ahi_results_patients[ahi_results_patients.thresholds == threshold]
        f1_patients_w = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_sleep_classes_p, average="weighted")
        f1_patients_m = metrics.f1_score(per_selected.ahi_real_classes, per_selected.ahi_sleep_classes_p, average="micro")
        f1_patients_ws_sleep_pred.append(f1_patients_w)
        f1_patients_ms_sleep_pred.append(f1_patients_m)
        del tn_sleep_p, fp_sleep_p, fn_sleep_p, tp_sleep_p, per_selected, f1_patients_w, f1_patients_m, y_preds_sleep_pred_binary

    #  Save all results in dataframe
    # Column names
    colnames = ["thresholds", "tns", "fps", "fns", "tps",
                "acs",  "prs",  "recs",  "f1s",
                "pred_positives", "dif_pred_real_positives", 
                "f1_patients_ws", "f1_patients_ms",
                "tn_sleeps", "fp_sleeps", "fn_sleeps", "tp_sleeps",
                "ac_sleeps", "pr_sleeps", "rec_sleeps",  "f1_sleeps", 
                "pred_positives_sleep", "dif_pred_real_positives_sleep", 
                "f1_patients_ws_sleep", "f1_patients_ms_sleep",
                "tn_sleeps_p", "fp_sleeps_p", "fn_sleeps_p", "tp_sleeps_p",
                "ac_sleep_ps", "pr_sleeps_p", "rec_sleeps_p",  "f1_sleeps_p", 
                "pred_positives_p", "dif_pred_real_positives_p", 
                "f1_patients_ws_p", "f1_patients_ms_p"]
    # All the lists
    classif_results = pd.concat([pd.DataFrame(thresholds), pd.DataFrame(tns_all), pd.DataFrame(fps_all), 
                                pd.DataFrame(fns_all), pd.DataFrame(tps_all), pd.DataFrame(acs_all),  pd.DataFrame(prs_all),  
                                pd.DataFrame(recs_all),  pd.DataFrame(f1s_all),
                                pd.DataFrame(pred_positives), pd.DataFrame(dif_pred_real_positives),
                                pd.DataFrame(f1_patients_ws), pd.DataFrame(f1_patients_ms),
                                pd.DataFrame(tns_sleep_all), pd.DataFrame(fps_sleep_all), pd.DataFrame(fns_sleep_all), pd.DataFrame(tps_sleep_all),
                                pd.DataFrame(acs_sleep_all),  pd.DataFrame(prs_sleep_all),  pd.DataFrame(recs_sleep_all),  pd.DataFrame(f1s_sleep_all),
                                pd.DataFrame(pred_positives_sleep), pd.DataFrame(dif_pred_real_positives_sleep),
                                pd.DataFrame(f1_patients_ws_sleep), pd.DataFrame(f1_patients_ms_sleep),
                                pd.DataFrame(tns_sleep_pred_all), pd.DataFrame(fps_sleep_pred_all), pd.DataFrame(fns_sleep_pred_all), pd.DataFrame(tps_sleep_pred_all),
                                pd.DataFrame(acs_sleep_pred_all),  pd.DataFrame(prs_sleep_pred_all),  pd.DataFrame(recs_sleep_pred_all),  pd.DataFrame(f1s_sleep_pred_all),
                                pd.DataFrame(pred_positives_sleep_pred), pd.DataFrame(dif_pred_real_positives_sleep_pred),
                                pd.DataFrame(f1_patients_ws_sleep_pred), pd.DataFrame(f1_patients_ms_sleep_pred)], axis=1)                     
    classif_results.columns = colnames
    # Save in:
    with open(output_dir+"ahi_scores/"+results_name+"_ahi_scores_classification_"+data_set+".csv", "w") as f:
        classif_results.to_csv(f)

    del indx_s, indx_sp, colnames
    del tns_all, fps_all, fns_all, tps_all, pred_positives, dif_pred_real_positives, f1_patients_ms
    del tns_sleep_all, fps_sleep_all, fns_sleep_all, tps_sleep_all, pred_positives_sleep, dif_pred_real_positives_sleep
    del f1_patients_ms_sleep, tns_sleep_pred_all, fps_sleep_pred_all, fns_sleep_pred_all, tps_sleep_pred_all
    del pred_positives_sleep_pred, dif_pred_real_positives_sleep_pred, f1_patients_ms_sleep_pred, threshold_it, threshold

    # PLOT CLASSIFICATION FIGURE and SAVE RESULTS OF PLOTS FOR LATER 
    fig_new = True


    ####### NO SLEEP CORRECTION ##########

    fpr, tpr, _ = metrics.roc_curve(y_reals, y_preds)
    auroc = metrics.roc_auc_score(y_reals, y_preds)

    pre_arr, rec_arr, _ = metrics.precision_recall_curve(y_reals, y_preds)
    aupr = metrics.average_precision_score(y_reals, y_preds)

    if fig_new:
        fig_subtitle = data_set+" set. Window Size: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_"+data_set
        plot_roc_pr_metrics(fpr, tpr, auroc, rec_arr, pre_arr, aupr, thresholds, acs_all, prs_all, recs_all, f1s_all, fig_subtitle, fig_saveName)
        

    # Save curves if wanted to plot later
    dict_curves = {"fpr_roc": fpr.tolist(), "tpr_roc": tpr.tolist(),  "auroc": auroc, "pre_pr": pre_arr.tolist(), 
                "rec_pr": rec_arr.tolist(), "aupr": aupr, "thresholds": thresholds, "acc": acs_all, "prs": prs_all, 
                "rec": recs_all, "f1": f1s_all, "f1_patients": f1_patients_ws}
    with open(output_dir+"curves/"+results_name+"_"+data_set+"_curves.txt", "w", ) as fp:
        json.dump(dict_curves, fp)

    del fpr, tpr, auroc, pre_arr, rec_arr, aupr, acs_all, prs_all, recs_all, f1s_all, f1_patients_ws
    del y_reals, y_preds
    del fig_subtitle, fig_saveName, dict_curves, fp
    
    ####### SLEEP GROUND TRUTH ##########

    fpr_sleep, tpr_sleep, _ = metrics.roc_curve(y_reals_sleep, y_preds_sleep)
    auroc_sleep = metrics.roc_auc_score(y_reals_sleep, y_preds_sleep)
        
    pre_arr_sleep, rec_arr_sleep, _ = metrics.precision_recall_curve(y_reals_sleep, y_preds_sleep)
    aupr_sleep = metrics.average_precision_score(y_reals_sleep, y_preds_sleep)

    if fig_new:
        fig_subtitle = data_set+" set. Window Size: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_"+data_set+"_s"
        plot_roc_pr_metrics(fpr_sleep, tpr_sleep, auroc_sleep, rec_arr_sleep, pre_arr_sleep, aupr_sleep, thresholds, acs_sleep_all, prs_sleep_all, recs_sleep_all, f1s_sleep_all, fig_subtitle, fig_saveName)
        

    # Save curves if wanted to plot later
    dict_curves = {"fpr_roc": fpr_sleep.tolist(), "tpr_roc": tpr_sleep.tolist(),  "auroc": auroc_sleep, "pre_pr": pre_arr_sleep.tolist(), 
                "rec_pr": rec_arr_sleep.tolist(), "aupr": aupr_sleep, "thresholds": thresholds, "acc": acs_sleep_all, "prs": prs_sleep_all, 
                "rec": recs_sleep_all, "f1": f1s_sleep_all, "f1_patients": f1_patients_ws_sleep}

    with open(output_dir+"curves/"+results_name+"_"+data_set+"_curves_s.txt", "w", ) as fp:
        json.dump(dict_curves, fp)

    del fpr_sleep, tpr_sleep, auroc_sleep, pre_arr_sleep, rec_arr_sleep, aupr_sleep, acs_sleep_all, prs_sleep_all, recs_sleep_all, f1s_sleep_all, f1_patients_ws_sleep
    del y_reals_sleep, y_preds_sleep
    del fig_subtitle, fig_saveName, dict_curves, fp
    
    ####### SLEEP PREDICTED ##########  

    fpr_sleep_pred, tpr_sleep_pred, _ = metrics.roc_curve(y_reals_sleep_pred, y_preds_sleep_pred)
    auroc_sleep_pred = metrics.roc_auc_score(y_reals_sleep_pred, y_preds_sleep_pred)
        
    pre_arr_sleep_pred, rec_arr_sleep_pred, _ = metrics.precision_recall_curve(y_reals_sleep_pred, y_preds_sleep_pred)
    aupr_sleep_pred = metrics.average_precision_score(y_reals_sleep_pred, y_preds_sleep_pred)

    if fig_new:
        fig_subtitle = data_set+" set. Window Size: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_"+data_set+"_sp"
        plot_roc_pr_metrics(fpr_sleep_pred, tpr_sleep_pred, auroc_sleep_pred, rec_arr_sleep_pred, pre_arr_sleep_pred, aupr_sleep_pred, thresholds, acs_sleep_pred_all, prs_sleep_pred_all, recs_sleep_pred_all, f1s_sleep_pred_all, fig_subtitle, fig_saveName)
        

    # Save curves if wanted to plot later
    dict_curves = {"fpr_roc": fpr_sleep_pred.tolist(), "tpr_roc": tpr_sleep_pred.tolist(),  "auroc": auroc_sleep_pred, "pre_pr": pre_arr_sleep_pred.tolist(), 
                "rec_pr": rec_arr_sleep_pred.tolist(), "aupr": aupr_sleep_pred, "thresholds": thresholds, "acc": acs_sleep_pred_all, "prs": prs_sleep_pred_all, 
                "rec": recs_sleep_pred_all, "f1": f1s_sleep_pred_all, "f1_patients": f1_patients_ws_sleep_pred}
        
    with open(output_dir+"curves/"+results_name+"_"+data_set+"_curves_s_pred.txt", "w", ) as fp:
        json.dump(dict_curves, fp)

    del fpr_sleep_pred, tpr_sleep_pred, auroc_sleep_pred, pre_arr_sleep_pred, rec_arr_sleep_pred, aupr_sleep_pred
    del acs_sleep_pred_all, prs_sleep_pred_all, recs_sleep_pred_all, f1s_sleep_pred_all, f1_patients_ws_sleep_pred
    del thresholds, y_reals_sleep_pred, y_preds_sleep_pred
    
    del fig_subtitle, fig_saveName, dict_curves, fp
    
    ###############################################################################################
    ## Choose/Use threshold (from VAL/TEST) for AHI-event classification                         ##
    ############################################################################################### 
    # If validation, chose the threshold that:
    if data_set == "VAL":
        ### Non correction
        # 1) Best F1-AHI-event classification
        classif_results_best_f1_classif = classif_results[classif_results.f1s == (classif_results.f1s).max()]
        threshold_best_f1_classif = classif_results_best_f1_classif.iloc[0]['thresholds']
        # 2) Crossing of Precision and Recall curves
        classif_results_crossing_pr = classif_results[abs(classif_results.prs - classif_results.recs) == abs(classif_results.prs - classif_results.recs).min()]
        threshold_crossing_pr =  classif_results_crossing_pr.iloc[0]['thresholds']
        # 3) Best F1_ahi classification
        classif_results_best_f1_ahi = classif_results[classif_results.f1_patients_ms == (classif_results.f1_patients_ms).max()]
        threshold_best_f1_ahi =  classif_results_best_f1_ahi.iloc[0]['thresholds']  
        # Save them     
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
        ### Sleep corrected - ground truth
        # 1) Best F1-AHI-event classification
        classif_results_best_f1_classif_s = classif_results[classif_results.f1_sleeps == (classif_results.f1_sleeps).max()]
        threshold_best_f1_classif_s = classif_results_best_f1_classif_s.iloc[0]['thresholds']
        # 2) Crossing of Precision and Recall curves
        classif_results_crossing_pr_s= classif_results[abs(classif_results.pr_sleeps - classif_results.rec_sleeps) == abs(classif_results.pr_sleeps - classif_results.rec_sleeps).min()]
        threshold_crossing_pr_s =  classif_results_crossing_pr_s.iloc[0]['thresholds']
        # 3) Best F1_ahi classification
        classif_results_best_f1_ahi_s = classif_results[classif_results.f1_patients_ms_sleep == (classif_results.f1_patients_ms_sleep).max()]
        threshold_best_f1_ahi_s =  classif_results_best_f1_ahi_s.iloc[0]['thresholds']    
        # Save them     
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('F1-CL-Threshold-s,'+str(threshold_best_f1_classif_s))
            f.write('\n')
            f.write(classif_results_best_f1_classif_s.iloc[[0]].to_string())
            f.write('\n')
            f.write('PR-Threshold-s,'+str(threshold_crossing_pr_s))
            f.write('\n')
            f.write(classif_results_crossing_pr_s.iloc[[0]].to_string())
            f.write('\n')
            f.write('F1-AHI-Threshold-s,'+str(threshold_best_f1_ahi_s))
            f.write('\n')
            f.write(classif_results_best_f1_ahi_s.iloc[[0]].to_string())
        ### Sleep corrected - Predicted
        # 1) Best F1-AHI-event classification
        classif_results_best_f1_classif_sp = classif_results[classif_results.f1_sleeps_p == (classif_results.f1_sleeps_p).max()]
        threshold_best_f1_classif_sp = classif_results_best_f1_classif_sp.iloc[0]['thresholds']
        # 2) Crossing of Precision and Recall curves
        classif_results_crossing_pr_sp = classif_results[abs(classif_results.pr_sleeps_p - classif_results.rec_sleeps_p) == abs(classif_results.pr_sleeps_p - classif_results.rec_sleeps_p).min()]
        threshold_crossing_pr_sp =  classif_results_crossing_pr_sp.iloc[0]['thresholds']
        # 3) Best F1_ahi classification 
        classif_results_best_f1_ahi_sp = classif_results[classif_results.f1_patients_ms_p == (classif_results.f1_patients_ms_p).max()]
        threshold_best_f1_ahi_sp =  classif_results_best_f1_ahi_sp.iloc[0]['thresholds']    
        # Save them 
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('F1-CL-Threshold-sp,'+str(threshold_best_f1_classif_sp))
            f.write('\n')
            f.write(classif_results_best_f1_classif_sp.iloc[[0]].to_string())
            f.write('\n')
            f.write('PR-Threshold-sp,'+str(threshold_crossing_pr_sp))
            f.write('\n')
            f.write(classif_results_crossing_pr_sp.iloc[[0]].to_string())
            f.write('\n')
            f.write('F1-AHI-Threshold-sp,'+str(threshold_best_f1_ahi_sp))
            f.write('\n')
            f.write(classif_results_best_f1_ahi_sp.iloc[[0]].to_string())
        
        
            
    # If test read thresholds
    elif data_set == "TEST":
        # Get the best rows from thresholds decided on validation set. 
        classif_results_best_f1_classif = classif_results[classif_results.thresholds == threshold_best_f1_classif]
        classif_results_crossing_pr = classif_results[classif_results.thresholds == threshold_crossing_pr]
        classif_results_best_f1_ahi = classif_results[classif_results.thresholds == threshold_best_f1_ahi]
        classif_results_best_f1_classif_s = classif_results[classif_results.thresholds == threshold_best_f1_classif_s]
        classif_results_crossing_pr_s = classif_results[classif_results.thresholds == threshold_crossing_pr_s]
        classif_results_best_f1_ahi_s = classif_results[classif_results.thresholds == threshold_best_f1_ahi_s]
        classif_results_best_f1_classif_sp = classif_results[classif_results.thresholds == threshold_best_f1_classif_sp]
        classif_results_crossing_pr_sp = classif_results[classif_results.thresholds == threshold_crossing_pr_sp]
        classif_results_best_f1_ahi_sp = classif_results[classif_results.thresholds == threshold_best_f1_ahi_sp]
        # Write same file for TEST
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "w") as f:
            f.write('F1-CL-Threshold,'+str(threshold_best_f1_classif))
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
            f.write('\n')
            f.write('F1-CL-Threshold,'+str(threshold_best_f1_classif_s))
            f.write('\n')
            f.write(classif_results_best_f1_classif_s.to_string())
            f.write('\n')
            f.write('PR-Threshold,'+str(threshold_crossing_pr_s))
            f.write('\n')
            f.write(classif_results_crossing_pr_s.to_string())
            f.write('\n')
            f.write('F1-AHI-Threshold,'+str(threshold_best_f1_ahi_s))
            f.write('\n')
            f.write(classif_results_best_f1_ahi_s.to_string())
            f.write('\n')
            f.write('F1-CL-Threshold-sp,'+str(threshold_best_f1_classif_sp))
            f.write('\n')
            f.write(classif_results_best_f1_classif_sp.to_string())
            f.write('\n')
            f.write('PR-Threshold-sp,'+str(threshold_crossing_pr_sp))
            f.write('\n')
            f.write(classif_results_crossing_pr_sp.to_string())
            f.write('\n')
            f.write('F1-AHI-Threshold-sp,'+str(threshold_best_f1_ahi_sp))
            f.write('\n')
            f.write(classif_results_best_f1_ahi_sp.to_string())

    del classif_results_best_f1_classif, classif_results_crossing_pr, classif_results_best_f1_ahi
    del classif_results_best_f1_classif_s, classif_results_crossing_pr_s, classif_results_best_f1_ahi_s
    del classif_results_best_f1_classif_sp, classif_results_crossing_pr_sp, classif_results_best_f1_ahi_sp
    ###############################################################################################
    ## With those thresholds do AHI Predictions and the scatter pots - confusion matrix          ##
    ############################################################################################### 
    
    patients_real = ahi_results_patients[ahi_results_patients.thresholds==0.5]
    patients_ahi = (patients_real.ahi_reals).values
    ahi_real_classes =  (patients_real.ahi_real_classes).values

    # Create weights for the regression depending on the frequency of each class in the Validation Set
    if data_set == "VAL":
        w0 = 1 - sum(ahi_real_classes==0)/len(ahi_real_classes)
        w1 = 1 - sum(ahi_real_classes==1)/len(ahi_real_classes)
        w2 = 1 - sum(ahi_real_classes==2)/len(ahi_real_classes)
        w3 = 1 - sum(ahi_real_classes==3)/len(ahi_real_classes)

        weights = []
        for pred in patients_ahi:
            if calculate_ahi_class(pred) == 0:
                weights.append(w0)
            elif calculate_ahi_class(pred) == 1:
                weights.append(w1)
            elif calculate_ahi_class(pred) == 2:
                weights.append(w2)
            elif calculate_ahi_class(pred) == 3:
                weights.append(w3)
        
        del w0, w1, w2, w3

    # Set colors of different classes
    colors_scale = ["r", "b", "g", "m"]
    colors_points = [colors_scale[int(c)] for c in ahi_real_classes]
    
    del patients_real
    #######################
    #del threshold_best_f1_classif, threshold_best_f1_classif_s, threshold_best_f1_classif_sp

    ### No sleep correction   
    
    # A. Best F1 Clasification AHI-events  
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_classif]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positives).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_classif,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_f1_classif,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_f1_classif,'+str(f1_score_plot))
        
        del reg
            
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positives).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_f1_classif+inter_best_f1_classif
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_classif,'+str(factor_best_f1_classif))
            f.write('\n')
            f.write('REG_intercept_best_f1_classif,'+str(inter_best_f1_classif))
            f.write('\n')
            f.write('Corrected_F1_best_f1_classif,'+str(f1_score_plot))
        
        del factor_best_f1_classif, inter_best_f1_classif   
            
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class 
    with open(output_dir+"best_ahis/"+results_name+"_threshold_best_f1_classif_"+data_set+".csv", "w") as f:
        patients_best.to_csv(f)
        

    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestF1classif_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
    
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_best_f1_classif, f1_score_plot
    del patients_best
    
    # B. Crossing of Pre and Rec curves
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_crossing_pr]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positives).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_best_th_crossing_pr,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_th_crossing_pr,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_th_crossing_pr,'+str(f1_score_plot))
        
        del reg
        
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positives).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_th_crossing_pr+inter_best_th_crossing_pr
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_th_crossing_pr,'+str(factor_best_th_crossing_pr))
            f.write('\n')
            f.write('REG_intercept_best_th_crossing_pr,'+str(inter_best_th_crossing_pr))  
            f.write('\n')
            f.write('Corrected_F1_best_th_crossing_pr,'+str(f1_score_plot))  
        del factor_best_th_crossing_pr, inter_best_th_crossing_pr
   
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class 
    with open(output_dir+"best_ahis/"+results_name+"_best_patient_PRscores_"+data_set+".csv", "w") as f:
        patients_best.to_csv(f)
        
    
    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestPCcross_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
    
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_crossing_pr, f1_score_plot
    del patients_best
    
    
    
    
    
    # C. Best F1 Clasification AHI-classes  
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_ahi]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positives).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_ahi,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_f1_ahi,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_f1_ahi,'+str(f1_score_plot))
        
        del reg
            
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positives).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_f1_ahi+inter_best_f1_ahi
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_ahi,'+str(factor_best_f1_ahi))
            f.write('\n')
            f.write('REG_intercept_best_f1_ahi,'+str(inter_best_f1_ahi))
            f.write('\n')
            f.write('Corrected_F1_best_f1_ahi,'+str(f1_score_plot))
        
        del factor_best_f1_ahi, inter_best_f1_ahi    
            
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class 
    with open(output_dir+"best_ahis/"+results_name+"_threshold_best_f1_ahi_"+data_set+".csv", "w") as f:
        patients_best.to_csv(f)
        

    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestF1AHI_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
    
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_best_f1_ahi, f1_score_plot
    del patients_best
    
    
    
    ### Sleep Correction - Groung truth 
    
    # A. Best F1 Clasification AHI-events  
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_classif_s]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_classif_s,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_f1_classif_s,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_f1_classif_s,'+str(f1_score_plot))
        
        del reg
            
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_f1_classif_s+inter_best_f1_classif_s
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_classif_s,'+str(factor_best_f1_classif_s))
            f.write('\n')
            f.write('REG_intercept_best_f1_classif_s,'+str(inter_best_f1_classif_s))
            f.write('\n')
            f.write('Corrected_F1_best_f1_classif_s,'+str(f1_score_plot))
        del factor_best_f1_classif_s, inter_best_f1_classif_s
            
    
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class         
    with open(output_dir+"best_ahis/"+results_name+"_threshold_best_f1_classif_"+data_set+"_s.csv", "w") as f:
        patients_best.to_csv(f)
        

    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestF1classif_s_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
            
        
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_best_f1_classif_s, f1_score_plot
    del patients_best
    
    # B. Crossing of Pre and Rec curves
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_crossing_pr_s]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_best_th_crossing_pr_s,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_th_crossing_pr_s,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_th_crossing_pr_s,'+str(f1_score_plot))
        
        del reg
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_th_crossing_pr_s+inter_best_th_crossing_pr_s
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_th_crossing_pr_s,'+str(factor_best_th_crossing_pr_s))
            f.write('\n')
            f.write('REG_intercept_best_th_crossing_pr_s,'+str(inter_best_th_crossing_pr_s))  
            f.write('\n')
            f.write('Corrected_F1_best_th_crossing_pr_s,'+str(f1_score_plot))  
        
        del factor_best_th_crossing_pr_s, inter_best_th_crossing_pr_s
    
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class 
    with open(output_dir+"best_ahis/"+results_name+"_best_patient_PRscores_"+data_set+"_s.csv", "w") as f:
        patients_best.to_csv(f)
        
    
    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestPCcross_s_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
        
        
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_crossing_pr_s, f1_score_plot
    del patients_best
    
    
    # C. Best F1 Clasification AHI-classes
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_ahi_s]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_ahi_s,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_f1_ahi_s,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_f1_ahi_s,'+str(f1_score_plot))
        
        del reg
            
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positive_sleeps).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_f1_ahi_s+inter_best_f1_ahi_s
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_ahi_s,'+str(factor_best_f1_ahi_s))
            f.write('\n')
            f.write('REG_intercept_best_f1_ahi_s,'+str(inter_best_f1_ahi_s))
            f.write('\n')
            f.write('Corrected_F1_best_f1_ahi_s,'+str(f1_score_plot))
        del factor_best_f1_ahi_s, inter_best_f1_ahi_s
            
    
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class         
    with open(output_dir+"best_ahis/"+results_name+"_threshold_best_f1_ahi_"+data_set+"_s.csv", "w") as f:
        patients_best.to_csv(f)
        

    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestF1AHI_s_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
            
        
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_best_f1_ahi_s, f1_score_plot
    del patients_best
    
    ### Sleep Correction - Predicted Sleep
    
    # A. Best F1 Clasification AHI-events  
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_classif_sp]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positive_sleeps_p).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_classif_sp,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_f1_classif_sp,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_f1_classif_sp,'+str(f1_score_plot))
        
        del reg
            
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positive_sleeps_p).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_f1_classif_sp+inter_best_f1_classif_sp
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_classif_sp,'+str(factor_best_f1_classif_sp))
            f.write('\n')
            f.write('REG_intercept_best_f1_classif_sp,'+str(inter_best_f1_classif_sp))
            f.write('\n')
            f.write('Corrected_F1_best_f1_classif_sp,'+str(f1_score_plot))
        
        del factor_best_f1_classif_sp, inter_best_f1_classif_sp
            
    
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class 
          
    with open(output_dir+"best_ahis/"+results_name+"_threshold_best_f1_ahi_"+data_set+"_sp.csv", "w") as f:
        patients_best.to_csv(f)
        

    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestF1classif_sp_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
            
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_best_f1_classif_sp, f1_score_plot    
    del patients_best
    
    # B. Crossing of Pre and Rec curves
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_crossing_pr_sp]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positive_sleeps_p).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_best_th_crossing_pr_sp,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_th_crossing_pr_sp,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_th_crossing_pr_sp,'+str(f1_score_plot))
        
        del reg
        
        
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positive_sleeps_p).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_th_crossing_pr_sp+inter_best_th_crossing_pr_sp
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_th_crossing_pr_sp,'+str(factor_best_th_crossing_pr_sp))
            f.write('\n')
            f.write('REG_intercept_best_th_crossing_pr_sp,'+str(inter_best_th_crossing_pr_sp))  
            f.write('\n')
            f.write('Corrected_F1_best_th_crossing_pr_sp,'+str(f1_score_plot))  
        
        del factor_best_th_crossing_pr_sp, inter_best_th_crossing_pr_sp
    
   
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class 
    with open(output_dir+"best_ahis/"+results_name+"_best_patient_PRscores_"+data_set+"_sp.csv", "w") as f:
        patients_best.to_csv(f)
        
    
    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestPCcross_sp_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)    
   
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_crossing_pr_sp, f1_score_plot
    del patients_best
    
    
    # C. Best F1 Clasification AHI-events  
    patients_best = ahi_results_patients[ahi_results_patients.thresholds==threshold_best_f1_ahi_sp]
    
    if data_set == "VAL":
        patients_predpos =  ((patients_best.per_positive_sleeps_p).values).reshape(-1, 1)
        # Do regression of AHI based on percentage of positives
        reg = LinearRegression(fit_intercept=inter)
        reg.fit(patients_predpos, patients_ahi, weights)
        # Do predictions
        predictions = reg.predict(patients_predpos)
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_VAL.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_ahi_sp,'+str(reg.coef_))
            f.write('\n')
            f.write('REG_intercept_best_f1_ahi_sp,'+str(reg.intercept_))
            f.write('\n')
            f.write('Corrected_F1_best_f1_ahi_sp,'+str(f1_score_plot))
        
        del reg
            
    
    elif data_set == "TEST":
        patients_predpos =  ((patients_best.per_positive_sleeps_p).values).reshape(-1, 1)
        predictions = patients_predpos*factor_best_f1_ahi_sp+inter_best_f1_ahi_sp
        predicted_class = [calculate_ahi_class(pred) for pred in predictions]
        f1_score_plot = metrics.f1_score(ahi_real_classes, predicted_class, average="micro")
        with open(output_dir+"best_ahis/"+results_name+"_thresholds_TEST.txt", "a") as f:
            f.write('\n')
            f.write('REG_coef_bestf_f1_ahi_sp,'+str(factor_best_f1_ahi_sp))
            f.write('\n')
            f.write('REG_intercept_best_f1_ahi_sp,'+str(inter_best_f1_ahi_sp))
            f.write('\n')
            f.write('Corrected_F1_best_f1_ahi_sp,'+str(f1_score_plot))
        
        del factor_best_f1_ahi_sp, inter_best_f1_ahi_sp
            
    
    patients_best["new_ahi"] = predictions 
    patients_best["new_ahi_class"] = predicted_class 
          
    with open(output_dir+"best_ahis/"+results_name+"_threshold_best_f1_ahi_"+data_set+"_sp.csv", "w") as f:
        patients_best.to_csv(f)
        

    if fig_new:
        fig_subtitle = data_set+" set. WS: "+str(ws)+", Trial: "+fold+", Channels: "+channels_name
        fig_saveName = output_dir+"figures/"+results_name+"_ahi_scores_classification_allLines_thBestF1AHI_sp_"+data_set
        plot_scatter_confusion(patients_ahi, ahi_real_classes, predictions, predicted_class, fig_subtitle, fig_saveName)
            
    del patients_predpos, predictions, predicted_class, fig_subtitle, fig_saveName
    del threshold_best_f1_ahi_sp, f1_score_plot    
    del patients_best