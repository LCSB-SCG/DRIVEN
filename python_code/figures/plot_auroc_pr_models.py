import numpy as np
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import json
import matplotlib.pylab as pylab

import seaborn as sns

import tikzplotlib


palette = sns.palettes.color_palette("bright", n_colors=7)
sns.set_palette(palette)
sns.set_style(
    style="white",
)


  
def read_dict_get_curves_sleep(src, model, channels, fold, data_set):
    file_dict = src+"EVAL_MODELS/SLEEP/curves/"+model+"_"+str(fold)+"_Ch_"+channels+"_sleep_"+data_set+"_metrics.txt"
    with open(file_dict, "r", ) as fp:
        # Load the dictionary from the file
        dict_curves = json.load(fp)
    
    return (dict_curves["fpr_roc"], dict_curves["tpr_roc"], dict_curves["auroc"], dict_curves["pre_pr"], 
            dict_curves["rec_pr"], dict_curves["aupr"])
    

def read_dict_get_curves(src, model, channels, fold, data_set):
    file_dict = src+"EVAL_MODELS/MOVING_AVERAGE/"+channels+"/curves/"+model+"_f"+str(fold)+"_ch_"+channels+"__"+data_set+"_curves.txt"
    with open(file_dict, "r", ) as fp:
        # Load the dictionary from the file
        dict_curves = json.load(fp)
    
    return (dict_curves["fpr_roc"], dict_curves["tpr_roc"], dict_curves["auroc"], dict_curves["pre_pr"], 
            dict_curves["rec_pr"], dict_curves["aupr"], dict_curves["acc"], dict_curves["prs"],
            dict_curves["rec"], dict_curves["f1"], dict_curves["f1_patients"])

"""
file1_train = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/SPLIT/dict_TRAIN_split2.json"    
file1_test = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/SPLIT/dict_TEST_split2.json"    
file1_val = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/SPLIT/dict_VAL_split2.json"    

positives = 0
negatives = 0

with open(file1_val, "r", ) as fp:
        # Load the dictionary from the file
        dict_curves = json.load(fp)

for key in dict_curves.keys():
    positives += dict_curves[key]["n_ahi"]
    negatives += dict_curves[key]["n_sleep"]

with open(file1_train , "r", ) as fp:
        # Load the dictionary from the file
        dict_curves = json.load(fp)

for key in dict_curves.keys():
    positives += dict_curves[key]["n_ahi"]
    negatives += dict_curves[key]["n_sleep"]
    #negatives += dict_curves[key]["n_awake"]


with open(file1_test , "r", ) as fp:
        # Load the dictionary from the file
        dict_curves = json.load(fp)

for key in dict_curves.keys():
    positives += dict_curves[key]["n_ahi"]
    negatives += dict_curves[key]["n_sleep"]
    #negatives += dict_curves[key]["n_awake"]


"""


if __name__ == '__main__':
    
    # python eval_predictions_ahiclass.py "/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/" 80 "VAL"

    params = {'legend.fontsize': 'large',
            #'figure.figsize': (15, 5),
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}
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
    
    src = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/"


    channels_try = [["2", "3", "5"]]

    comb_2 = list(map(list,list(combinations(channels_try[0], 2) )))
    comb_1 = list(map(list,list(combinations(channels_try[0], 1) )))

    channels_try.extend(comb_2)
    channels_try.extend(comb_1)
        
    
    ch_names = ["Abdo.", "Thor.", "Flow", "SpO2"]
    channels_try = [["2_5"]]
    sleep = False
    apnea = True
    
    if apnea:
        output_dir = src+"EVAL_MODELS/MOVING_AVERAGE/"
        
        #sns.set_style("darkgrid")
            
        colors = ["#FA0505", "#DB1CE3", "#183C87", "#2AD593", "#EBB714", "#798686", "#AF505A"]
        for model in ["Model2"]: #, "Model3"]:
            print(model)
            for fold in [2]:
                print(fold)
                for data_set in ["VAL"]: #, "VAL"]:
                    print(data_set)
            
                    auroc = False
                    if auroc:
                        fig = plt.figure(figsize=(5, 5))
                        i = 0
                        for channels in channels_try:
                            
                            channels = "_".join(channels)
                            channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])
                            fpr_roc, tpr_roc, auroc, pre_pr, rec_pr, aupr, ac, prs, recs, f1s, f1s_ahi = read_dict_get_curves(src, model, channels, fold, data_set)
                            plt.plot(fpr_roc, tpr_roc, label=channels_name+f' AUROC = {auroc:.2f}')
                            i += 1
                        
                        plt.title("ROC Curve")
                        plt.xlabel("FPR")
                        plt.ylabel("TPR")
                        plt.xlim([0, 1]) 
                        plt.ylim([0, 1]) 
                        #axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
                        plt.legend()
                        fig.tight_layout()
                        #fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig2_A_v1.pdf", format="pdf", bbox_inches="tight")
                        #tikzplotlib.save("/scratch/users/mretamales/OSA_scratch/new_pipeline/test.tex")
                        
                        fig = plt.figure(figsize=(5, 5))
                        i = 0
                        for channels in channels_try:
                            
                            channels = "_".join(channels)
                            channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])
                            fpr_roc, tpr_roc, auroc, pre_pr, rec_pr, aupr, ac, prs, recs, f1s, f1s_ahi = read_dict_get_curves(src, model, channels, fold, data_set)
                            plt.plot(rec_pr, pre_pr, label=channels_name+f' AUPR = {aupr:.2f}')
                            i += 1
                                
                        
                        plt.title("PR Curve")
                        plt.xlabel("Recall")
                        plt.ylabel("Precision")
                        plt.xlim([0, 1]) 
                        plt.ylim([0, 1]) 
                        #axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
                        plt.legend()
                        fig.tight_layout()
                        #fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig2_B_v1.pdf", format="pdf", bbox_inches="tight")
                        
                    
                    
                    for channels in channels_try:
                        fig = plt.figure(figsize=(5, 5))
                        channels = "_".join(channels)
                        
                        #
                        print("1")
                        fpr_roc, tpr_roc, auroc, pre_pr, rec_pr, aupr, ac, prs, recs, f1s, f1s_ahi = read_dict_get_curves(src, model, channels, fold, data_set)
                
                        thresholds = np.arange(40,90)/100
                        plt.plot(thresholds, ac, label="Accuracy")
                        plt.plot(thresholds, prs, label="Precision")
                        plt.plot(thresholds, recs, label="Recall")
                        plt.plot(thresholds, f1s, label="F1-Event-Classification")
                        plt.plot(thresholds, f1s_ahi, label="F1-AHI-Classification")
                        plt.plot([0.76, 0.76], [0,1], label="Best Threshold")
                            
                            
                    
                        #plt.title("Metrics")
                        if channels == "2_5":
                            plt.title("Val. Ch. Abdomial and SpO2")
                        plt.xlabel("Threshold")
                        plt.ylabel("Score")
                        plt.legend()
                        plt.grid()
                        plt.xlim([0.4, 0.9]) 
                        plt.ylim([0, 1]) 
                        plt.yticks(np.arange(0, 1.1, 0.2))
                        plt.xticks(np.arange(0.4, 0.91, 0.1))
                        #plt.xlim([0, 1]) 
                        #plt.ylim([0, 1]) 
                        #axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
                        plt.legend()
                        fig.tight_layout()
                        fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig_sup_metrics_"+channels+"_VAL.pdf", format="pdf", bbox_inches="tight")
                
                
                
                """
                fig = plt.figure(figsize=(5, 5))
                for channels in channels_try:
                    channels = "_".join(channels)
                    channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])
                    fpr_roc, tpr_roc, auroc, pre_pr, rec_pr, aupr, ac, prs, recs, f1s, f1s_ahi = read_dict_get_curves(src, model, channels, fold, data_set)
                    plt.plot(fpr_roc, tpr_roc, label=channels_name+f' AUROC = {auroc:.2f}')
                    #axs[1].plot(rec_pr, pre_pr, label=channels_name+f' AUPR = {aupr:.2f}')
                    ""if channels == "2_5":
                        print("1")
                        thresholds = np.arange(40,90)/100
                        axs[2].plot(thresholds, ac, label="Accuracy")
                        axs[2].plot(thresholds, prs, label="Precision")
                        axs[2].plot(thresholds, recs, label="Recall")
                        axs[2].plot(thresholds, f1s, label="F1-Event-Classification")
                        axs[2].plot(thresholds, f1s_ahi, label="F1-Patients-Classification")""
                        
                
                
                plt.title("ROC Curve")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.xlim([0, 1]) 
                plt.ylim([0, 1]) 
                #axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
                plt.legend()
                
                fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig2_A_v1.png")
                
                
                ""
                axs[1].set_title("PR Curve")
                axs[1].set_xlabel("Recall")
                axs[1].set_ylabel("Precision")
                axs[1].set_xlim([0, 1]) 
                axs[1].set_ylim([0, 1]) 
                #axs[1].text(-0.1, 1.1, "B", size=20, weight='bold', transform=axs[1].transAxes)
                axs[1].legend()


                axs[2].set_title("Metrics Ch. Abdomial and SpO2")
                axs[2].set_xlabel("Threshold")
                axs[2].set_ylabel("Score")
                axs[2].legend()
                #axs[2].grid()
                axs[2].text(-0.1, 1.1, "B", size=20, weight='bold', transform=axs[2].transAxes)
                axs[2].set_xlim([0.4, 0.9]) 
                axs[2].set_ylim([0, 1]) 
                axs[2].set_yticks(np.arange(0, 1.1, 0.2))
                axs[2].set_xticks(np.arange(0.4, 0.91, 0.1))
                fig.tight_layout()
                #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                #fig.suptitle(data_set+" set. Window Size: "+str(ws)+", Trial: "+str(fold), fontsize=20)
                """

    if sleep:
        
        #sns.set_style("darkgrid")
            
        colors = ["#FA0505", "#DB1CE3", "#183C87", "#2AD593", "#EBB714", "#798686", "#AF505A"]
        for model in ["Model2"]: #, "Model3"]:
            print(model)
            for fold in [2]:
                print(fold)
                for data_set in ["TEST"]: #, "VAL"]:
                    print(data_set)
            
                    
                    fig = plt.figure(figsize=(5, 5))
                    i = 0
                    for channels in channels_try:
                        
                        channels = "_".join(channels)
                        channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])
                        fpr_roc, tpr_roc, auroc, pre_pr, rec_pr, aupr = read_dict_get_curves_sleep(src, model, channels, fold, data_set)
                        plt.plot(fpr_roc, tpr_roc, label=channels_name+f' AUROC = {auroc:.2f}')
                        i += 1
                            
                    
                    plt.title("Test ROC Curve")
                    plt.xlabel("FPR")
                    plt.ylabel("TPR")
                    plt.xlim([0, 1]) 
                    plt.ylim([0, 1]) 
                    #axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
                    plt.legend()
                    fig.tight_layout()
                    fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/sleep_roc_T.pdf", format="pdf", bbox_inches="tight")
                    
                    
                    fig = plt.figure(figsize=(5, 5))
                    i = 0
                    for channels in channels_try:
                        
                        channels = "_".join(channels)
                        channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])
                        fpr_roc, tpr_roc, auroc, pre_pr, rec_pr, aupr = read_dict_get_curves_sleep(src, model, channels, fold, data_set)
                        plt.plot(rec_pr, pre_pr, label=channels_name+f' AUPR = {aupr:.2f}')
                        i += 1
                            
                    
                    plt.title("Test PR Curve")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.xlim([0, 1]) 
                    plt.ylim([0, 1]) 
                    #axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
                    plt.legend()
                    fig.tight_layout()
                    fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/sleep_pr_T.pdf", format="pdf", bbox_inches="tight")
                    
        

           

        
        

        
    
    

   

        
        
    
    
    
    """ 
    
    for model in ["Model2"]: #, "Model3"]:
        print(model)
        for fold in [2]:
            print(fold)
            for data_set in ["TEST"]: #, "VAL"]:
                print(data_set)
        
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                for channels in channels_try:
                    channels = "_".join(channels)
                    channels_name = ", ".join([ch_names[int(ch)-2] for ch in channels.split("_")])
                    fpr_roc, tpr_roc, auroc, pre_pr, rec_pr, aupr, ac, prs, recs, f1s, f1s_ahi = read_dict_get_curves(src, model, channels, fold, data_set)
                    axs[0].plot(fpr_roc, tpr_roc, label=channels_name+f' AUROC = {auroc:.2f}')
                    axs[1].plot(rec_pr, pre_pr, label=channels_name+f' AUPR = {aupr:.2f}')
                    if channels == "2_5":
                        print("1")
                        thresholds = np.arange(40,90)/100
                        axs[2].plot(thresholds, ac, label="Accuracy")
                        axs[2].plot(thresholds, prs, label="Precision")
                        axs[2].plot(thresholds, recs, label="Recall")
                        axs[2].plot(thresholds, f1s, label="F1-Event-Classification")
                        axs[2].plot(thresholds, f1s_ahi, label="F1-Patients-Classification")
                        
                
                
                axs[0].set_title("ROC Curve")
                axs[0].set_xlabel("FPR")
                axs[0].set_ylabel("TPR")
                axs[0].set_xlim([0, 1]) 
                axs[0].set_ylim([0, 1]) 
                axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
                axs[0].legend()

                axs[1].set_title("PR Curve")
                axs[1].set_xlabel("Recall")
                axs[1].set_ylabel("Precision")
                axs[1].set_xlim([0, 1]) 
                axs[1].set_ylim([0, 1]) 
                #axs[1].text(-0.1, 1.1, "B", size=20, weight='bold', transform=axs[1].transAxes)
                axs[1].legend()


                axs[2].set_title("Metrics Ch. Abdomial and SpO2")
                axs[2].set_xlabel("Threshold")
                axs[2].set_ylabel("Score")
                axs[2].legend()
                #axs[2].grid()
                axs[2].text(-0.1, 1.1, "B", size=20, weight='bold', transform=axs[2].transAxes)
                axs[2].set_xlim([0.4, 0.9]) 
                axs[2].set_ylim([0, 1]) 

                axs[2].set_yticks(np.arange(0, 1.1, 0.2))
                axs[2].set_xticks(np.arange(0.4, 0.91, 0.1))
                fig.tight_layout()
                #fig.tight_layout(rect=[0, 0.03, 1, 0.90])
                #fig.suptitle(data_set+" set. Window Size: "+str(ws)+", Trial: "+str(fold), fontsize=20)
                fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig2_v4.png")
                
   """             
    
         
        
        