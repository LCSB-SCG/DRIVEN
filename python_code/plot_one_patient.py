import h5py
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
import os
import math
import sys
import matplotlib.pylab as pylab
from matplotlib.patches import ConnectionPatch
import matplotlib

params = {'legend.fontsize': 'xx-large',
        #'figure.figsize': (15, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
    
    
    
patient = "mesa-sleep-2548.hdf5" # initial one

#patient = "mesa-sleep-2798.hdf5"



model = "Model2"
channels = "2_5"
fold = 2

file_pred = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/PREDICTIONS/TEST/15s/"+model+"_"+str(fold)+"_Ch_"+channels+"/"+patient
file_pred_sleep = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/PREDICTIONS/TEST/15s/sleep/"+model+"_"+str(fold)+"_Ch_"+channels+"_sleep/"+patient
file_raw = "/work/projects/heart_project/OSA_MW/all_files_ahi_sleep_complete/DATA/"+patient

with h5py.File(file_pred, 'r') as f:
    y_real = np.array(f["y_real"][:]) 
    y_pred = np.array(f["y_pred"][:])
    y_sleep = np.array(f["y_sleep"][:])


with h5py.File(file_pred_sleep, 'r') as f:
    sleep_pred = np.array(f["sleep_pred"][:])
    y_sleep2 = np.array(f["y_sleep"][:])


with h5py.File(file_raw , 'r') as f:
    y_real_fs = np.array(f["y"][:]) 
    y_sleep_fs = np.array(f["sleep_label"][:])
            

#sleep_pred_thr = (sleep_pred > 0.5)



       

# Take out the sleeping parts
#indx_awake = np.nonzero(y_sleep == 0)

indx_s = np.nonzero(y_sleep)
#indx_s_pred = np.nonzero(sleep_pred_thr)
#print(indx_s[0])
y_pred_sleep = np.array(y_pred)[indx_s]
y_real_sleep = np.array(y_real)[indx_s]
        
                  
#profusion_file = sys.argv[-1] #

roc_pr = True
conf = True
pred = True

#dir_figures = "/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/EVAL_MODELS/Examples/"+model+"_"+str(fold)+"_Ch_"+channels+"/"

#try:
#    os.mkdir(dir_figures)
#except:
#    pass
     
#dir_patient = dir_figures+(patient.split(".")[0]) 

#try:
#    os.mkdir(dir_patient)
#except:
#    pass



y_real_sec = np.concatenate( [y_real_fs[i*64] for i in range(int(len(y_real_fs)/64))] , axis=0 )[15:]
y_sleep_sec = (np.concatenate( [y_sleep_fs[i*64] for i in range(int(len(y_sleep_fs)/64))] , axis=0 )[15:]) > 0

y_real_stride_time = np.arange(0,39540,15)

y_pred_sleep = y_pred*y_sleep

y_pred_sleep_c= []

for c in y_pred_sleep:
    if c < 0.73:
        y_pred_sleep_c.append("green")
    else:
        y_pred_sleep_c.append("red")
    

fig, axs = plt.subplots(2, 1, figsize=(15,8))


axs[0].scatter(y_real_stride_time[400], y_pred_sleep[400], c="red" ,zorder=1, label = "Predicted apnea/hypopnea event")
axs[0].scatter(y_real_stride_time[400], y_pred_sleep[400], c="green" ,zorder=1, label = "Predicted normal")
axs[0].scatter(y_real_stride_time[400:2066], y_pred_sleep[400:2066], c=y_pred_sleep_c[400:2066] ,zorder=3)

axs[0].plot(np.arange(6000,30990), y_real_sec[6000:30990], 'tab:blue', zorder=2, label = "Ground truth")
axs[0].set_title("All night study")
axs[0].set_ylabel("AHI Event Probability")

axs[0].set_xlim([6000,30990])
axs[0].set_xlabel("Time [s]")
axs[0].legend(ncol=3, bbox_to_anchor=(0.87, 1.4))
#axs[0].text(-0.05, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)
   
rounded_rect = matplotlib.patches.FancyBboxPatch((23010,0), 3600, 1, boxstyle='Round, pad=0', ec="black", fill=False, lw=3)
axs[0].add_patch(rounded_rect) 


axs[1].scatter(y_real_stride_time[1534:1774], y_pred_sleep[1534:1774], c=y_pred_sleep_c[1534:1774],zorder=2)
#plt.plot(y_sleep_sec, 'g', zorder=1)
#axs[1].plot(np.arange(15000,18600), y_real_sec[15000:18600], 'b', zorder=1, label = "True AHI Events")
axs[1].plot(np.arange(23010,26610), y_real_sec[23010:26610], 'tab:blue', zorder=1, label = "True AHI Events")
axs[1].fill_between(np.arange(23010,26610), y_real_sec[23010:26610], alpha=0.8)
#axs[1].legend()
axs[1].set_title("Zoom: 1 hour")
axs[1].set_ylabel("AHI Event Probability")
axs[1].set_xlabel("Time [s]")
axs[1].set_xlim([23010,26610])


#axs[1].text(-0.05, 1.1, "B", size=20, weight='bold', transform=axs[1].transAxes)

#circle = matplotlib.patches.Ellipse((25440,0.5),900, 1, ec="black", fill=False, lw=3)
#axs[1].add_patch(circle)

ax3 = fig.add_subplot(111)

ax3.plot([6.8,0],[6.05,4.15],'k--')
ax3.plot([8.25,10],[6.05,4.15],'k--')


ax3.set_xlim([0,10])
ax3.set_ylim([0,10])
ax3.axis("off")

fig.tight_layout()
#plt.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig4_v6.pdf", format="pdf", bbox_inches="tight")
plt.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig4_v7.png")




# blue: ground truth 
# Red - green: Predicted Normal - Predicted AHI 
"""    
fig, axs = plt.subplots(3, 1, figsize=(15,10))
#plt.scatter(y_real_stride_time, y_real, c='b',zorder=2, la)
#axs[0].scatter(y_real_stride_time[400], y_pred_sleep[400], c="red" ,zorder=1, label = "AHI Event Probability > Threshold")
#axs[0].scatter(y_real_stride_time[400], y_pred_sleep[400], c="green" ,zorder=1, label = "AHI Event Probability < Threshold")
axs[0].scatter(y_real_stride_time[400:2066], y_pred_sleep[400:2066], c=y_pred_sleep_c[400:2066] ,zorder=3)
#plt.plot(y_sleep_sec, 'g', zorder=1)
axs[0].plot(np.arange(6000,30990), y_real_sec[6000:30990], 'tab:blue', zorder=2, label = "True AHI Events")
axs[0].set_title("All night study")
axs[0].set_ylabel("Probability")
#axs[0].set_ylim([0,1.])
axs[0].set_xlim([6000,30990])
axs[0].set_xlabel("Time [s]")
axs[0].text(-0.05, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)

#circle = matplotlib.patches.Ellipse((24810,0.5),3600, 1, ec="black", fill=False, lw=3)
#axs[0].add_patch(circle)

rounded_rect = matplotlib.patches.FancyBboxPatch((23010,0), 3600, 1, boxstyle='Round, pad=0', ec="black", fill=False, lw=3)
axs[0].add_patch(rounded_rect)


axs[1].scatter(y_real_stride_time[1534:1774], y_pred_sleep[1534:1774], c=y_pred_sleep_c[1534:1774],zorder=2)
#plt.plot(y_sleep_sec, 'g', zorder=1)
#axs[1].plot(np.arange(15000,18600), y_real_sec[15000:18600], 'b', zorder=1, label = "True AHI Events")
axs[1].plot(np.arange(23010,26610), y_real_sec[23010:26610], 'tab:blue', zorder=1, label = "True AHI Events")
#axs[1].legend()
axs[1].set_title("Zoom: 1 hour")
axs[1].set_ylabel("Probability")
axs[1].set_xlabel("Time [s]")
axs[1].set_xlim([23010,26610])
axs[1].text(-0.05, 1.1, "B", size=20, weight='bold', transform=axs[1].transAxes)

#circle = matplotlib.patches.Ellipse((25440,0.5),900, 1, ec="black", fill=False, lw=3)
#axs[1].add_patch(circle)

rounded_rect = matplotlib.patches.FancyBboxPatch((24990,0), 900, 1, boxstyle='Round, pad=0', ec="black", fill=False, lw=3)

axs[1].add_patch(rounded_rect)

"""
fig, axs = plt.subplots(1, 1, figsize=(15,4))

axs.scatter(y_real_stride_time[1666:1726], y_pred_sleep[1666:1726], c=y_pred_sleep_c[1666:1726],zorder=2)
#plt.plot(y_sleep_sec, 'g', zorder=1)
axs.legend()
axs.scatter(y_real_stride_time[1666], y_pred_sleep[1666], c="red" ,zorder=1, label = "Predicted apnea/hypopnea event")
axs.scatter(y_real_stride_time[1666], y_pred_sleep[1666], c="green" ,zorder=1, label = "Predicted normal")
axs.plot(np.arange(24990,25890), y_real_sec[24990:25890], 'tab:blue', zorder=1, label = "Ground truth")


axs.set_title("Zoom: 15 minutes")
axs.set_ylabel("Probability")
axs.set_xlabel("Time [s]")
axs.set_xlim([24990,25890])
#axs.text(-0.05, 1.1, "C", size=20, weight='bold', transform=axs[2].transAxes)
axs.legend(ncol=3, bbox_to_anchor=(.9, 1.5))

#xy0 = (23010,0)
#xy1 = (10,0)

#ax3 = fig.add_subplot(111)
#ax3.plot([7,0],[0.7,0.5],'k--')
#ax3.plot([6.8,0],[8.8,6.22],'k--')
#ax3.plot([8.25,10],[8.8,6.22],'k--')

#ax3.plot([5.48,0],[5,2.5],'k--')
#ax3.plot([8,10],[5,2.5],'k--')
#ax3.plot([6.6,0],[8.8,3.8],'k--')
#ax3.plot([6.6,0],[8.8,3.8],'k--')
#ax3.set_xlim([0,10])
#ax3.set_ylim([0,10])
#ax3.axis("off")

#con = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
#                      axesA=axs[0], axesB=axs[1], color="red")

#axs[1].add_artist(con)

#line = matplotlib.lines.Line2D(xy0 ,xy1, transform=fig.transFigure)
#fig.lines = line



fig.tight_layout()
plt.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig4_sup.pdf", format="pdf", bbox_inches="tight")
#fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig4_v2.png")

"""


### ROC + PR ###########################################################################################################################











if roc_pr:
    fpr, tpr, thresholds = metrics.roc_curve(y_real_sleep, y_pred_sleep)
    score = metrics.roc_auc_score(y_real_sleep, y_pred_sleep)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    axes[0].plot(fpr,tpr)
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_title(f'ROC curve, AUROC: {score:.2f}')
    precision, recall, thresholds = metrics.precision_recall_curve(y_real_sleep, y_pred_sleep)
    score = metrics.average_precision_score(y_real_sleep, y_pred_sleep)
    axes[1].plot(recall, precision)
    axes[1].set_ylabel('Precision')
    axes[1].set_xlabel('Recall')
    axes[1].set_title(f'PR curve, AUPRC: {score:.2f}')
    plt.savefig(dir_patient+'/ROC_PR.png')

        ### Confusion matrix #####################################################################################################################

if conf:
    ncols = 1
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*ncols))
    thr = 0.73
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_real_sleep, y_pred_sleep > thr, ax=axes)
    axes.title.set_text("Threshold: "+ str(thr))
    plt.tight_layout()  
    plt.savefig(dir_patient+'/Conf.png')
    
    


### PSG pred ###########################################################################################################################
if pred:

    time_seconds = np.arange(time[0], time[-1]+1)
    osa_pred = np.zeros(len(time_seconds))
    pass_count = np.zeros(len(time_seconds))

    for i in range(len(time)):
        sec = int(time[i])

        try:
            osa_pred[sec-40:sec] += y_pred[i]
            pass_count[sec-40:sec] +=1
        except:
            osa_pred[sec-40:-1] += y_pred[i]
            pass_count[sec-40:-1] +=1
        
        
    osa_pred = osa_pred/pass_count
    osa_pred = osa_pred/(max(osa_pred))
    print(max(osa_pred))


    osa_real = np.zeros(len(time_seconds))

    
    tree = ET.parse(profusion_file)
    events = tree.findall('ScoredEvents/ScoredEvent')
    events_df = pd.DataFrame(columns = ['Name', 'Start', 'Duration'])

    for event in events:
        name = event.findall('Name')[0].text
        if name in ["Unsure", "Central Apnea", "Hypopnea", "Obstructive Apnea"]:
            start = int(float(event.findall('Start')[0].text))
            duration = int(float(event.findall('Duration')[0].text))
            osa_real[start-40:start-40+duration] = 1

    
    time_end = time[-1] 
    n_windows = math.ceil(time_end/500)
    
    ncols = 5
    nrows = math.ceil(n_windows/ncols)
    
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*ncols))

    for i in range(nrows*ncols):
        col = math.floor(i/nrows)
        row = i - col * nrows
        axes[row, col].plot(time_seconds[i*500:(i+1)*500], osa_pred[i*500:(i+1)*500], label="Predicted Probability")
        axes[row, col].plot(time_seconds[i*500:(i+1)*500], osa_real[i*500:(i+1)*500], label="Real")
        axes[row, col].set_ylim([0, 1.1])

    axes[0, 0].legend()
    plt.tight_layout()  
    fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig4_v2.png")
    
"""
