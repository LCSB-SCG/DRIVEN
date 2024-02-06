import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import matplotlib.pylab as pylab
from sklearn import metrics

import tikzplotlib


import seaborn as sns


palette = sns.palettes.color_palette("pastel", n_colors=4)
sns.set_palette(palette)
sns.set_style(
    style="white",
)


params = {'legend.fontsize': 'large', 
        #'figure.figsize': (15, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
    

channels_try = [["2", "3", "5"]]

comb_2 = list(map(list,list(combinations(channels_try[0], 2) )))
comb_1 = list(map(list,list(combinations(channels_try[0], 1) )))

channels_try.extend(comb_2)
channels_try.extend(comb_1)
    
for channels in channels_try:
    
    channels = "_".join(channels)
    patients_best = pd.read_csv("/work/projects/heart_project/OSA_MW/all_30_ws_10648_files_ahi_sleep_newSF/EVAL_MODELS/MOVING_AVERAGE/"+channels+"/best_ahis/Model2_f2_ch_"+channels+"__best_patient_PRscores_TEST.csv")

    #patients_best = patients_best[patients_best.patients.str.contains('mros')]


    fig = plt.figure(figsize=(5, 5))


    colors = patients_best.ahi_real_classes
    #print(colors)
    colors_scale = [palette[0], palette[1], palette[2], palette[3]]
    colors_points = [colors_scale[int(c)] for c in colors]


    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=4)

    plt.scatter(0,0, color=palette[0], label = "Healthy")
    plt.scatter(0,0, color=palette[1], label = "Mild")
    plt.scatter(0,0, color=palette[2], label = "Moderate")
    plt.scatter(0,0, color=palette[3], label = "Severe")
    plt.scatter(patients_best.ahi_reals, patients_best.ahi_sleep_scores, c=colors_points)
    plt.plot([0,110],[5,5], 'k--')
    plt.plot([0,110],[15,15], 'k--')
    plt.plot([0,110],[30,30], 'k--')
    plt.plot([0,110],[0,110], 'k-')


    plt.text(75, 7, "Pred. AHI = 5", fontsize=12)
    plt.text(75, 17, "Pred. AHI = 15", fontsize=12)
    plt.text(75, 32, "Pred. AHI = 30", fontsize=12)
    plt.xlim([0, 110]) 
    plt.ylim([0, 110]) 
    plt.legend() 
    plt.xlabel("True AHI")
    plt.ylabel("Predicted AHI")
    #axs[0].set_title("Abdo. and SPO2 AHI predictions")
    #plt.text(0, 1.1, "Abdo. and SPO2 AHI Pred.", size=18, transform=axs[0].transAxes)
    #plt.text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)


    #sns.scatterplot(data=patients_best, x="ahi_reals", y="ahi_sleep_scores", hue="ahi_real_classes")
    fig.tight_layout()
    #fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig3_A_v6.png")
    fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig_sup_scatter_"+channels+".pdf", format="pdf", bbox_inches="tight")


    fig, ax = plt.subplots(figsize=(5, 5))

    cm = metrics.confusion_matrix(patients_best.ahi_sleep_classes, patients_best.ahi_real_classes, labels=[0, 1, 2, 3])

    ax.matshow(cm, cmap=plt.cm.Blues)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if  cm[i, j] > thresh else "black") 


    labels = ['Healthy', 'Mild', 'Moderate', 'Severe']

    f1_score_plot = metrics.f1_score(patients_best.ahi_real_classes, patients_best.ahi_sleep_classes, average="micro")
            
    #ax.set_title(str(f1_score_plot))
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.ylabel('Predicted AHI Class')
    plt.xlabel('True AHI Class')
    #ax.text(-0.1, 1.1, "B", size=20, weight='bold', transform=ax.transAxes)
    #ax.text(0, 1.1, "Abdo. and SPO2 Conf. Matrix", size=18, transform=ax.transAxes)


    fig.tight_layout()
    #fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig3_B_v5.png")
    fig.savefig("/scratch/users/mretamales/OSA_scratch/new_pipeline/fig_sup_confusion_"+channels+".pdf", format="pdf", bbox_inches="tight")

"""

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

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
#axs[0].set_title("Abdo. and SPO2 AHI predictions")
axs[0].text(0, 1.1, "Abdo. and SPO2 AHI Pred.", size=18, transform=axs[0].transAxes)
axs[0].text(-0.1, 1.1, "A", size=20, weight='bold', transform=axs[0].transAxes)

cm = metrics.confusion_matrix(patients_best.ahi_real_classes, patients_best.ahi_sleep_classes, labels=[0, 1, 2, 3])

axs[1].matshow(cm, cmap=plt.cm.Blues)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axs[1].text(j, i, format(cm[i, j]),
                ha="center", va="center",
                color="white" if  cm[i, j] > thresh else "black") 


labels = ['Healthy', 'Mild', 'Moderate', 'Severe']

f1_score_plot = metrics.f1_score(patients_best.ahi_real_classes, patients_best.ahi_sleep_classes, average="micro")
        
print(f1_score_plot)
axs[1].set_title(str(f1_score_plot))
axs[1].set_xticklabels([''] + labels)
axs[1].set_yticklabels([''] + labels)
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('True')
axs[1].text(-0.1, 1.1, "B", size=20, weight='bold', transform=axs[1].transAxes)
axs[1].text(0, 1.1, "Abdo. and SPO2 Conf. Matrix", size=18, transform=axs[1].transAxes)



sensors = ["Abdo., Thor., SPO2", "Abdo., SPO2", "Thor., SPO2", "SPO2", "Abdo.", "Thor."]
f1s_ahis = [0.76, 0.76, 0.75, 0.74, 0.58, 0.54]

table1 = pd.DataFrame.from_dict({"Sensors": sensors, "F1 AHI": f1s_ahis})


tableplot = axs[2].table(cellText=table1.values, colLabels=table1.columns, loc='center', bbox=[0,0,1,1])

tableplot.auto_set_font_size(False)
tableplot.set_fontsize(12)
cellDict = tableplot.get_celld()
cellDict[0,0].set_text_props(weight='bold')
cellDict[0,0].set_fontsize(16)
cellDict[0,1].set_text_props(weight='bold')
cellDict[0,1].set_fontsize(16)


axs[2].axis('off')
axs[2].axis('tight')
axs[2].text(-0.1, 1.1, "C", size=20, weight='bold', transform=axs[2].transAxes)

fig.tight_layout()
#fig.tight_layout(rect=[0, 0.03, 1, 0.90])
#fig.suptitle("TEST set. Window Size: 30, Trial: 2, Channels: Abdominal and SPO2", fontsize=20)


"""


        








