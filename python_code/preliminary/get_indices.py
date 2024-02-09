import h5py
import numpy as np
import json


def find_and_select_indices_sleep(labels):
    twos_indices = [i for i, label in enumerate(labels) if label == 2] #sleep - ahi
    ones_indices = [i for i, label in enumerate(labels) if label == 1] #sleep - noahi
    zeros_indices = [i for i, label in enumerate(labels) if label == 0] #awake
    num_twos = len(twos_indices)
    num_ones = len(ones_indices)
    num_zeros = len(zeros_indices)
    return num_zeros, num_ones, num_twos, zeros_indices, ones_indices, twos_indices



src  = "/work/projects/heart_project/OSA_MW/all_10_ws_10648_files_ahi_sleep_newSF/"


parts = ["VAL", "TEST"]
splits = ["1", "2", "3"]

part = "TRAIN"
split = "1"

stride = "5"

for part in parts:
    for split in splits:
        files = src+"SPLIT/"+part+"_split"+split+".txt"
        with open(files, 'r') as f:
            files = f.readlines()

        f = np.array([file[:-1] for file in files])

        current_file = f[1]

        labels_dic = {}
        for current_file in f:
            current_file_name = src+"DATA_"+stride+"s/"+current_file
            with h5py.File(current_file_name, 'r') as hf:
                y=hf["label_y_s"][:,0]
            
            labels = (y).astype(int)
            n_zeros, n_ones, n_twos, zeros, ones, twos = find_and_select_indices_sleep(labels) 
            n_select = min(n_zeros, n_ones, n_twos, 64)
            labels_dic[current_file] = {"n_awake": n_zeros, "n_sleep": n_ones, "n_ahi": n_twos, "n_select": n_select, "awake": zeros, "sleep": ones, "ahi": twos}
            #print(current_file)
            print(n_zeros, n_ones, n_twos)
            

        json.dump(labels_dic, open(src+"SPLIT/dict_"+part+"_split"+split+"_"+stride+"+s.json", 'w' ) )





