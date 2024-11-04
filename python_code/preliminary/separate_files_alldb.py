"""
DRIVEN is © 2024, University of Luxembourg

DRIVEN is published and distributed under the Academic Software License v1.0 (ASL). 

DRIVEN is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the ASL for more details. 

You should have received a copy of the ASL along with this program; if not, write to LCSB-innovation@uni.lu.  It is also published at https://turbogap.fi/wiki/index.php/Academic_Software_Licence.

You may contact the original licensor at LCSB-innovation@uni.lu.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Dec 7 2023

@author: MRGR
"""
import glob
import random
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os
import sys

np.random.seed(42)
random.seed(42)


src = "/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep/" #sys.argv[1] 


files=np.array(glob.glob(src+"DATA/"+"*.hdf5", recursive=False))
files = [f.split("/")[-1] for f in files]
print(len(files))
random.shuffle(files)

ids_mesa = [f[11:15] for f in files if "mesa" in f]
ids_shhs1 = [f[6:12] for f in files if "shhs1" in f]
ids_shhs2 = [f[6:12] for f in files if "shhs2" in f]
ids_mros1 = [f[12:18] for f in files if "mros-visit1" in f]
ids_mros2 = [f[12:18] for f in files if "mros-visit2" in f]


###### SPLIT DATA

######
# Split 1: Train/Val shhs and mros, Test: mesa
######

# Shhs1
train_shhs1, val_shhs1 = train_test_split(ids_shhs1, test_size=0.2, random_state=0)
# Shhs2 -  Move the ones repeated
train_shhs2 = np.intersect1d(train_shhs1, ids_shhs2)
val_shhs2 = np.intersect1d(val_shhs1, ids_shhs2)
# Split 80/20 the rest of the files
rest_shhs2 = np.setxor1d(np.setxor1d(ids_shhs2, train_shhs2), val_shhs2)
train_shhs2_r, val_shhs2_r = train_test_split(rest_shhs2, test_size=0.2, random_state=0)

# Mros1
train_mros1, val_mros1 = train_test_split(ids_mros1, test_size=0.2, random_state=0)
# Mros2 -  Move the ones repeated
train_mros2 = np.intersect1d(train_mros1, ids_mros2)
val_mros2 = np.intersect1d(val_mros1, ids_mros2)
# Split 80/20 the rest of the files
rest_mros2 = np.setxor1d(np.setxor1d(ids_mros2, train_mros2), val_mros2)
train_mros2_r, val_mros2_r = train_test_split(rest_mros2, test_size=0.2, random_state=0)


# Get the final list:
train_split1 = ["shhs1-"+str(f)+".hdf5" for f in train_shhs1]
train_split1.extend(["shhs2-"+str(f)+".hdf5" for f in train_shhs2])
train_split1.extend(["shhs2-"+str(f)+".hdf5" for f in train_shhs2_r])
train_split1.extend(["mros-visit1-"+str(f)+".hdf5" for f in train_mros1])
train_split1.extend(["mros-visit2-"+str(f)+".hdf5" for f in train_mros2])
train_split1.extend(["mros-visit2-"+str(f)+".hdf5" for f in train_mros2_r])

val_split1 = ["shhs1-"+str(f)+".hdf5" for f in val_shhs1]
val_split1.extend(["shhs2-"+str(f)+".hdf5" for f in val_shhs2])
val_split1.extend(["shhs2-"+str(f)+".hdf5" for f in val_shhs2_r])
val_split1.extend(["mros-visit1-"+str(f)+".hdf5" for f in val_mros1])
val_split1.extend(["mros-visit2-"+str(f)+".hdf5" for f in val_mros2])
val_split1.extend(["mros-visit2-"+str(f)+".hdf5" for f in val_mros2_r])


print(len(val_split1)/len(train_split1))

# Test files (mesa)
test_split1 = ["mesa-sleep-"+str(f)+".hdf5" for f in ids_mesa]


# Write files 
file_training=src +'SPLIT/TRAIN_split1.txt'
np.random.shuffle(train_split1)
with open(file_training, "w") as f:
    f.writelines(line + '\n' for line in train_split1)


file_val=src +'SPLIT/VAL_split1.txt'
np.random.shuffle(val_split1)
with open(file_val, "w") as f:
    f.writelines(line + '\n' for line in val_split1)


file_test=src +'SPLIT/TEST_split1.txt'
np.random.shuffle(test_split1)
with open(file_test, "w") as f:
    f.writelines(line + '\n' for line in test_split1)



######
# Split 2: Train/Val shhs Test: mros mesa
######

# Shhs1
train_shhs1, val_shhs1 = train_test_split(ids_shhs1, test_size=0.2, random_state=0)
# Shhs2 -  Move the ones repeated
train_shhs2 = np.intersect1d(train_shhs1, ids_shhs2)
val_shhs2 = np.intersect1d(val_shhs1, ids_shhs2)
# Split 80/20 the rest of the files
rest_shhs2 = np.setxor1d(np.setxor1d(ids_shhs2, train_shhs2), val_shhs2)
train_shhs2_r, val_shhs2_r = train_test_split(rest_shhs2, test_size=0.2, random_state=0)


# Get the final list:
train_split2 = ["shhs1-"+str(f)+".hdf5" for f in train_shhs1]
train_split2.extend(["shhs2-"+str(f)+".hdf5" for f in train_shhs2])
train_split2.extend(["shhs2-"+str(f)+".hdf5" for f in train_shhs2_r])

val_split2 = ["shhs1-"+str(f)+".hdf5" for f in val_shhs1]
val_split2.extend(["shhs2-"+str(f)+".hdf5" for f in val_shhs2])
val_split2.extend(["shhs2-"+str(f)+".hdf5" for f in val_shhs2_r])


print(len(val_split2)/len(train_split2))

# Test files (mesa)
test_split2 = ["mesa-sleep-"+str(f)+".hdf5" for f in ids_mesa]
test_split2.extend(["mros-visit1-"+str(f)+".hdf5" for f in ids_mros1])
test_split2.extend(["mros-visit2-"+str(f)+".hdf5" for f in ids_mros2])

file_training=src +'SPLIT/TRAIN_split2.txt'
np.random.shuffle(train_split2)
with open(file_training, "w") as f:
    f.writelines(line + '\n' for line in train_split2)


file_val=src +'SPLIT/VAL_split2.txt'
np.random.shuffle(val_split2)
with open(file_val, "w") as f:
    f.writelines(line + '\n' for line in val_split2)


file_test=src +'SPLIT/TEST_split2.txt'
np.random.shuffle(test_split2)
with open(file_test, "w") as f:
    f.writelines(line + '\n' for line in test_split2)







######
# Split 3: Train/Val/Test shhs and mros and mesa
######

# Shhs1
train_shhs1, test_shhs1 = train_test_split(ids_shhs1, test_size=0.2, random_state=0)
train_shhs1, val_shhs1 = train_test_split(train_shhs1, test_size=0.25, random_state=0)
# Shhs2 -  Move the ones repeated
train_shhs2 = np.intersect1d(train_shhs1, ids_shhs2)
val_shhs2 = np.intersect1d(val_shhs1, ids_shhs2)
test_shhs2 = np.intersect1d(test_shhs1, ids_shhs2)
# Split 80/20 the rest of the files
rest_shhs2 = np.setxor1d(np.setxor1d(np.setxor1d(ids_shhs2, train_shhs2), val_shhs2),test_shhs2)
train_shhs2_r, test_shhs2_r = train_test_split(rest_shhs2, test_size=0.2, random_state=0)
train_shhs2_r, val_shhs2_r = train_test_split(train_shhs2, test_size=0.2, random_state=0)


# Mros1
train_mros1, test_mros1 = train_test_split(ids_mros1, test_size=0.2, random_state=0)
train_mros1, val_mros1 = train_test_split(train_mros1, test_size=0.25, random_state=0)
# mros2 -  Move the ones repeated
train_mros2 = np.intersect1d(train_mros1, ids_mros2)
val_mros2 = np.intersect1d(val_mros1, ids_mros2)
test_mros2 = np.intersect1d(test_mros1, ids_mros2)
# Split 80/20 the rest of the files
rest_mros2 = np.setxor1d(np.setxor1d(np.setxor1d(ids_mros2, train_mros2), val_mros2),test_mros2)
train_mros2_r, test_mros2_r = train_test_split(rest_mros2, test_size=0.2, random_state=0)
train_mros2_r, val_mros2_r = train_test_split(train_mros2, test_size=0.2, random_state=0)

# Mesa
train_mesa, test_mesa = train_test_split(ids_mesa, test_size=0.2, random_state=0)
train_mesa, val_mesa = train_test_split(train_mesa, test_size=0.25, random_state=0)


# Get the final list:
train_split3 = ["shhs1-"+str(f)+".hdf5" for f in train_shhs1]
train_split3.extend(["shhs2-"+str(f)+".hdf5" for f in train_shhs2])
train_split3.extend(["shhs2-"+str(f)+".hdf5" for f in train_shhs2_r])
train_split3.extend(["mros-visit1-"+str(f)+".hdf5" for f in train_mros1])
train_split3.extend(["mros-visit2-"+str(f)+".hdf5" for f in train_mros2])
train_split3.extend(["mros-visit2-"+str(f)+".hdf5" for f in train_mros2_r])
train_split3.extend(["mesa-sleep-"+str(f)+".hdf5" for f in train_mesa])

val_split3 = ["shhs1-"+str(f)+".hdf5" for f in val_shhs1]
val_split3.extend(["shhs2-"+str(f)+".hdf5" for f in val_shhs2])
val_split3.extend(["shhs2-"+str(f)+".hdf5" for f in val_shhs2_r])
val_split3.extend(["mros-visit1-"+str(f)+".hdf5" for f in val_mros1])
val_split3.extend(["mros-visit2-"+str(f)+".hdf5" for f in val_mros2])
val_split3.extend(["mros-visit2-"+str(f)+".hdf5" for f in val_mros2_r])
val_split3.extend(["mesa-sleep-"+str(f)+".hdf5" for f in val_mesa])


print(len(val_split3)/len(train_split3))

test_split3 = ["shhs1-"+str(f)+".hdf5" for f in test_shhs1]
test_split3.extend(["shhs2-"+str(f)+".hdf5" for f in test_shhs2])
test_split3.extend(["shhs2-"+str(f)+".hdf5" for f in test_shhs2_r])
test_split3.extend(["mros-visit1-"+str(f)+".hdf5" for f in test_mros1])
test_split3.extend(["mros-visit2-"+str(f)+".hdf5" for f in test_mros2])
test_split3.extend(["mros-visit2-"+str(f)+".hdf5" for f in test_mros2_r])
test_split3.extend(["mesa-sleep-"+str(f)+".hdf5" for f in test_mesa])

print(len(test_split3)/len(val_split3))


# Write files 
file_training=src +'SPLIT/TRAIN_split3.txt'
np.random.shuffle(train_split3)
with open(file_training, "w") as f:
    f.writelines(line + '\n' for line in train_split3)


file_val=src +'SPLIT/VAL_split3.txt'
np.random.shuffle(val_split3)
with open(file_val, "w") as f:
    f.writelines(line + '\n' for line in val_split3)


file_test=src +'SPLIT/TEST_split3.txt'
np.random.shuffle(test_split3)
with open(file_test, "w") as f:
    f.writelines(line + '\n' for line in test_split3)
















