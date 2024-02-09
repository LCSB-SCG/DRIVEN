import h5py
from tensorflow.keras.utils import to_categorical
import sys
import numpy as np
import glob
import os


if __name__ == '__main__':
    
    fold=int(sys.argv[1] ) 
    src=sys.argv[2]
    predict =  sys.argv[3] #ahi or sleep
    last_layer = int(sys.argv[4])



    if last_layer == 3 :
        s_w= src+"FEAT_DATA_"+str(fold)+"_"+predict+"/"
    elif last_layer == 1 :
        s_w= src+"FEAT_DATA_"+str(fold)+"_"+predict+"_2Out/"
    else:
        print("need last layer to be 1 or 3")
    
    
    if predict == "ahi":    
        variable_pred = "y"
    elif predict == "sleep":  
        variable_pred = "sleep_label"
    else:
        print("need prediction to be ahi or sleep")
            
    ## Training
    files=np.array( glob.glob(s_w+'TRAINING/*hdf5'))

    if layer_3:
        output =  s_w + "TRAINING_COMBINED.hdf5"
        f1 = h5py.File(output, mode='w') 
        f1.create_dataset("x", (1, 10240), np.float32,maxshape=(None,10240))
        f1.create_dataset(variable_pred, (1,2), np.float32,maxshape=(None,2)) 
    else:
        output =  s_w + "TRAINING_COMBINED_2Out.hdf5"
        f1 = h5py.File(output, mode='w') 
        f1.create_dataset("x", (1, 16), np.float32,maxshape=(None,16))
        f1.create_dataset(variable_pred, (1,2), np.float32,maxshape=(None,2)) 


    print(len(files))
    indx=0
    #current_file = files[2]
    for current_file in files:
        try:
            with h5py.File( current_file, 'r') as hf:
                y=hf[variable_pred][:,0]
                x= hf["x"][:,:]
            
            y = to_categorical(y[:])
            kk=list(range(indx, indx + len(y)))  
            f1["x"].resize(indx+ len(y),axis=0)
            f1[variable_pred].resize(indx+ len(y),axis=0)
            f1["x"][kk, ...] = x[:]
            f1[variable_pred][kk, ...] = y[:]
            indx += len(y)
            #os.remove(current_file)
        
        except Exception as e:
            print("ERROR: " + current_file + " Message: " + str(e))
            
    f1.close()

    ## Validation 
    files=np.array( glob.glob(s_w+'VALIDATION/*hdf5'))

    if layer_3:
        output =  s_w + "VALIDATION_COMBINED.hdf5"
        f1 = h5py.File(output, mode='w') 
        f1.create_dataset("x", (1, 10240), np.float32,maxshape=(None,10240))
        f1.create_dataset(variable_pred, (1,2), np.float32,maxshape=(None,2)) 
    else:
        output =  s_w + "VALIDATION_COMBINED_2Out.hdf5"
        f1 = h5py.File(output, mode='w') 
        f1.create_dataset("x", (1, 16), np.float32,maxshape=(None,16))
        f1.create_dataset(variable_pred, (1,2), np.float32,maxshape=(None,2)) 


    print(len(files))
    indx=0
    for current_file in files:
        try:
            with h5py.File( current_file, 'r') as hf:
                y=hf[variable_pred][:,0]
                x= hf["x"][:,:]
            
            y = to_categorical(y[:])
            kk=list(range(indx, indx + len(y)))  
            
            f1["x"].resize(indx+ len(y),axis=0)
            f1[variable_pred].resize(indx+ len(y),axis=0)

            f1["x"][kk, ...] = x[:]
            f1[variable_pred][kk, ...] = y[:]

            indx += len(y)
            #os.remove(current_file)
        except Exception as e:
            print("ERROR: " + current_file + " Message: " + str(e))
            
    f1.close()



    print(" ========================= FINISHED ===============================")

