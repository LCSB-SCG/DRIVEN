# import tensorflow as tf
import optuna
import h5py
import random
import sys 
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tensorflow.keras.metrics import AUC
import gc
pr_metric = AUC(curve='PR', num_thresholds=200) # The higher the threshold value, the more accurate it is calculated.
import time
from itertools import combinations


# from numba import jit
def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print(" ================================ Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.==========================================")
    else:
        print("Toc: start time not set")
        


def objective(trial):
    # Check on 10 % of data for h-parameter optimization
    X_sample, _, y_sample, _ = train_test_split(x_train, y_train, 
                                                train_size=0.1, random_state=42)
    # Splitting the data
    X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X_sample,
                                                                y_sample,
                                                                test_size=0.3,
                                                                random_state=42)
    # Defining the parameters space
    param = {
        'objective': 'binary',  # You can change this for multiclass problem
        'verbose': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 2, 200),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4,
                                                    1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4,
                                                    1.0),
        'subsample' :trial.suggest_uniform("subsample", 0.5, 1.0),
        'subsample_freq' : trial.suggest_int("subsample_freq", 0, 10),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'colsample_bytree': trial.suggest_uniform("colsample_bytree", 0.5, 1.0),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    # Training the model
    model = lgb.train(param,
                        lgb.Dataset(X_train2, y_train2),
                        valid_sets=[lgb.Dataset(x_val, y_val)],
                        early_stopping_rounds=35,
                        verbose_eval=False)
    # training
    preds = model.predict(X_valid2)
    auc_v = roc_auc_score(y_valid2, preds)
    preds = model.predict(X_train2)
    auc_t = roc_auc_score(y_train2, preds)
    result= 2 * auc_v - auc_t
    #print(" ============ Result  " + str(result) +"                                ")
    gc.collect()
    return result
                
                
                        
np.random.seed(42)
random.seed(42)
#  LABELS
#  {'obstapnea','oa','obsthypopnea'};%           1
#  {'centralapnea','ca','centralhypopnea'}; %    2
#  {'hypopnea',}; %                              3
#  {'mixedapnea','ma'};%                         4
#  {'apnea'}; %                                  5
#  {'snorea'}; %                                 6 
 
#  CHANNELS
# channel{1}={'EKG','ECGL','ECG','ECGLECGR'};%                               ECG
# channel{2}={'ABDO','ABDOMINAL','ABD','ABDORES','ABDOMEN'}; %               ABDOMINAL
# channel{3}={'THOR','THORACIC','CHEST','THORRES','THORAX'}; %               CHEST
# channel{4}={'FLOW','AUX','CANNULAFLOW','NASALFLOW','NEWAIR','AIRFLOW'}; %        NASSAL
# channel{5}={'SPO2','SAO2','SPO2'}; %         
# channel{6}={'RRI}; %     

if __name__ == '__main__':
## INPUTS
    # # Parameters
    fold = int(sys.argv[1])
    src = sys.argv[2] # src="/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/"
    predict =  sys.argv[3] #ahi or sleep
    last_layer = int(sys.argv[4]) #-3 or -1
    
    
    if last_layer == 3:
        dnn_features=1280
        s_w= src+"FEATURES/FEAT_DATA_"+str(fold)+"_"+predict+"/"
        train_f =  s_w + "TRAINING_COMBINED.hdf5"
        val_f= s_w + "VALIDATION_COMBINED.hdf5"
        src_o= src +  "WEIGHTS_2out_"+predict+"_"+str(fold)+"_all/"  
        
        
    elif last_layer == 1:
        dnn_features=2
        s_w= src+"FEATURES/FEAT_DATA_"+str(fold)+"_"+predict+"_2Out/"
        train_f =  s_w + "TRAINING_COMBINED_2Out.hdf5"
        val_f= s_w + "VALIDATION_COMBINED_2Out.hdf5"
        src_o= src +  "WEIGHTS_2out_"+predict+"_"+str(fold)+"_all/"  

    try:
        os.mkdir(src_o)    
    except:
        pass
    
    sens=[[2, 3, 4, 5]] #ABDO - THOR - FLOW - SPO2 -  - SPO2+delays...
    comb_3 = list(map(list,list(combinations(sens[0], 3) )))
    comb_2 = list(map(list,list(combinations(sens[0], 2) )))
    comb_1 = list(map(list,list(combinations(sens[0], 1) )))


    sens.extend(comb_3)
    sens.extend(comb_2)
    sens.extend(comb_1)
    
    current_sens = sens[0]
    for current_sens in sens:
        
        sen_names = "_".join(map(str,current_sens))
        model_name = src_o+"LGB_OUT_"+predict+"_"+str(sen_names)+".txt"
        if not (os.path.exists( model_name )):
            os.mknod(model_name) 
            print(sen_names)   
            
            if last_layer == 3:
                results_file = src_o+"LGBMCOEF_f_"+predict+"_"+str(sen_names)+'.txt' 
                model_file = src_o+"LGBD_MODEL_f"+predict+"_"+str(sen_names)+'.pkl' 
            
            elif last_layer == 1:
                results_file = src_o+"LGBMCOEF_f_"+predict+"_"+str(sen_names)+'_m.txt' 
                model_file = src_o+"LGBD_MODEL_f"+predict+"_"+str(sen_names)+'_m.pkl' 
            
            inx_feat=np.array([] ,dtype=int)  
            
            for l in current_sens:
                inx_feat= np.append(inx_feat, list(range((l-2)*dnn_features, (l-1)*dnn_features)))  
                
            if 5 in current_sens: #spo2
                for spo in [4, 5, 6, 7]:
                    inx_feat= np.append(inx_feat, list(range(spo*dnn_features, (spo+1)*dnn_features)))  
                    
    
            ## TRAINING
            data=train_f
            with h5py.File(data, 'r') as hf: 
                tic()
                y = hf["y"][:,0]
            
            unique_values = list(set(y))
            print(unique_values)

            print( "TRAINING SIZE: " + str( len(y) ) )

        
            y_train = pd.Series(y) 
                
                
            ## READ DATA IN CHUNKS
            with h5py.File(data, 'r') as f:
                total_size = f['x'].shape[0]
                x = []
                chunk_size=1000
                # Read and append the data in chunks
                for i in range(0, total_size, chunk_size):
                    # Calculate the end index for the chunk
                    end = min(i + chunk_size, total_size)
                    # print(i)
                    # Read the chunk and append it to the list
                    chunk = f['x'][i:end,inx_feat]
                    x.append(chunk)
            x = np.concatenate(x, axis=0)
            x_train = pd.DataFrame(x)
                 
            del y,x
            toc()
            
            
            ## VALIDATION
            data=val_f
            with h5py.File(data, 'r') as hf: 
                tic()
                y = hf["y"][:,0]
                print( "VALIDATION SIZE: " + str( len(y) ) )
    
                
            y_val= pd.Series(y) 
                
                
            ## READ DATA IN CHUNKS
            with h5py.File(data, 'r') as f:
                total_size = f['x'].shape[0]
                x = []
                chunk_size=100000
                # Read and append the data in chunks
                for i in range(0, total_size, chunk_size):
                    # Calculate the end index for the chunk
                    end = min(i + chunk_size, total_size)
                    # Read the chunk and append it to the list
                    chunk = f['x'][i:end,inx_feat]
                    x.append(chunk)
                    
            x = np.concatenate(x, axis=0)
            x_val= pd.DataFrame(x )

            del y,x
            toc()
            
            # ## ============ LGBMC CLASSIFIER ============
    
            tic()      
            ## OPTIMIZE PARAMETERS
            # Running the optimization
            # print("Optuna optimization")
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, gc_after_trial=True)
            toc()
            
            ## TRAIN THE MODEL
            tic()
            model = lgb.LGBMClassifier(metric=None)
            
            ## Best Parameters
            print(study.best_trial.params)
            
            model.set_params(**study.best_trial.params)
            model.fit(x_train, y_train , eval_set=[(x_val, y_val)] , verbose=1 , early_stopping_rounds=35)
            toc()
            # model.fit(x_train, y_train)
            
            
            #PREDICTIONS
            y_pred=model.predict(x_val)
            accuracy =accuracy_score(y_val,y_pred)
            recall=recall_score(y_val,y_pred )
            precision=precision_score(y_val,y_pred )
            confusion=confusion_matrix(y_val,y_pred) 
            
            y_proba = model.predict_proba(x_val)[:, 1]
            
            # Compute AUC-ROC
            auc_roc = roc_auc_score(y_val, y_proba)
            print("AUC-ROC:", auc_roc)
            
            # Compute AUC-PR and precision-recall curve
            precision, recall, _ = precision_recall_curve(y_val, y_proba)
            auc_pr = average_precision_score(y_val, y_proba)
            print("AUC-PR:", auc_pr)
        
            ## FEATURES RELEVANCE:
            relevance=model.feature_importances_
            np.savetxt(results_file, relevance, fmt='%f', delimiter='\n') 
            
            # ## SAVE RESULT
            with open(model_name, 'a') as file:
                file.write("Confusion_=\n")
                file.write(str(confusion))
                file.write("\nAUROC="+str(auc_roc))
                file.write("\nAUPR="+str(auc_pr))    
                file.write("\naccuracy="+str(accuracy))
   
            
            # # save model
            joblib.dump(model, model_file)




