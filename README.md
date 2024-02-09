# DRIVEN: TOWARDS AUTOMATIC HOME-BASED SLEEP APNEA ESTIMATION USING DEEP LEARNING


Apnea and hypopnea are common sleep disorders characterized by complete or partial obstructions of the airways, respectively. 
Early detection and treatment of apnea can significantly reduce morbidity and mortality.
However, clinical detection is costly and uncomfortable for patients. 

**DRIVEN (Detection of obstRuctIVE slEep apNea):**
Detects when apnea and hypopnea events occur throughout the night and estimates AHI at home from simple and comfortable sensors.
This assist physicians in diagnosing the severity of apneas. 


![fig1](https://github.com/LCSB-SCG/DRIVEN/assets/26947730/30606229-7144-45ba-abe3-f34ee22636b6)


## Contents
- [Environments](#environments)
- [Files description](#files-description)
- [Code description](#code-description)
- [Results](#results)

## Environments
- **MATLAB** Version R2021a
- **Python**:
  1) For training the NN (optimized for GPU):
     - Python-3.8.6-GCCcore-10.2.0
     - CUDAcore-11.2.1
     - cuDNN-8.1.0.77-CUDA-11.2.1
     - [Library requirements](https://github.com/LCSB-SCG/DRIVEN/tree/main/envitonments/requirements/DRIVEN_trainNN_requirements.txt)
  3) For the rest of the scripts.
     - Python-3.8.6-GCCcore-10.2.0
     - [Library requirements](https://github.com/LCSB-SCG/DRIVEN/tree/main/envitonments/requirements/DRIVEN_requirements.txt)


## Files description
- **Polysonomography:** Exam used to diagnosis and detect sleep apnea and hypopnea, conducted in a sleep facility or at home. PSG records several physiological signals, including electroencephalogram (EEG), electrocardiogram (ECG), electrooculogram (EOG), chin muscle activity, leg movements, respiratory effort, nasal airflow, and oxygen saturation (SpO2). The polysomnography data is available in European Data Format (EDF).
- **Profusion:** Medical anotation of the events detected in the PSG. (One per PSG). The file includes annotations for every 30-second sample of sleep stages and instances of sleep problems, with information of initiation and duration time, as well as the sensor used to detect the anomaly and the profusion files. The profusion file is available in XML format.
- **Window files:** File with channel recodings (PSG) and labeling (Profusion) together. For each file, 30-second windows are created which have all the relevant information to trai/validate/test the models. This file is available in HDF5 format. 
- **Models:** Trained EfficientNetV2 and LightGBM classifier to predict AHI-positive-events (apneas and hypopneas).
- **Predictions:** Predictions with the trained models for every window of the window-files, contains the probability of the window to have an AHI-positive-event. This file is available in HDF5 format. 

![Data prep](https://github.com/LCSB-SCG/DRIVEN/assets/26947730/b6c098c6-403a-4da3-bd86-0cb3397177d7)


## Code description

- [**Preliminary**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/preliminary):\
  **label_ahi.py:** Get profusion file per patient and label the sleep stages and AHI-evenst per second.\
  **separate_files_alldb.py:** Split the dataset for train-validation-test.\
  **get_indices.py:** Loop through window files and make dictionary with number of windows awake, sleep-ahi-negative, sleep-ahi-positive, and indices for each of these windows.
  
- [**From PSG to HDF5**](https://github.com/LCSB-SCG/DRIVEN/tree/main/create_hdf5_from_psg):\
  Pipeline to merge PSG and profusion labels and generate windows of the selected size with the relevant channels.\
  For each PSG file, it will check that the relevant channels are present, and that the channel has the same frequency throughout the timeseries. The initial and final 30 minutes will be removed. Each channel (except from SpO2) will be Z-normalized, and all channels will be set in the same sampling frequency. For each window, the 
  
  
- [**Training**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/training):\
  **train_nn.py:** Load model (from file or new) and train with samples from patients for one channel. The training is done by sampling the same number of positives and negatives per file processed. The script can be used to train and detect AHI-events or sleep-awake classification (ahi_or_sleep = "ahi" or "sleep").

      python train_nn.py $batch_size $fold $channel $window_size $new_model $working_directory $ahi_or_sleep $stride

  **train_lgbm.py:** Load extracted freatures for training and validation and train classifier for all possible combibanitos of sensors. The training is done by sampling the same number of positives and negatives per file processed. The script can be used to train and detect AHI-events or sleep-awake classification (ahi_or_sleep = "ahi" or "sleep") and to test for the last last or the third-last layers ($last_layer = 1 or 3)

      python train_lgbm.py $fold $working_directory $ahi_or_sleep $last_layer


- [**Feature extraction**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/feature_extraction):
 
- [**Make predictions**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/make_prediction):

- [**Models evaluation**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/models_evaluation):

- [**Figures**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/figures):

## Results


