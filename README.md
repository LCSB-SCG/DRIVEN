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
- [Code description](#code-description)
- [Results](#results)

## Environments
- **MATLAB** Version R2021a
- **Python**:
  1) For training the NN (optimized for GPU):
     - Python-3.8.6-GCCcore-10.2.0
     - CUDAcore-11.2.1
     - cuDNN-8.1.0.77-CUDA-11.2.1
     - [Library requirements](https://github.com/LCSB-SCG/DRIVEN/tree/main/envitonments/DRIVEN_trainNN.txt)
  3) For the rest of the scripts.
     - Python-3.8.6-GCCcore-10.2.0
     - [Library requirements](https://github.com/LCSB-SCG/DRIVEN/tree/main/envitonments/DRIVEN.yml)


## Code description

- [**From PSG to HDF5**](https://github.com/LCSB-SCG/DRIVEN/tree/main/create_hdf5_from_psg):
    
- [**Preliminary**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/preliminary):
  
- [**Training**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/training):

- [**Feature extraction**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/feature_extraction):
 
- [**Make predictions**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/make_prediction):

- [**Models evaluation**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/models_evaluation):

- [**Figures**](https://github.com/LCSB-SCG/DRIVEN/tree/main/python_code/figures):

## Results


