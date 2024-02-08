import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd



# 2
def getSleepStages_arraySeconds(tree):
    stages_vector = []
    stages = tree.findall('SleepStages/SleepStage')
    for stage in range(len(stages)):
        st = int(stages[stage].text)
        stages_vector.extend([st] * 30)
    
    return np.array(stages_vector)



# 3
def getSleepEvents_stages(tree):
    events = tree.findall('ScoredEvents/ScoredEvent')
    events_df = pd.DataFrame(columns = ['Name', 'Start', 'Duration', 'LowestSpO2', 'Desaturation', 'SleepStage'])
    sleepStages = getSleepStages_arraySeconds(tree)
    for event in events:
        name = event.findall('Name')[0].text
        start = float(event.findall('Start')[0].text)
        duration = float(event.findall('Duration')[0].text)
        try:
            sleepStage = sleepStages[int(start)]
        except:
            sleepStage = sleepStages[-1]
        
        if name == 'SpO2 desaturation':
            lowestspo2 = event.findall('LowestSpO2')[0].text
            desaturation = event.findall('Desaturation')[0].text
        else:
            lowestspo2 = '0'
            desaturation = '0'
        
        name = event.findall('Name')[0].text
        event_df = pd.DataFrame.from_dict({'Name':[name], 'Start':[start], 
                                           'Duration':[duration], 'LowestSpO2':[lowestspo2], 
                                           'Desaturation': [desaturation],
                                           'SleepStage': [sleepStage]})
        events_df = pd.concat([events_df, event_df], ignore_index=True)
        
    return sleepStages, events_df



# 4
def ahi_event(apnea_spo_patient_df, ds_distance, ar_distance):
    events_ahi_df = pd.DataFrame(columns = ['Name', 'Start', 'Duration'])
    hyp_s_try = False
    for row, event in apnea_spo_patient_df.iterrows():
        if event["Name"]=="Central Apnea":
            if event["SleepStage"] > 0:
                event_ahi = pd.DataFrame.from_dict({'Name':[event["Name"]], 
                                                    'Start':[event["Start"]], 
                                                    'Duration':[event["Duration"]]})
                events_ahi_df = pd.concat([events_ahi_df, event_ahi], ignore_index=True)
            
        elif event["Name"]=="Obstructive Apnea":
            if event["SleepStage"] > 0:
                event_ahi = pd.DataFrame.from_dict({'Name':[event["Name"]],
                                                    'Start':[event["Start"]],
                                                    'Duration':[event["Duration"]]})
                events_ahi_df = pd.concat([events_ahi_df, event_ahi], ignore_index=True)
            
        elif event["Name"]=="Unsure":
            if event["SleepStage"] > 0:
                hyp_s_try = True
                h_s_name = event["Name"]
                h_s_start = event["Start"]
                h_s_duration = event["Duration"]
                h_s_ds_end = event["Start"]+h_s_duration+ds_distance
                h_s_ar_end = event["Start"]+h_s_duration+ar_distance
            
        elif event["Name"] == "Hypopnea":
            if event["SleepStage"] > 0:
                hyp_s_try = True
                h_s_name = event["Name"]
                h_s_start = event["Start"]
                h_s_duration = event["Duration"]
                h_s_ds_end = event["Start"]+h_s_duration+ds_distance
                h_s_ar_end = event["Start"]+h_s_duration+ar_distance
            
        elif event["Name"]=="SpO2 desaturation":
            if float(event["Desaturation"]) >= 3:
                if hyp_s_try:
                    if h_s_start <= event["Start"] <= h_s_ds_end:
                        event_ahi = pd.DataFrame.from_dict({'Name':[h_s_name],
                                                            'Start':[h_s_start],
                                                            'Duration':[h_s_duration]})
                        events_ahi_df = pd.concat([events_ahi_df, event_ahi], ignore_index=True)
                        hyp_s_try = False
                    
            
        elif "Arousal" in event["Name"]:
            if hyp_s_try:
                if h_s_start <= event["Start"] <= h_s_ar_end:
                    event_ahi = pd.DataFrame.from_dict({'Name':[h_s_name],
                                                        'Start':[h_s_start],
                                                        'Duration':[h_s_duration]})
                    events_ahi_df = pd.concat([events_ahi_df, event_ahi], ignore_index=True)
                    hyp_s_try = False
                
            
        else:
            print(event["Name"])
        
    return events_ahi_df




# 5
def label_seconds(events_ahi_df, time):
    labels = np.zeros(time).astype(int)
    for row, event in events_ahi_df.iterrows():
        start = event["Start"]
        end = start + event["Duration"]
        labels[int(start):int(end)] = 1
    
    return labels

# 6
def get_ahi(xml_file, ahi_file):
    ahi_read = pd.read_csv(ahi_file)
    patient = xml_file.split('/')[-1]
    ahi = float(ahi_read[ahi_read["Patient"] ==  patient]["ahi_a0h3a"])
    return ahi

# This will be run for one patient
def getLabels(xml_file):
    # 1. Read xml file 
    tree = ET.parse(xml_file)
    # 2. Get sleep stages / per second
    # 3. Get sleep events
    sleepStages, events_df = getSleepEvents_stages(tree)
    # 2.1 Sleep time
    sleep_time = (((0 < sleepStages) & (sleepStages < 6)).sum())/60
    # 4. Select the sleep events that count in AHI 
    ds_distance = 45
    ar_distance = 5
    events_ahi_df = ahi_event(events_df, ds_distance, ar_distance)
    n_events_ahi = len(events_ahi_df)
    # 5. Label seconds 
    ahi_labels = label_seconds(events_ahi_df, len(sleepStages))
    # 6. Get ahi and calculate 
    #ahi_db = get_ahi(xml_file, ahi_file)
    ahi_c = n_events_ahi / sleep_time * 60
    #return ahi_db, ahi_c, sleepStages, labels, events_df, sleep_time, events_ahi_df
    res_dict = {"ahi_db": 0, "ahi_c": ahi_c, "sleepStages": list(sleepStages), "ahi_labels": list(ahi_labels)}
    return res_dict







if __name__ == '__main__':
    

    ahi_file = "/work/projects/heart_project/OSA_MW/ahi_eval/ahi_final.csv"
    labels_dir = "/work/projects/heart_project/OSA_MW/LABELS_MAT_AHI/"
    
    files_ahi = (pd.read_csv(ahi_file))
    data = files_ahi[files_ahi["Patient"]=="shhs1-203181-profusion.xml"].iloc[0]
    
    for row, data in files_ahi.iterrows():
        
        patient = data["Patient"]
        #xml_file = work/projects/heart_project/backup/OSA_LABELS_RAW/annotations-events-profusion_mesa
        #print(patient)
        
        if "mesa" in patient:
            xml_file = "/work/projects/heart_project/backup/OSA_LABELS_RAW/annotations-events-profusion_mesa/"+patient
            
        elif "shhs1" in patient:
            xml_file = "/work/projects/heart_project/backup/OSA_LABELS_RAW/annotations-events-profusion_shhs/shhs1/"+patient
        
        elif "shhs2" in patient:
            xml_file = "/work/projects/heart_project/backup/OSA_LABELS_RAW/annotations-events-profusion_shhs/shhs2/"+patient
        
        elif "mros-visit1" in patient:
            xml_file = "/work/projects/heart_project/backup/OSA_LABELS_RAW/annotations-events-profusion_mros/visit1/"+patient
        
        elif "mros-visit2" in patient:
            xml_file = "/work/projects/heart_project/backup/OSA_LABELS_RAW/annotations-events-profusion_mros/visit2/"+patient
        
        else:
            print("NOT FOUND")
            print(patient)
            continue
            
        try:
            labels_dict = getLabels(xml_file)
            labels_dict["ahi_db"] = data["ahi_a0h3a"]
            print(labels_dict["ahi_db"])
            print(labels_dict["ahi_c"])

        except:
            print("NOT FOUND")
            print(patient)

	patient_mat = labels_dir+patient[:-14]+"-label.mat"
	scipy.io.savemat(patient_mat, labels_dict, do_compression=True)










