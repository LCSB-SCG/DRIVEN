% Run same main but: for files of table, reading their directorues


spo2_delays = [10 15 20 25];

% Generate random seed to go through different files
kk = 10;
reset(RandStream.getGlobalStream,sum(100*clock*kk));
warning('off', 'all')

%% LABELS
% {'normal'}; % 0
% {'ahi'};    % 1

% CHANNELS
% CHANNELS
channels{1}={'EKG','ECGL','ECG','ECGLECGR'};%                               ECG
channels{2}={'ABDO','ABDOMINAL','ABD','ABDORES','ABDOMEN'}; %               ABDOMINAL
channels{3}={'THOR','THORACIC','CHEST','THORRES','THORAX'}; %               CHEST
channels{4}={'FLOW','AUX','CANNULAFLOW','NASALFLOW','NEWAIR', 'AIRFLOW' }; %        NASSAL
channels{5}={'SPO2','SAO2','SPO2'}; %                                       O2
%channels{6}={'LEG'}; %  Leg movement

%min_frequency{1}=128;
%min_frequency{2}=64;
%min_frequency{3}=64;
%min_frequency{4}=64;
%min_frequency{5}=64;
%min_frequency{6}=32;

min_frequency = 64;

% DATA_FOLDERS
%files = "/work/projects/heart_project/OSA_MW/OSA_subset_"+n_files+"_patients_ahi.txt";
%files = "OSA_subset_1000_patients_ahi.txt"; %Created by sample_X_fies.m
files = "/work/projects/heart_project/OSA_MW/TEST_SET/OSA_subset_100_patients_ahi.txt";
files ="/work/projects/heart_project/OSA_MW/OSA_all_filtered_patients_ahi.txt";
files_T = readtable(files, 'Delimiter', ' ');


%%

% Create Output Folder
%dir_out=   "/work/projects/heart_project/OSA_MW/all_"+num2str(tw)+"_ws_"+num2str(stride)+"_st_spo2D_1000_files_ahi_newSF/";
dir_out=   "/work/projects/heart_project/OSA_MW/all_files_ahi_sleep_complete/";
%dir_out=   "/work/projects/heart_project/OSA_MW/all_"+num2str(tw)+"_ws_10648_files_ahi_sleep/";
 try
    mkdir(dir_out);
catch
 end 


%fold_out=   dir_out+"DATA_1D/";
fold_out=   dir_out+"DATA/";
try
    mkdir(fold_out);
catch
end 

% Go through FILES
indxx2= randperm(height(files_T));
n_file = 4;
n_file = 67;
n_file = 909;
n_file=1;
%indxx2 = [1:1:10];
for  n_file= indxx2
    %Get the directories of the psg and labels
    name_file = string(files_T{n_file, 1}); % 'mros-visit1-aa1224.edf';
    edf_dir = string(files_T{n_file, 2}); % "/work/projects/heart_project/backup/OSA_MROS/mros/polysomnography/edfs/visit1/";
    label_dir = string(files_T{n_file, 4}); % "/work/projects/heart_project/backup/OSA_LABELS_RAW/mros/Annotation_visit1/";   

    gen_dat_files_complete(name_file, label_dir, edf_dir, fold_out, channels, min_frequency, spo2_delays);

end


disp("================ Thread finished ===================")

