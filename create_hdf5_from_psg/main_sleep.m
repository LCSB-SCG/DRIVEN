% Run same main but: for files of table, reading their directorues
% Change the tw size

kk = 10;
tws = [10 30 80];
tws = [30];

strides = [5];
spo2_delays = [10 15 20 25];

% Generate random seed to go through different files
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

min_frequency{1}=128;
min_frequency{2}=64;
min_frequency{3}=64;
min_frequency{4}=64;
min_frequency{5}=64;
%min_frequency{6}=32;


% DATA_FOLDERS
files = "/work/projects/heart_project/OSA_MW/TEST_SET/OSA_subset_100_patients_ahi.txt";
files ="/work/projects/heart_project/OSA_MW/OSA_all_filtered_patients_ahi.txt";
files_T = readtable(files, 'Delimiter', ' ');


%% Go through time windows

tw_index = randperm(length(tws));
tw_i = 3;

for tw_i = tw_index
    
    tw = tws(tw_i);
    stride = strides(tw_i);
    disp(tw)
    % Create Output Folder
    dir_out=   "/work/projects/heart_project/OSA_MW/all_"+num2str(tw)+"_ws_10648_files_ahi_sleep_newSF/";
    try
        mkdir(dir_out);
    catch
    end 


    fold_out=   dir_out+"DATA_"+num2str(tw)+"s/";
    try
        mkdir(fold_out);
    catch
    end 

    % Go through FILES
    indxx2= randperm(height(files_T));

    for  n_file= indxx2
        %Get the directories of the psg and labels
        name_file = string(files_T{n_file, 1}); % 'mros-visit1-aa1224.edf';
        edf_dir = string(files_T{n_file, 2}); % "/work/projects/heart_project/backup/OSA_MROS/mros/polysomnography/edfs/visit1/";
        label_dir = string(files_T{n_file, 4}); % "/work/projects/heart_project/backup/OSA_LABELS_RAW/mros/Annotation_visit1/";   
        
        gen_dat_files_sleep(tw, stride, name_file, label_dir, edf_dir, fold_out, channels, min_frequency, spo2_delays);

    end

end
disp("================ Thread finished ===================")

