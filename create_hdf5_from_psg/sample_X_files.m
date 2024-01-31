% Get all the directories of the stored data
% Determine a number of files (1000?) to try stuff
% Sotre all information in .txt file


% Number of files:
n_files = 1000;
% file out:
file_train = "/scratch/users/mretamales/OSA_scratch/create_hdf5_from_psg/OSA_subset_"+n_files+"_patients_ahi.txt";
files = "/scratch/users/mretamales/OSA_scratch/channel_eval/OSA_all_patients_locations_ahi_sf.txt";

files_T = readtable(files, 'Delimiter', ' ');

files_ok = files_T(find(files_T.All_ch),:);
files_ok = files_ok(find(1-files_ok.Flow_1),1:4);

file_ok_write = "/work/projects/heart_project/OSA_MW/OSA_all_filtered_patients_ahi.txt";
writetable(files_ok,file_ok_write,'Delimiter',' ')

%selected = randperm(height(files_ok),n_files);
%file_s = files_ok(selected,:);
%writetable(file_train,file_out,'Delimiter',' ')
files_train = readtable(file_train, 'Delimiter', ' ');


n_files = 100;
[tf,idx] = ismember(files_ok.File,files_train.File);
files_ok_test = files_ok(find(1-tf),1:4);

selected = randperm(height(files_ok_test),n_files);
file_s = files_ok_test(selected,:);

file_test = "/work/projects/heart_project/OSA_MW/TEST_SET/OSA_subset_"+n_files+"_patients_ahi.txt";
writetable(file_s,file_test,'Delimiter',' ')