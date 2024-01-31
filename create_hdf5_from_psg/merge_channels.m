files = dir("/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/DATA/");

for fs=10129:length(files)
    try
        file_name = (fullfile("/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/DATA/",files(fs).name));
        
        x1 = h5read(file_name, '/x2');
        x2 = h5read(file_name, '/x3');
        x3 = h5read(file_name, '/x4');
        x4 = h5read(file_name, '/x5');
        x5 = h5read(file_name, '/x7');
        x6 = h5read(file_name, '/x8');
        x7 = h5read(file_name, '/x9');
        x8 = h5read(file_name, '/x10');
        
        c = single(cat(4,x1,x2,x3,x4,x5,x6,x7,x8));
        
        h5create(file_name,"/X", size(c),'Datatype','single');
        h5write(file_name,"/X",c)

    catch
        disp("DIDNT WORK")
        disp(files(fs).name)
    end
        
end 

fs = 10130;
file_name = (fullfile("/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/DATA/",files(fs).name));
x8 = h5read(file_name, '/X');   
h5disp(file_name, '/')



file_name = "/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/DATA/shhs1-203553.hdf5";
%h5disp(file_name, '/')        
x = h5read(file_name, '/X'); 
TF = isnan(x);


if sum(sum(sum(TF))) > 0
    disp(file_name)
end 


for fs=1:length(files)
    try
        file_name = (fullfile("/work/projects/heart_project/OSA_MW/all_80_ws_10648_files_ahi_sleep_newSF/DATA/",files(fs).name));
        
        x = h5read(file_name, '/X'); 
        TF = isnan(x);
    
        if sum(sum(sum(TF))) > 0
            disp("NAN")
            disp(file_name)
        end 

    catch
        disp("DIDNT WORK")
        disp(files(fs).name)
    end
        
end 

