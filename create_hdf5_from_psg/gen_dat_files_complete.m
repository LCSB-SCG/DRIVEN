function [] = gen_dat_files_complete(name_file, label_dir, edf_dir, fold_out, channels, min_frequency, spo2_delays)
try

    file=split(name_file,'.');
    file=file{1};

    % NAME OF OUTPUT FILE
    name_out = file;
    

    % CREATE A TEMPORARY FILE TO ALLOW OTHER THREADS TO GENERATE OTHER FILES
    tmp_file=fold_out+name_out+".remove";
    file_out= fold_out+name_out +".hdf5";
    if isfile(file_out) || isfile(tmp_file)
        % work on next file
        return
    else
        fid = fopen(tmp_file,'w'); fclose(fid);
        tic
        disp(file)

        % READ DATA
        [DATA_s,~] = edfread(edf_dir+name_file);

        % CHECK IF ALL 5 CHANNELS ARE AVAILABLE AND GET INDEXES
        sensors = DATA_s.Properties.VariableNames;
        [all_channels,inx] =find_channel_indx(channels,sensors) ;
        if not(all_channels)
            disp("Not All channels")
            return
        end

        % DATA LENGTH
        duration = DATA_s.Properties.RowTimes(end);
        duration=split(string(duration(1))," ");
        duration=str2double(duration{1})+1; % seconds - starts from 0
        
        % to take 30 mins at the beggining and at the end
        min_init = 30; 
        min_end = 30;
        
        sec_init = 60*min_init;
        sec_end = 60*min_end;
        
        duration_short = duration-sec_init-sec_end;
        dim = min_frequency*duration_short;

        % CREATE TMP AND HDF5 FILES
        try
            h5create(file_out,'/y', [1 dim ],'Datatype','int8');
            h5create(file_out,'/time', [1 dim ],'Datatype','int8');
            h5create(file_out,'/sleep_label', [1 dim ],'Datatype','int8');
            h5create(file_out,'/label_y_s', [1 dim ],'Datatype','int8');
            
        catch
            disp("File already started: "+ name_out)
            return
        end

        % Get time step 
        time_ini=DATA_s.("Record Time")(1);
        time_end=DATA_s.("Record Time")(2);

        time_w = split(  string(time_end-time_ini)," " );
        time_w = str2double( time_w{1} );
        disp(time_w)

        %% GET SAMPLING FS AND CHECK IF DATA LENGTH IS CONSISTENT ACROSS THE CHANNELS
        fs_data=[];
        ee = 1;
        for ee=1:length(inx)
            %Frequency
            sen_inx=inx(ee);
            
            DATA=DATA_s{:,sen_inx};
            if iscell(DATA(1,1))
                l_sample=length(DATA{1,1});
            else
                l_sample=length(DATA(1,1));
            end
            fs=l_sample/time_w;
            fs_data=[fs_data fs];

            if fs>1
                DATA=cat(1,DATA{:,1});
            end
            if duration*fs > length(DATA)
                disp( "Inconsistent channels: "+ name_out )
                
                return
            end
        end

        % GET LABELS
        label_table = load(label_dir+file+'-label.mat');
        labbb = label_table.ahi_labels;
        sleep_label = label_table.sleepStages;
        
        if length(labbb) >= duration
            labbb = labbb(1:duration);
            sleep_label = sleep_label(1:duration);
        end 

        total_sensors=length(inx);

        label_short = labbb(sec_init+1:sec_init+duration_short);
        sleep_short = sleep_label(sec_init+1:sec_init+duration_short);
        
        label_short_dim = repelem(label_short,min_frequency);
        sleep_short_dim = repelem(sleep_short,min_frequency);
        time_short_dim = linspace(0,duration_short, dim);
        
        h5write(file_out,'/y', label_short_dim);
        h5write(file_out,'/time', time_short_dim);
        h5write(file_out,'/sleep_label',  sleep_short_dim);
        
        %qq=5;
        %qq=2;
        %qq = 4;
        %qq=6;
        for qq=2:1:length(inx) % loop through sensors BUT ECG
            sen_inx=inx(qq);
            
            DATA=DATA_s{:,sen_inx};
            fs=fs_data(qq);
            if fs>1
                DATA=cat(1,DATA{:,1});
            end
           
            DATA_short = DATA(sec_init*fs+1:(sec_init+duration_short)*fs);
            
            h5create(file_out,"/x"+num2str(qq), [1 dim ],'Datatype','single');
            
            if qq==5 %% SPO2
                sp_ind = 7;
                 for spo2_delay=spo2_delays
                    h5create(file_out,"/x"+num2str(sp_ind), [1 dim],'Datatype','single');
                    sp_ind = sp_ind + 1;
                 end
                 DATA_short = DATA(sec_init*fs+1:(sec_init+duration_short+25)*fs);
            end

            %take out first 30 minutes (patient is awake and still not in
            %normal position)
            % take out last 30 minues (patient is waking up)
            

            %% GEN IMG
            %[count_img,output_vec]=gen_img(qq, DATA,file_out,tw,fs,dim,min_frequency{qq},labbb,total_sensors);
            gen_img_complete(qq,DATA_short,file_out,dim, spo2_delays);
        end
        
        toc

end 
catch ME
    disp("SOMETHING_WRONG_IN_( "+file+ " )_the message was:\n%s" + ME.message)
    try
        delete(file_out);
        delete(tmp_file);
    catch
    end
end
try
    delete(tmp_file);
catch
    disp("NO_TMP_FILE :"+ file)
end

