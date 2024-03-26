function [img_count,output_vec] = gen_img_stride_ahi_types(qq,DATA_short,file_out,tw,stride, fs,dim, label_short,total_sensors, spo2_delays, sleep_short, sec_init)

    img_count=1;
    ini_data=stride;
    sec_init = sec_init+stride;
    flagg=1;
    N=length(label_short);
    class_n=0;
    class_o=0;
    %output_vec=[class_n class_o];
    
    
    % NORMALIZATION BY PATIENT WHOLE DATA - excluding 5%
    m_tmp=mean(DATA_short);
    dist = abs(DATA_short - m_tmp);
    [~, sortIndex] = sort(dist);
    index_95perc = sortIndex(1:floor(0.95 * numel(DATA_short))); % get out outliers?
    x_95percent = DATA_short(index_95perc);
    mean_95=mean(x_95percent);
    std_95=std(x_95percent); 
    
    
    while(flagg)
        
        % OSA EVENT
        
        if qq == 1
            x=DATA_short(ini_data*fs: ini_data*fs + (tw)*fs -1);
            try
                [~,qrs_i_raw,~]=pan_tompkin2(double(x),fs,0);
                qrs_i_raw=qrs_i_raw/fs;
                R_WAVES=[];
                if length(qrs_i_raw)>1
                    for k11=2:length(qrs_i_raw)
                        R_WAVES(k11-1)=qrs_i_raw(k11)-qrs_i_raw(k11-1);
                    end
                else
                    R_WAVES=zeros(1,tw);
                    %disp("R_WAVES len < 1")
                end
            catch
                R_WAVES=zeros(1,tw);
                disp("R_WAVES error")
            end

            % WRITE RRI DATA WHEN GOING THROUGH ECG
            x_pad_ecg=imresize(R_WAVES, [1,dim], "nearest" );
            x_pad_ecg=mat2gray(x_pad_ecg,[0 3]);
            info = h5info(file_out, "/x"+num2str(qq));
            curSize = info.Dataspace.Size;
            h5write(file_out,"/x"+num2str(qq), x_pad_ecg,[ 1 1 curSize(end)+1] , [1  dim 1]);
            img_count=img_count+1;
            
        elseif qq ~=5 % not SpO2 (that needs div by 100)
            x=DATA_short(ini_data*fs: ini_data*fs + (tw)*fs -1);
            
            %x=DATA(ini_data*fs: ini_data*fs + (tw-1)*fs );
            x=(x-mean_95)/std_95;

            % Resize to match NN input dimension and minimum sampling frequency of
            % some sleep records that have 128hz for all channels
            x_pad=imresize(x', [1,dim], "nearest" );

            
            % WRITE DATA FROM CHANNELS (2-3-4)
            info = h5info(file_out, "/x"+num2str(qq));
            curSize = info.Dataspace.Size;
            h5write(file_out,"/x"+num2str(qq), x_pad,[ 1 1 curSize(end)+1] , [1  dim 1]);
            img_count=img_count+1;
            
            
            
        else % SPO2 channel % Check for delays in different subchannels
            % channel 5 no delay
            x=DATA_short((ini_data)*fs: (ini_data)*fs + (tw)*fs -1)/100;
            x_pad=imresize(x', [1,dim], "nearest" );
            x_pad=mat2gray(x_pad,[0 ,1]);

            % WRITE DATA CHANNELS (5)
            info = h5info(file_out, "/x"+num2str(qq));
            curSize = info.Dataspace.Size;
            h5write(file_out,"/x"+num2str(qq), x_pad,[ 1 1 curSize(end)+1] , [1  dim 1]);
            
            % O2 saturation [0,100] -> [0,1]
            % go through the other delays from the vector
            sp_ind = 7;
            for spo2_delay=spo2_delays
                x=DATA_short((ini_data+spo2_delay)*fs: (ini_data+spo2_delay)*fs + (tw)*fs -1)/100;
                x_pad=imresize(x', [1,dim], "nearest" );
                x_pad=mat2gray(x_pad,[0 ,1]);

                % plot(x_pad)
                % WRITE DATA FROM CHANNELS (1-5)
                info = h5info(file_out, "/x"+num2str(sp_ind));
                curSize = info.Dataspace.Size;
                h5write(file_out,"/x"+num2str(sp_ind), x_pad,[ 1 1 curSize(end)+1] , [1  dim 1]);
                sp_ind = sp_ind + 1;
            end 
            img_count=img_count+1;
            
        end
        
        
        
        
        output_vec=[0 0];
        if qq ==total_sensors
            current_sleep = sleep_short( ini_data:ini_data+tw-1);
            if sum(current_sleep)>0
                label_s = 1;
            else 
                label_s = 0;
            end

            current_label=label_short(ini_data:ini_data+tw-1);
            if label_s == 0
                label = 0;
            else 
                %1. get the values that are not 0
                current_label_not0 = current_label((current_label>0));
                if isempty(current_label_not0)
                    label = 0;
                else
                    current_label_not04 = current_label_not0(current_label_not0<4);
                    % If there is no more than 4
                    if isempty(current_label_not04)
                        label = 4;
                    else
                        label = mode(current_label_not04);
                    end 
                end 
                %label=mode(current_label(current_label>0)); %NEEDS TO ALSO BE LESS THAN 4 
                %label = mode(current_label(find((current_label>0).*(current_label<4))));

            end 

            label_3c = label_s+label;
            
            h5write(file_out,'/y', label,[1 curSize(end)+1] , [1 1]);
            h5write(file_out,'/time', sec_init,[1 curSize(end)+1] , [1 1]);
            h5write(file_out,'/sleep_label', label_s,[1 curSize(end)+1] , [1 1]);
            h5write(file_out,'/label_y_s', label_3c,[1 curSize(end)+1] , [1 1]);
              
            h5write(file_out,'/event', current_label,[1 1 curSize(end)+1] , [1 tw 1]);
            h5write(file_out,'/sleep_time', current_sleep,[1 1 curSize(end)+1] , [1 tw 1]);

            if label>0
                class_o=class_o+1; % Considers all numbers 
            else
                class_n=class_n+1;
            end
            
            output_vec=[class_n class_o];
        end 

        ini_data=ini_data+stride;
        sec_init=sec_init+stride;
        a=ini_data + tw + spo2_delays(end);

        if ( a>=N  )
            flagg=0;

        end
    end
    
end

