function [] = gen_img_complete(qq, DATA_short, file_out, dim, spo2_delays)
%{
DRIVEN is © 2024, University of Luxembourg

DRIVEN is published and distributed under the Academic Software License v1.0 (ASL). 

DRIVEN is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the ASL for more details. 

You should have received a copy of the ASL along with this program; if not, write to LCSB-innovation@uni.lu.  It is also published at https://turbogap.fi/wiki/index.php/Academic_Software_Licence.

You may contact the original licensor at LCSB-innovation@uni.lu.
%}   

    if qq ~=5 % not SpO2 (that needs div by 100)
        % NORMALIZATION BY PATIENT WHOLE DATA - excluding 5%
        m_tmp=mean(DATA_short);
        dist = abs(DATA_short - m_tmp);
        [~, sortIndex] = sort(dist);
        index_95perc = sortIndex(1:floor(0.95 * numel(DATA_short))); % get out outliers?
        x_95percent = DATA_short(index_95perc);
        mean_95=mean(x_95percent);
        std_95=std(x_95percent); 
        
        
        x=(DATA_short-mean_95)/std_95;

        % Resize to match NN input dimension and minimum sampling frequency of
        % some sleep records that have 128hz for all channels
        x_pad=imresize(x', [1,dim], "nearest" );


        % WRITE DATA FROM CHANNELS (2-3-4)
        h5write(file_out,"/x"+num2str(qq), x_pad);


    else % SPO2 channel % Check for delays in different subchannels
        % channel 5 no dela
        x=DATA_short(1:end-25)/100;
        x_pad=imresize(x', [1,dim], "nearest" );
        x_pad=mat2gray(x_pad,[0 ,1]);

        % WRITE DATA CHANNELS (5)
        h5write(file_out,"/x"+num2str(qq), x_pad);

        % O2 saturation [0,100] -> [0,1]
        % go through the other delays from the vector
        sp_ind = 7;
        for spo2_delay=spo2_delays
            x=DATA_short(spo2_delay:end-25+spo2_delay)/100;
            x_pad=imresize(x', [1,dim], "nearest" );
            x_pad=mat2gray(x_pad,[0 ,1]);

            h5write(file_out,"/x"+num2str(sp_ind), x_pad);
            sp_ind = sp_ind + 1;
        end 
        

    end


end

