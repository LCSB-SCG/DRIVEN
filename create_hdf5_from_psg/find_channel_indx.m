function [all_channels,inx] =find_channel_indx(channels,current_channel)
%{
DRIVEN is © 2024, University of Luxembourg

DRIVEN is published and distributed under the Academic Software License v1.0 (ASL). 

DRIVEN is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the ASL for more details. 

You should have received a copy of the ASL along with this program; if not, write to LCSB-innovation@uni.lu.  It is also published at https://turbogap.fi/wiki/index.php/Academic_Software_Licence.

You may contact the original licensor at LCSB-innovation@uni.lu.
%}
inx=[];
count=0;
for ch=1:length(channels)
    flag=0;
    sensors = erase(current_channel, {' ', '.', '_', '-'});
    sensors=upper(sensors);
    for pp=1:length(current_channel)
        for ss=1:length(channels{ch})
            if any(strcmpi(sensors, channels{ch}{ss}) )
                inx=[inx  find( ismember(sensors,channels{ch}{ss}) )];
                flag=1;
                break
            end
        end
        if flag
            count=count+1;
            break
        end
    end
end
if count==ch && length(inx)==ch
    all_channels=1;
else
    all_channels=0;
end

end
