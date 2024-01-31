function [all_channels,inx] =find_channel_indx(channels,current_channel)
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
% function [all_channels,inx] = find_channel_indx(channels,current_channel)
% % Function to find the channels of the sensors to get them from the file
% inx=[];
% count=0;
% 
% sensors = erase(current_channel, {' ', '.', '_', '-'});
% sensors=upper(sensors);
% 
% for ch=1:length(channels)
% 
%     ind_ch = find(ismember(sensors,string(channels{ch})));
%     if ind_ch > 0
%         inx = [inx ind_ch];
%         count = count + 1;
%     end 
% 
% end
% if count==ch && length(inx)==ch
%     all_channels=1;
% else
%     all_channels=0;
% end
% 
% end
