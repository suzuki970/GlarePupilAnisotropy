%%%%%%%%%%%%%%%%%%%%%%%%%%%% New analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
%% read json data
fileName = '/Users/yutasuzuki/Desktop/Experimental_data/P06/data/data2.json'; % filename in JSON extension
str = fileread(fileName); % dedicated for reading files as text
data = jsondecode(str); % Using the jsondecode function to parse JSON from string
y=data.PDR;
sub=data.sub;

locs = ["Upper","Lower","Center","Left","Right","Upper","Lower","Center","Left","Right"];
pattern = ["Glare","Glare","Glare","Glare","Glare",...
           "Control","Control","Control","Control","Control"];

condition_locs = locs(data.condition);
condition_pattern = pattern(data.condition);

%% latenct at min value
tmp = [];
for iSub = unique(sub)'
    ind = find((sub==iSub)' & condition_locs=="Center");
    tmp = [tmp;mean(y(ind,:),1)];
end
tmp = mean(tmp,1);
[latency(1),latency(2)] = min(tmp);

%% compornent
early = y(:,1:latency(2));
early = mean(early,2);

late = y(:,latency(2):end);
late = mean(late,2);

summary = struct('sub',data.sub,'early',early,'late',late,...
                'condition',data.condition);

summary = struct2table(summary);
mean = groupsummary(summary,{'sub','condition'},"mean");
grand_mean = groupsummary(mean,{'condition'},"mean");
sd = groupsummary(mean,{'condition'},"std");

%% plot Early
figure(1);
h = [grand_mean.mean_mean_early(1:5) grand_mean.mean_mean_early(6:10)];
err = [sd.std_mean_early(1:5) sd.std_mean_early(6:10)] / sqrt(length(unique(summary.sub)));

b1=bar(h,'BarWidth', 1);
b1(2).FaceColor = [0.5 0.5 0.5];
b1(1).FaceColor = [1 1 1];
b1(1).EdgeColor=[0.27 .35 .27];
hold on;
ngroups = size(h, 1);
nbars = size(h, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    b2=errorbar(x, h(:,i),err(:,i), '.k');
end
hold off
xticklabels({'TOP','BOTTOM','CENTER','LEFT','RIGHT'})
legend('Glare','Control');
ylabel('Pupil Diameter Change (a.u.)');
% title('Changes in Pupillary Diameter (1-4 seconds)');
set(gcf, 'PaperPositionMode', 'auto');
ax = gca;
ax.YGrid = 'on';

%% plot Late
figure(2);
h = [grand_mean.mean_mean_late(1:5) grand_mean.mean_mean_late(6:10)];
err = [sd.std_mean_late(1:5) sd.std_mean_late(6:10)] / sqrt(length(unique(summary.sub)));

b1=bar(h,'BarWidth', 1);
b1(2).FaceColor = [0.5 0.5 0.5];
b1(1).FaceColor = [1 1 1];
b1(1).EdgeColor=[0.27 .35 .27];
hold on;
ngroups = size(h, 1);
nbars = size(h, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    b2=errorbar(x, h(:,i),err(:,i), '.k');
end
hold off
xticklabels({'TOP','BOTTOM','CENTER','LEFT','RIGHT'})
legend('Glare','Control');
ylabel('Pupil Diameter Change (a.u.)');
% title('Changes in Pupillary Diameter (1-4 seconds)');
set(gcf, 'PaperPositionMode', 'auto');
ax = gca;
ax.YGrid = 'on';
