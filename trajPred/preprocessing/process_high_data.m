%% Process dataset into mat files %%

%% This is a modified version of the code : https://github.com/nachiket92/conv-social-pooling %%


clear;
clc;

%% Inputs:
% Locations of raw input files:

for inx = 1:60
    data{inx} = sprintf('../dataset/HighD/merged_%d.csv', inx);


%% Fields: 

%{ 
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: Lane Id
7: Lateral maneuver
8: Lateral maneuver time
9: Longitudinal maneuver
10: Longitudinal maneuver intensity
11-49: Neighbor Car Ids at grid location
%}



%% Load data and add dataset id
fprintf('Loading data...')
for ind = 1:60
    traj{ind} = load(data{ind});
    traj{ind} = single([ind*ones(size(traj{ind},1),1),traj{ind}]);
    fprintf(sprintf('%d done\n', ind))
fprintf(' done\n')

for k = 1:60
    num_elem = length(traj{k}(:,1));
    traj_temp = zeros(num_elem, 49);
    traj_temp(:, 1:6) = traj{k}(:,[1,2,3,6,7,15]);
    traj{k} = traj_temp;
    traj{k}(traj{k}(:,6)>=6,6) = 6;
    end
end
clear traj_temp;

% poolobj = parpool(2);

%% Parse fields (listed above):
disp('Parsing fields:')
lastsize = 0;
for ii = 1:60
    
    %% Get cell of trajectory id list and cells of corresponding trajectory data
    vehidlist = unique(traj{ii}(:, 2));
    vehtraj = cell(length(vehidlist));
    for l = 1:length(vehidlist)
        vehtraj{l} = traj{ii}(traj{ii}(:,2)==vehidlist(l),:);
    end
    
    %% Get cell of time list and cells of corresponding trajectory data
    timelist = unique(traj{ii}(:,3));
    timetraj = cell(length(timelist));
    num_elem = length(timelist);
    for l = 1:num_elem
        timetraj{l} = traj{ii}(traj{ii}(:, 3) == timelist(l), :);
    end
    
    num_elem = length(traj{ii}(:,1));
    advance = (ii-1)*100/60;
    advance_prev = advance;
    for k = 1:num_elem
        advance = (ii-1)*100/60 + (k-1)*100/(60*num_elem);
        if advance >= advance_prev+1
            advance_prev = advance;
            fprintf(repmat('\b', 1, lastsize));
            lastsize = fprintf('Computing... %d%%', floor(advance));
        end
        
        time = traj{ii}(k,3);
        vehId = traj{ii}(k,2);
        trajind = find(vehidlist == vehId);
        trajind = trajind(1);
        timeind = find(timelist == time);
        timeind = timeind(1);
                
        [lower, ~] = find_sorted(vehtraj{trajind}(:,3), time);
        ind = lower;
        lane = traj{ii}(k,6);
        
        
        % Get lateral maneuver:
        ub = min(size(vehtraj{trajind},1),ind+50);
        lb = max(1, ind-10);
        lane_change = vehtraj{trajind}(lb+1:ub,6) - vehtraj{trajind}(lb:ub-1,6);
        index_maneuver = floor(mean(find(lane_change)));
        if vehtraj{trajind}(ub,6)>vehtraj{trajind}(ind,6) || vehtraj{trajind}(ind,6)>vehtraj{trajind}(lb,6)
            traj{ii}(k,7) = 3;
            traj{ii}(k,8) = lb + index_maneuver - ind;
        elseif vehtraj{trajind}(ub,6)<vehtraj{trajind}(ind,6) || vehtraj{trajind}(ind,6)<vehtraj{trajind}(lb,6)
            traj{ii}(k,7) = 2;
            traj{ii}(k,8) = lb + index_maneuver - ind;
        else
            traj{ii}(k,7) = 1;
        end
        
        
        % Get longitudinal maneuver:
        ub = min(size(vehtraj{trajind},1),ind+50);
        lb = max(1, ind-10);
        if ub==ind || lb ==ind
            traj{ii}(k,9) =1;
        else
            vHist = (vehtraj{trajind}(ind,5)-vehtraj{trajind}(lb,5))/(ind-lb);
            vFut = (vehtraj{trajind}(ub,5)-vehtraj{trajind}(ind,5))/(ub-ind);
            if vFut/vHist <0.8
                traj{ii}(k,9) =2;
                traj{ii}(k,10) = (vHist-vFut)/vHist;
            else
                traj{ii}(k,9) =1;
                traj{ii}(k,10) = (vHist-vFut)/vHist;
            end
        end
        
        
        % Get grid locations:
        frameEgo = timetraj{timeind}(timetraj{timeind}(:,6) == lane,:);
        frameL = timetraj{timeind}(timetraj{timeind}(:,6) == lane-1,:);
        frameR = timetraj{timeind}(timetraj{timeind}(:,6) == lane+1,:);
        if ~isempty(frameL)
            for l = 1:size(frameL,1)
                y = frameL(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 1+round((y+90)/15);
                    traj{ii}(k,10+gridInd) = frameL(l,2);
                end
            end
        end
        for l = 1:size(frameEgo,1)
            y = frameEgo(l,5)-traj{ii}(k,5);
            if abs(y) <90 && y~=0
                gridInd = 14+round((y+90)/15);
                traj{ii}(k,10+gridInd) = frameEgo(l,2);
            end
        end
        if ~isempty(frameR)
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 27+round((y+90)/15);
                    traj{ii}(k,10+gridInd) = frameR(l,2);
                end
            end
        end
        
    end
end
fprintf(repmat('\b', 1, lastsize));
fprintf('Computing... done\n');

%% Split train, validation, test
fprintf('Splitting into train, validation and test sets...')

trajAll = [traj{1};traj{2};traj{3};traj{4};traj{5};traj{6};traj{7};traj{8};traj{9};traj{10};traj{11};traj{12};traj{13};traj{14};traj{15};traj{16};traj{17};traj{18};traj{19};traj{20};traj{21};traj{22};traj{23};traj{24};traj{25};traj{26};traj{27};traj{28};traj{29};traj{30};traj{31};traj{32};traj{33};traj{34};traj{35};traj{36};traj{37};traj{38};traj{39};traj{40};traj{41};traj{42};traj{43};traj{44};traj{45};traj{46};traj{47};traj{48};traj{49};traj{50};traj{51};traj{52};traj{53};traj{54};traj{55};traj{56};traj{57};traj{58};traj{59};traj{60};];
clear traj;

trajTr = [];
trajVal = [];
trajTs = [];
for k = 1:60
    ul1 = round(0.7*max(trajAll(trajAll(:,1)==k,2)));
    ul2 = round(0.8*max(trajAll(trajAll(:,1)==k,2)));
    
    trajTr = [trajTr;trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :)];
    trajVal = [trajVal;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :)];
    trajTs = [trajTs;trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :)];
end

 tracksTr = {};
for k = 1:60
    trajSet = trajTr(trajTr(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
        tracksTr{k,carIds(l)} = vehtrack;
    end
end

tracksVal = {};
for k = 1:60
    trajSet = trajVal(trajVal(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
        tracksVal{k,carIds(l)} = vehtrack;
    end
end

tracksTs = {};
for k = 1:60
    trajSet = trajTs(trajTs(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:5)';
        tracksTs{k,carIds(l)} = vehtrack;
    end
end
fprintf(' done\n')

%% Filter edge cases: 
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing

fprintf('Filtering edge cases...')

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
    t = trajTr(k,3);
    if tracksTr{trajTr(k,1),trajTr(k,2)}(1,31) <= t && tracksTr{trajTr(k,1),trajTr(k,2)}(1,end)>t+1
        indsTr(k) = 1;
    end
end
trajTr = trajTr(find(indsTr),:);


indsVal = zeros(size(trajVal,1),1);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    if tracksVal{trajVal(k,1),trajVal(k,2)}(1,31) <= t && tracksVal{trajVal(k,1),trajVal(k,2)}(1,end)>t+1
        indsVal(k) = 1;
    end
end
trajVal = trajVal(find(indsVal),:);


indsTs = zeros(size(trajTs,1),1);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    if tracksTs{trajTs(k,1),trajTs(k,2)}(1,31) <= t && tracksTs{trajTs(k,1),trajTs(k,2)}(1,end)>t+1
        indsTs(k) = 1;
    end
end
trajTs = trajTs(find(indsTs),:);
fprintf(' done\n')

%% Save mat files:
fprintf('Saving mat files...')

traj = trajTr;
tracks = tracksTr;
save('TrainSet_traj_v2.mat', 'traj', '-v7.3')
save('TrainSet_tracks_v2.mat', 'tracks')

traj = trajVal;
tracks = tracksVal;
save('ValSet_traj_v2.mat','traj','-v7.3');
save('ValSet_tracks_v2.mat','tracks');

traj = trajTs;
tracks = tracksTs;
save('TestSet_traj_v2.mat','traj','-v7.3');
save('TestSet_tracks_v2.mat','tracks');
fprintf(' done\n')
disp('All done.')
