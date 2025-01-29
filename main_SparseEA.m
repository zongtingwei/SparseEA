% the main function of SparseEA
% the data set is in the file /dataset e.g.: dataset/colon.mat
% the results are saved as a .mat file, and main results are:
%     unionPF is the PF of the final population
%     unionPFfit is the object function (size of features and the error rate on test set)

clc;
clear;

tic;

algorithmName = 'SparseEA';  
dataNameArray = {'lung'}; % data set
global maxFES
maxFES = 100;  % max number of iteration
global choice
choice = 0.6; % the threshold choose features
global sizep
sizep = 300; % size of population

global LOOCV
LOOCV = 2; % use 5-fold when LOOCV = 1
global fold
fold = 5; %%%%%%%%%%%

iterator = 5;
global CNTTIME
CNTTIME = maxFES * iterator;

for data = 1:length(dataNameArray)
    clc;
    clearvars -except dataNameArray data algorithmName maxFES choice sizep iterator LOOCV fold
    outcome = cell(iterator, 8);
    errorOnTest = zeros(iterator, 2);
    aveTrain1 = 0;
    aveTrain2 = 0;
    unionPF = [];
    unionPFfit = [];
    aveunionPF = [0 0];
    trainPF = [];
    trainPFfit = [];

    dataName = dataNameArray{data};
    file = ['dataset/', dataName, '.mat'];
    load(file);
    dataMat = ins;
    numFeatures = size(dataMat, 2);

    for i = 1:iterator
        fprintf('-----Now: %d-----\n', i);
        dataName = dataNameArray{data};
        if LOOCV == 1
            fprintf('LOOCV\n');
            [train_F, train_L, test_F, test_L] = DIVDATA5fold(dataName, i);  % read data
        else
            [train_F, train_L, test_F, test_L] = DIVDATA37(dataName);
        end

        % return cell "outcome" as results, SparseEA is the algorithm
        [outcome{i, 1}, outcome{i, 2}, outcome{i, 3}, outcome{i, 4}, outcome{i, 5}, outcome{i, 6}, outcome{i, 7}] = SparseEA(train_F, train_L, maxFES, sizep);

        ofit = outcome{i, 4};
        Tsite = outcome{i, 5};
        Toff = outcome{i, 3};
        Tgood = Toff(Tsite, :);
        unionPF = [unionPF; Tgood];
        tempE = 0;
        tempF = 0;

        for j = 1:size(Tgood, 1) % test population "unionPF" on test set
            FeatureSubset = Tgood(j, :);
            CInsTrain = train_F;
            CInsTrain(:, ~FeatureSubset) = 0;
            mdl = ClassificationKNN.fit(CInsTrain, train_L, 'NumNeighbors', 5);
            [label] = predict(mdl, test_F);
            Popscore = 0;
            for k = 1:size(test_F, 1)
                if label(k) == test_L(k)
                    Popscore = Popscore + 1;
                end
            end
            temp1 = 1 - Popscore / size(test_F, 1);
            temp2 = sum(FeatureSubset) / numFeatures; % Normalize the number of features by the total number of features
            if temp2 > 0 % Only consider particles with non-zero features
                unionPFfit = [unionPFfit; [temp1 temp2]]; % unionPFfit  
            end
        end
        errorOnTest(i, 1) = tempE / size(Tgood, 1); % errorOnTest
        errorOnTest(i, 2) = tempF / size(Tgood, 1);

        % Evaluate on training set
        for j = 1:size(Tgood, 1)
            FeatureSubset = Tgood(j, :);
            CInsTrain = train_F;
            CInsTrain(:, ~FeatureSubset) = 0;
            mdl = ClassificationKNN.fit(CInsTrain, train_L, 'NumNeighbors', 5);
            [label] = predict(mdl, CInsTrain);
            Popscore = 0;
            for k = 1:size(CInsTrain, 1)
                if label(k) == train_L(k)
                    Popscore = Popscore + 1;
                end
            end
            temp1 = 1 - Popscore / size(CInsTrain, 1);
            temp2 = sum(FeatureSubset) / numFeatures; % Normalize the number of features by the total number of features
            if temp2 > 0 % Only consider particles with non-zero features
                trainPFfit = [trainPFfit; [temp1 temp2]]; % unionPFfit 
            end
        end
        aveTrain1 = aveTrain1 + outcome{i, 6}(1);
        aveTrain2 = aveTrain2 + outcome{i, 6}(2);
    end
    aver_errorOnTest = mean(errorOnTest); % average error on test set
    aveTrain1 = aveTrain1 / iterator;
    aveTrain2 = aveTrain2 / iterator;
    [FrontNOunion, ~] = NDSort(unionPFfit(:, 1:2), size(unionPFfit, 1));
    siteunionPF = find(FrontNOunion == 1);  
    firsttestFront = unionPFfit(siteunionPF, :);
    [FrontNOtrain, ~] = NDSort(trainPFfit(:, 1:2), size(trainPFfit, 1));
    sitetrainPF = find(FrontNOtrain == 1);
    firsttrainFront = trainPFfit(sitetrainPF, :);
    aveunionPF = mean(unionPFfit(siteunionPF, :)); 

    % Output the First Pareto Front on test set
    fprintf('Outputting the First Pareto Front on Test Set:\n');
    disp(firsttestFront);

    % Save the results
    savename = [algorithmName '-' dataNameArray{data}];
    save(savename, 'unionPF', 'unionPFfit', 'firsttestFront', 'firsttrainFront'); % Save test set Pareto front

    % Output and save the First Pareto Front on training set
    fprintf('Outputting the First Pareto Front on Training Set:\n');
    disp(firsttrainFront);
end

toc;