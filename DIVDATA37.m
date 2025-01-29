function [train_F, train_L, test_F, test_L] = DIVDATA37(dataName)
file = ['dataset/', dataName, '.mat'];
load(file)

dataMat = ins;
len = size(dataMat, 1);
% 归一化
maxV = max(dataMat);
minV = min(dataMat);
range = maxV - minV;
newdataMat = (dataMat - repmat(minV, [len, 1])) ./ (repmat(range, [len, 1]));

% 获取数据集的标签
lab = lab;

% 随机打乱数据集索引
indices = randperm(length(lab));

% 计算训练集和测试集的索引分割点
splitIdx = round(0.7 * length(lab)); % 70% for training

% 根据索引分割训练集和测试集
trainIdx = indices(1:splitIdx);
testIdx = indices(splitIdx+1:end);

train_F = newdataMat(trainIdx, :);
train_L = lab(trainIdx);
test_F = newdataMat(testIdx, :);
test_L = lab(testIdx);
end