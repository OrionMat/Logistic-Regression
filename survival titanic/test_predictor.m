clear;
close all;
clc;

data = csvread('titanic_train.csv');
data = data(2:end, :); % first row of the data is the headers -> ignore row 1

% randomly split into testing and training sets
m = size(data, 1);
training_split = 0.75;
[Xtrain, ytrain, Xtest, ytest] = testTrainSplit(data, m, training_split);
clear data

% regularized logistic regression
[m, n] = size(Xtrain);
mTest = length(ytest);
Xtrain = [ones(m, 1) Xtrain]; % add bias
Xtest = [ones(mTest, 1) Xtest]; % add bias
lambda = 0






% initial cost and training accuracy
initial_theta = zeros(n + 1, 1);
[J, ~] = costFunction_Reg (initial_theta, Xtrain, ytrain, m, n, lambda)
probTrain = sigmoid(Xtrain*initial_theta);
predictionsTrain = round(probTrain);
fprintf('Initial Train Accuracy: %f%%\n', mean(double(predictionsTrain == ytrain)) * 100);


% train predictor
[theta, cost] = trainLogisticClassifier_Reg(Xtrain, ytrain, m, n, lambda)


% trained cost, training and test accuracy
probTrain = sigmoid(Xtrain*theta);
predictionsTrain = round(probTrain);
fprintf('Fitted Train Accuracy: %f%%\n', mean(double(predictionsTrain == ytrain)) * 100);

probTest = sigmoid(Xtest*theta);
predictionsTest = round(probTest);
fprintf('Fitted Test Accuracy: %f%%\n', mean(double(predictionsTest == ytest)) * 100);

