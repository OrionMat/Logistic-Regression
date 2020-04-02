clear;
close all;
clc;

data = csvread('titanic_train.csv');
% first row of the data is the headers -> skip row 1
X = data(2:end, 1:end-1); 
y = data(2:end, end);
[m, n] = size(X);
% add bias
X = [ones(m, 1) X];
clear data

% regularized logistic regression
lambda = 0

% initial predictor
initial_theta = zeros(n + 1, 1);
[J, ~] = costFunction_Reg (initial_theta, X, y, m, n, lambda)
% training accuracy
probTrain = sigmoid(X*initial_theta);
predictionsTrain = round(probTrain);
fprintf('Train Accuracy: %f%%\n', mean(double(predictionsTrain == y)) * 100);

% trained predictor
[theta, cost] = trainLogisticClassifier_Reg(X, y, m, n, lambda)




% plots
plotData(X, y);