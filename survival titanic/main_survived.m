clear;
close all;
clc;

data = csvread('titanic_train.csv');
X = data(2:end, 1:end-1); % first row of the data is the headers -> skip row 1
y = data(2:end, end);
[m, n] = size(X);
X = [ones(m, 1) X]; % add bias
clear data

% regularized logistic regression
lambda = 0

% initial cost and accuracy
initial_theta = zeros(n + 1, 1);
[J, ~] = costFunction_Reg (initial_theta, X, y, m, n, lambda)
probTrain = sigmoid(X*initial_theta);
predictionsTrain = round(probTrain);
fprintf('Train Accuracy: %f%%\n', mean(double(predictionsTrain == y)) * 100);

% trained predictor
[theta, cost] = trainLogisticClassifier_Reg(X, y, m, n, lambda)

% make prediction
x = [1; 1; 1; 30; 0; 0; 9] % [bias; Pclass;	Sex; Age;	SibSp; Parch;	Fare]
[prediction, probability] = predict(x, theta)



% plot
plotData(X, y);