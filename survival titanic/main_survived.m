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
initial_theta = zeros(n + 1, 1);
[J, ~] = costFunction_Reg (initial_theta, X, y, m, n, lambda) % initial cost
[theta, cost] = trainLogisticClassifier_Reg(X, y, m, n, lambda)