clear;
close all;
clc;

data = csvread('winequality-red.csv');
data = data(2:end, :); % first row of the data is the headers -> ignore row 1

% randomly split into testing and training sets
m = size(data, 1);
training_split = 0.7;
[X, y, Xtest, ytest] = testTrainSplit(data, m, training_split);
clear data

% important variables
num_labels = 10;                % 1 to 10
mTest = length(ytest);
[m, n] = size(X);
X = [ones(m, 1) X];             % add bias
Xtest = [ones(mTest, 1) Xtest]; % add bias
lambda = 0

% initial training accuracy
initial_allTheta = zeros(num_labels, n + 1);
probsTrain = sigmoid(X*initial_allTheta');
[_, p] = max(probsTrain, [], 2);
fprintf('Initial training accuracy: %f%%\n', mean(double(p == y)) * 100);

% train predictor
[allTheta] = trainMultiClassifier(X, y, m, n, lambda, num_labels);

% trained training accuracy
probsTrain = sigmoid(X*allTheta');
[_, p] = max(probsTrain, [], 2);
fprintf('Trained training accuracy: %f\n', mean(double(p == y)) * 100);



##% test accuracy
##probTest = sigmoid(Xtest*theta);
##predictionsTest = round(probTest);
##fprintf('Fitted Test Accuracy: %f%%\n', mean(double(predictionsTest == ytest)) * 100);

