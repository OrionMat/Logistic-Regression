clear;
close all;
clc;

load('digits.mat'); % data stored in arrays X, y

% important variables
num_labels = 10;    % 1 to 10
[m, n] = size(X);
X = [ones(m, 1) X]; % add bias
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
fprintf('Fitted training accuracy:  %f%%\n', mean(double(p == y)) * 100);

% randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), 2:end);
displayDigits(sel);




