clear;
close all;
clc;

load('digits.mat'); % data stored in arrays X, y


% randomly split into testing and training sets
m = size(X, 1);
training_split = 0.7;
[X, y, Xtest, ytest] = testTrainSplit([X y], m, training_split);

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

% test accuracy
mTest = length(ytest);
Xtest = [ones(mTest, 1) Xtest]; % add bias
probsTest = sigmoid(Xtest*allTheta');
[_, pTest] = max(probsTest, [], 2);
fprintf('Fitted test accuracy: %f%%\n', mean(double(pTest == ytest)) * 100);


##% randomly select 100 data points to display
##rand_indices = randperm(m);
##sel = X(rand_indices(1:100), 2:end);
##displayDigits(sel);

##failIdxs = find(pTest != ytest)
##displayDigits(Xtest(failIdxs(1:100), 2:end));

%  randomly go throgh example cases displaying the image and the prediction
rp = randperm(m);

for i = 1:m
  displayDigits(X(rp(i), 2:end));

  probsTrain = sigmoid(X(rp(i), :)*allTheta');
  [_, p] = max(probsTrain, [], 2);
  
  fprintf('Prediction: %d (digit %d)\n', p, mod(p, 10));
  
  % Pause with quit option
  s = input('paused - press enter to continue, q to exit:','s');
  if s == 'q'
    break
  end
end


