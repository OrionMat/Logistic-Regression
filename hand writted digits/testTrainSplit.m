function [Xtrain, ytrain, Xtest, ytest] = testTrainSplit (data, m, training_split)
  num_training = round(training_split * m);
  rand_idxs = randperm(m);

  training_idxs = rand_idxs(1:num_training);
  testing_idxs = rand_idxs(num_training+1:end);

  Xtrain = data(training_idxs, 1:end-1); 
  ytrain = data(training_idxs, end);
  Xtest = data(testing_idxs, 1:end-1); 
  ytest = data(testing_idxs, end); 
  
endfunction


% randomly split into testing and training sets
##m = size(X, 1);
##training_split = 0.7;
##[X, y, Xtest, ytest] = testTrainSplit(X, m, training_split);
##mTest = length(ytest);
##Xtest = [ones(mTest, 1) Xtest]; % add bias
##% test accuracy
##probsTest = sigmoid(Xtest*allTheta');
##[_, pTest] = max(probsTest, [], 2);
##fprintf('Fitted test accuracy: %f%%\n', mean(double(pTest == ytest)) * 100);