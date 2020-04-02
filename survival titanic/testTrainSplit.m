function [Xtrain, ytrain, Xtest, ytest] = testTrainSplit (data, m, training_split)
  num_training = round(training_split * m); % ~ 75% training examples
  rand_idxs = randperm(m);

  training_idxs = rand_idxs(1:num_training);
  testing_idxs = rand_idxs(num_training+1:end);

  Xtrain = data(training_idxs, 1:end-1); 
  ytrain = data(training_idxs, end);
  Xtest = data(testing_idxs, 1:end-1); 
  ytest = data(testing_idxs, end); 
endfunction
