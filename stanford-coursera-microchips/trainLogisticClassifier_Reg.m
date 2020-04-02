function [theta, cost] = trainLogisticClassifier_Reg (X, y, m, n, lambda)
  % initialise theta and minimize cost function
  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [theta, cost] = fminunc(@(t)(costFunction_Reg(t, X, y, m, n, lambda)), initial_theta, options);

  % training accuracy
  probTrain = sigmoid(X*theta);
  predictionsTrain = round(probTrain);
  fprintf('Train Accuracy: %f%%\n', mean(double(predictionsTrain == y)) * 100);
endfunction