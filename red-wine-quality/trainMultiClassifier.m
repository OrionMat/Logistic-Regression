function [allTheta] = trainMultiClassifier(X, y, m , n, lambda, num_labels)
  
  initial_theta = zeros(n+1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 500);
  allTheta = zeros(num_labels, n + 1);

  for c = 1:num_labels
    [theta, _, _] = fmincg(@(t)(LRcostFunction(t, X, (y == c), m, n, lambda)), initial_theta, options);
    allTheta(c,:) = theta;
  end

end
