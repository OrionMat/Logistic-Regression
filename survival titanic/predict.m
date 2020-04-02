function [prediction, proability] = predict (x, theta)
  proability = sigmoid(theta'*x);
  prediction = round(proability);
endfunction
