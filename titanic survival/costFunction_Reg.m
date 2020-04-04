function [J, grad] = costFunction_Reg (theta, X, y, m, n, lambda)

  L = eye(n+1, n+1);
  L(1,1)=0;

  costReg = (lambda/(2*m))*theta(2:end,:)'*theta(2:end,:);
  gradReg = (lambda/m)*L*theta;
  
  J = -(1/m) * (y'*log(sigmoid(X*theta)) + (1 - y)'*log(1-sigmoid(X*theta))) + costReg;
  grad = (1/m) * X'*(sigmoid(X*theta)-y) + gradReg;
  
endfunction
