function Xpoly = polynomialExpand_2feature(X, degree)
  
  feat1 = X(:, 2);
  feat2 = X(:, 3);
  Xpoly(:,1) = X(:,1);
  
  for i = 1:degree
    for j = 0:i
      Xpoly(:, end+1) = (feat1.^(i-j)) .* (feat2.^j);
    endfor
  endfor
  
endfunction