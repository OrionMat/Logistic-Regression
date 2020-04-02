function Xpoly = polynomialExpand_2feature(feat1, feat2, degree)
  
  Xpoly = ones(size(feat1(:,1)));
  
  for i = 1:degree
    for j = 0:i
      Xpoly(:, end+1) = (feat1.^(i-j)) .* (feat2.^j);
    endfor
  endfor
  
endfunction