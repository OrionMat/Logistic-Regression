function retval = descisionBoundry_Nfeature (theta, degree)
  % grid range
  u = linspace(-1, 1.5, 100);
  v = linspace(-1, 1.5, 100);
  y = zeros(length(u), length(v));
  
  % evaluate y = theta*x over the grid
  for i = 1:length(u)
      for j = 1:length(v)
          y(i,j) = polynomialExpand_2feature(u(i), v(j), degree)*theta;
      end
  end
  
  % transpose y before calling contour
  y = y'; 

  % plot y = 0 (need to specify the range [0, 0])
  contour(u, v, y, [0, 0], 'm', 'LineWidth', 3)
endfunction