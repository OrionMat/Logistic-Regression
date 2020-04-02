function descisionBoundry_2feature (theta, X, y)
  % only two features (not including bias) -> only need two points to plot descision boundary

  x1_points = [min(X(:,2)),  max(X(:,2))];                    % for the two points choose max and min values of x1 -> longest line
  x2_points = (1/theta(3))*(-theta(2)*x1_points - theta(1));  % compute the decision boundary -> theta0 + theta1*x1 + theta2*x2 = 0

  plot(x1_points, x2_points, 'm', 'LineWidth', 3)   
  axis([-1 1.5 -1 1.5]) 
endfunction
