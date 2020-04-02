function plotData(X, y)

  % find indices of positive and negative Examples
  pos = find(y == 1); 
  neg = find(y == 0);

  feat123 = [X(:,2), X(:,3), X(:,4)]

  Xpos = feat123(pos, :);
  Xneg = feat123(neg, :);

  figure 1;
  scatter3(Xpos(:, 1), Xpos(:, 2), Xpos(:, 3), 'r', '+');
  hold on
  scatter3(Xneg(:, 1), Xneg(:, 2), Xneg(:, 3), 'b');
  xlabel('Pclass');
  ylabel('Sex: male=1, female=0');
  zlabel('Age');

end
