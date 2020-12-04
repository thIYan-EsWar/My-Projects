function J = computeCost (X, y, theta, lambda)
  
  [m, n] = size(X);
  J = 0.0;
  
  h = sigmoid(X * theta);
  J = (1.0 / m) * sum(-y .* log(h) - (1.0 - y) .* log(1.0 - h));
  
  reg = (lambda / (2 * m)) * sum(theta .^ 2);
  J = J  + reg;

endfunction
