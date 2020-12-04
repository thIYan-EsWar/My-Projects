function J = costFunction (X, y, theta, lambda)
  [m, n] = size(X);
  
  z = X * theta;
  g = sigmoid(z);
  
  J = (-1 / (2 * m)) * sum(y .* log(g) + (1 - y) .* (log(1 - g)));
  reg = (lambda / (2 * m)) * sum(theta .^ 2);
  
  J = J + reg;

endfunction
