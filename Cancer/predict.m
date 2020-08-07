function p = predict (theta1, theta2, X)
  
  m = size(X, 1);
  h1 = sigmoid([ones(m, 1) X] * theta1);
  h2 = sigmoid([ones(m, 1) h1] * theta2);
  
  p = h2(:) > 0.5;
  
endfunction
