function accuracy = normAccuracy (X, y, theta)
  [m, n] = size(X);
  f1_score = 0.5;
  
  g = sigmoid([ones(m, 1) X] * theta);
  predict = g > f1_score;
  
  accuracy = double(mean(predict == y));

endfunction
