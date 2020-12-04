function flattened_data = meanNormalization (X)
  flattened_data = (mean(X) - X)/range(X);
  
  flattened_data(:);

endfunction
