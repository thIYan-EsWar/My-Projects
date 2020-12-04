function accuracy_ret = accuracy (idx, y)
  accuracy_ret = mean(double((idx - 1) == y)) * 100;

endfunction
