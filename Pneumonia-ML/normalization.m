function theta = normalization (X, y)
  theta = inv(X' * X) * X' * y;
  %% 401 x 5216 * 5216 x 401 = 401 x 401
  %% 401 x 401 * 401 x 5216 = 401 x 5216
  %% 401 x 5216 * 5216 x 1 = 401 x 1

endfunction
