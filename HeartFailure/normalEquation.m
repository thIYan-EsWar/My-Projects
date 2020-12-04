function grad = normalEquation (X, y, lambda)
  mul = X' * X;

  [m, n] = size(X);
  lambda_mat = eye(n);
  lambda_mat(1, 1) = 0;
  lambda_mat = lambda * lambda_mat;
  
  grad = pinv(mul + lambda_mat) * X' * y;

endfunction
