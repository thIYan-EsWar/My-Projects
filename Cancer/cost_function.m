function [J grad] = cost_function (X, y, theta, input_layers, hidden_layers, lambda)
  
  m = size(X, 1);
  J = 0;
  X = [ones(m, 1) X];
  
  theta1 = reshape(theta(1: (1 + input_layers) * hidden_layers), 
                    1 + input_layers, hidden_layers);
  theta2 = reshape(theta(1 + ((1 + input_layers) * hidden_layers): end),
                    hidden_layers + 1, 1);
                    
  theta1_grad = zeros(size(theta1));
  theta2_grad = zeros(size(theta2));  
                    
  a1 = X;  # 601 x 10
  z2 = a1 * theta1;     # 601 x 10 * 10 x 12
  g2 = sigmoid(z2);     # 601 x 12
  a2 = [ones(m, 1) g2]; # 601 x 13
  z3 = a2 * theta2;     # 601 x 13 * 13 x 1
  h = sigmoid(z3);
  
  J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h));
  
  for i = 1: m,
    
    a1 = X(i, :);
    z2 = a1 * theta1;
    g2 = sigmoid(z2);
    a2 = [1, g2];
    z3 = a2 * theta2;
    a3 = sigmoid(z3);
    
    del3 = a3 - y(i);
    del2 = theta2 * del3 .* sigmoid_gradient([1, z2]);
    
    del2 = del2(2: end);
    theta1_grad = theta1_grad + a1' * del2';
    theta2_grad = theta2_grad + a2' * del3;
  
  endfor

  theta1_grad = theta1_grad / m;
  theta2_grad = theta2_grad / m;
  
  t1 = theta1(:, 2:end);
  t2 = theta2(2: end);
  
  reg = (lambda / (2 * m)) * sum(sum(t1 .^ 2)) + (lambda / (2 * m)) * sum(t2 .^ 2);
  J = J + reg;
  
  theta1_grad(:, 2: end) = theta1_grad(:, 2: end) + (lambda / m) * theta1(:, 2: end);
  theta2_grad(2: end) = theta2_grad(2: end) + (lambda / m) * theta2(2: end);
  
  grad = [theta1_grad(:); theta2_grad(:)];
  
endfunction