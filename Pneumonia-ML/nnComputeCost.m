function [J grad] = nnComputeCost (nnparams, ...
                            input_layer, ...
                            hidden_layer, ...
                            X, y, lambda)
  m = length(y);
  J = 0.0;
    
  theta1 = reshape(nnparams(1: (input_layer + 1) * hidden_layer), input_layer + 1, hidden_layer);
  theta2 = reshape(nnparams(1 + (input_layer + 1) * hidden_layer : end), hidden_layer + 1, 1);
  
  theta1_grad = zeros(size(theta1));
  theta2_grad = zeros(size(theta2));
    
  a1 = [ones(m, 1) X];    %% 5216 x 401
  z2 = a1 * theta1;       %% 5216 x 401 * 401 x 25 = 5216 x 25
  a2 = sigmoid(z2);       
  a2 = [ones(m, 1) a2];   %% 5216 x 26
  z3 = a2 * theta2;       %% 5216 x 26 * 26 x 1 = 5216 x 1
  a3 = sigmoid(z3);
  
  h = a3;
  
  J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h));
  reg = (lambda / (2 * m)) * sum(sum(theta1 .^ 2) + sum(theta2 .^ 2));
  
  J = J + reg;
  
  for t= 1: m,
    a1 = [1, X(t, :)];        %% 1 x 401
    z2 = a1 * theta1;         %% 1 x 401 * 401 x 25 = 1 x 25
    a2 = [1, sigmoid(z2)];    %% 1 x 26
    z3 = a2 * theta2;         %% 1 x 26 * 26 x 1 = 1 x 1
    a3 = sigmoid(z3);
    
    del3 = a3 - y(t);         %% 1 x 1
    del2 = theta2 * del3 .* sigmoidGradient([1, z2])';%% 26 x 1 * 1 x 1 .* 26 x 1
    del2 = del2(2:end);
    
    theta2_grad = theta2_grad + a2' * del3; %% 26 x 1 + 26 x 1 * 1 x 1
    theta1_grad = theta1_grad + (del2 * a1)'; %% 401 x 25 + (25 x 1 * 1 x 401)'
  endfor;
  
  theta1_grad = theta1_grad / m;
  theta2_grad = theta2_grad / m;
  
  theta1_grad(:, 2:end) = theta1_grad(:, 2:end) + (lambda / (2 * m)) * theta1(:, 2:end);
  theta2_grad(2:end) = theta2_grad(2:end) + (lambda / (2 * m)) * theta2(2:end);
  
  grad = [theta1_grad(:); theta2_grad];

endfunction