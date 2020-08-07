close; clear; clc;

data = dlmread('train_data.txt');
X = data(:, 2: 10);
y = data(:, 11);
[m, n] = size(X);

for i = 1: m,
  if y(i) == 2,
    y(i) = 0;
  else,
    y(i) = 1;
  endif
endfor

input_layers = 9;
hidden_layers = 12;

Theta1 = random_initialize(input_layers, hidden_layers);
Theta2 = random_initialize(hidden_layers, 1);

Theta = [Theta1(:); Theta2(:)];

lambda = 0;
J = cost_function(X, y, Theta, input_layers, hidden_layers, lambda);
printf('The initial cost without regression... %.3f \n', J);

lambda = 5;
[J, grad] = cost_function(X, y, Theta, input_layers, hidden_layers, lambda);
printf('The initial cost with regression... %.3f \n', J);

lambda = 1;
costFunction = @(p)cost_function(X, y, p, input_layers ...
                                , hidden_layers, lambda);
                                
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, Theta, options);

grad1 = reshape(nn_params(1: (1 + input_layers) * hidden_layers), 
                  1 + input_layers, hidden_layers);
grad2 = reshape(nn_params(1 + ((1 + input_layers) * hidden_layers): end),
                  hidden_layers + 1, 1);

test = dlmread('test_data.txt');
X_test = test(:, 2: 10);
y_test = test(:, 11);
m_test = size(X_test, 1);

for i = 1: m_test,
  if y_test(i) == 2,
    y_test(i) = 0;
  else,
    y_test(i) = 1;
  endif
endfor
                  
pred = predict(grad1, grad2, X_test);
accuracy = mean(pred == y_test) * 100;