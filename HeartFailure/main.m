clear; close; clc; 

%% ---------------------------data correction--------------------------- %%
% data = csvread('data.csv');

% [m, n] = size(data);
% perm_index = randperm(m);

% data_perm = zeros(m, n);
% j = 1;
% for i = perm_index,
% 
%   data_perm(j) = data(i);
%   j = j + 1;
  
% endfor;

% train = data(1:280, :);
% test = data(281:end, :);

% csvwrite('train_data.csv', train);
% csvwrite('test_data.csv', test);

%% ---------------------------data seggregation--------------------------- %%
data_train = csvread('train_data.csv');
data_test = csvread('test_data.csv');

X_train = data_train(1:end, 1:12);
y_train = data_train(1:end, 13);

[m_train, n_train] = size(X_train);

X_test = data_test(1:end, 1:12);
y_test = data_test(1:end, 13);

[m_test, n_test] = size(X_test);

% theta = randomInitialization(n_train + 1, 1);
% csvwrite('features.csv', theta);
theta = csvread('features.csv');

alpha = 0.01;
lambda = 0;

X_train_scaled = zeros(m_train, n_train);
for i = 1: n_train,
  X = X_train(:, i);
  X_train_scaled(:, i) = featureScaling(X, range(X));
  
endfor;

X_train_scaled = [ones(m_train, 1) X_train_scaled];
J = costFunction(X_train_scaled, y_train, theta, lambda);

fprintf('||----------------------------------------------||\n');
fprintf('        The initial cost is... %.3f\n', J);
fprintf('||----------------------------------------------||\n');
gradient = normalEquation(X_train_scaled, y_train, lambda);

fprintf('||-----------------Gradients---------------------||\n');
fprintf('                    %.3f\n', gradient);
fprintf('||----------------------------------------------||\n');

fprintf('||----------------------------------------------||\n');
fprintf('             Test results... %.3f\n', costFunction(X_train_scaled, y_train, gradient, lambda));
fprintf('||----------------------------------------------||\n');
                                 
X_test_scaled = zeros(m_test, n_test);
for i = 1: n_test,
  X = X_test(:, i);
  X_test_scaled(:, i) = featureScaling(X, range(X));
  
endfor;

g = sigmoid([ones(m_test, 1) X_test_scaled] * gradient);
fprintf('||------------------Results----------------------||\n');
fprintf('         Predicted values Actual Values\n');
fprintf('             %.3f        %.3f\n', g, y_test);
fprintf('||----------------------------------------------||\n');
g = g > 0.5;
accuracy = mean(double(g == y_test)) * 100;
fprintf('||----------------------------------------------||\n');
fprintf('            The accuracy is... %.3f\n', accuracy);
fprintf('||----------------------------------------------||\n');


