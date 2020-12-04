close; clear; clc;

data = csvread('Train_data.csv');

X = data(:, 1:400);
y = data(:, 401);

[m, n] = size(X);
lambda = 0;

X = X ./ 255.0;

%% =========================================================
theta = zeros(n + 1, 1);
cost = computeCost([ones(m, 1) X], y, theta, lambda);

fprintf('The initial cost of the model without theta initializtion... %.3f\n', cost);

%% Theta = initializeWeights(n + 1, 1);
%% cost = computeCost([ones(m, 1) X], y, Theta, lambda);

%% csvwrite('Initial_features.csv', Theta);

%% fprintf('The initial cost of the model with theta initializtion... %.3f\n', cost);

%% J = computeCost([ones(m, 1) X], y, Theta, 10);
%% fprintf('The initial cost of the model with theta initializtion and regression... %.3f\n', J);

%% theta = normalization([ones(m, 1) X], y);
%% csvwrite('Normal_Equation_weights.csv', theta);

theta = csvread('Normal_Equation_weights.csv');
cost = computeCost([ones(m, 1) X], y, theta, lambda);
fprintf('Final cost of the model using noraml equations...  %.3f\n', cost);

%% =========================================================
%% input_layer = n;
%% hidden_layer = 25.0;

%% theta1 = initializeWeights(input_layer + 1, hidden_layer);
%% theta2 = initializeWeights(hidden_layer + 1, 1);

%% nnparams = [theta1(:); theta2(:)];
%% csvwrite('NN_initial_feature.csv', nnparams);
%% nnparams = csvread('NN_initial_feature.csv');

%% cost = nnComputeCost(nnparams, input_layer, hidden_layer, X, y, lambda);
%% fprintf('The initial cost of the model using neural network... %.3f\n', cost);

%% cost_lambda = nnComputeCost(nnparams, input_layer, hidden_layer, X, y, 1);
%% fprintf('The initial cost of the model using neural network with regression... %.3f\n', cost_lambda);

%% costFunction = @(p) nnComputeCost(p, input_layer, hidden_layer, X, y, lambda);
%% options = optimset('MaxIter', 50);

%% [nn_params cost] = fmincg(costFunction, nnparams, options);

%% fprintf('The final cost of the model... %.3f\n', cost);










