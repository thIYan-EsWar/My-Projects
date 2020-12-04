close; clear; clc;

%% data = dlmread('Test_data.txt');
%% m = length(data);
%% data_perm = data;

%% count = 1;
%% for i = randperm(m),
%%   data_perm(count, :) = data(i, :);
%%   count = count + 1;
  
%% endfor;

%% csvwrite('Test_data.csv', data_perm);

data = csvread('Test_data.csv');
X = data(:, 1: 400);
y = data(:, 401);

input_layer = 400;
hidden_layer = 25;

X = X ./ 255.0;
nnparams = csvread('NN_Final_Param.csv');

[accuracy_val, predictions] = nnAccuracy(nnparams, input_layer, hidden_layer, X, y);
fprintf('Percentage accuracy of the model is... %.3f\n', accuracy_val * 100);

%% theta = csvread('Normal_Equation_weights.csv');

%% accuracy = normAccuracy(X, y, theta);
%% fprintf('Percentage accuracy of the model using normalization is... %.3f\n', accuracy * 100);

figure;
plot(1:17, predictions, 'r-');



