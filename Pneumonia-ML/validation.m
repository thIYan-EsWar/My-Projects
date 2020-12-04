close; clear; clc;

data = dlmread('Validation_data.txt');

X = data(:, 1:400);
y = data(:, 401);

X = X ./ 255.0;

nnparams = csvread('NN_Final_Param.csv');
input_layer = 400;
hidden_layer = 25;

featureSelection(X, y, nnparams, input_layer, hidden_layer, 8);




