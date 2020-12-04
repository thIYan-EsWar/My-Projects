% ================= To clear Commandwindow ================= %
clc; clear; close;

% =============== Create train and test data =============== %
%data = csvread('heart.csv')(2:end, :);
%[m, n] = size(data);
%train_count = ceil(m * 0.90);

%idx = randperm(m);
%data_perm = zeros(m, n);
%count = 1;

%for i = idx,
%  data_perm(count, :) = data(i, :);
%  count++;
%endfor

%csvwrite('Train.csv', data_perm(1:train_count, :));
%csvwrite('Test.csv', data_perm(train_count + 1:end, :));

% =================== Load train data ===================== %
data = csvread('Train.csv');
X = data(:, 1:13);
y = data(:, 14);

[m, n] = size(X);
for i = 1: n,
  X(:, i) = meanNormalization(X(:, i));  
endfor

clusters = 2;
max_iter = 12;
epislon = 0.12;

% =================== ComputeCentroid ===================== %
%csvwrite('Cluster.csv', (rand(2, n) * 2 * epislon) - epislon);
init_centroids = csvread('Init_Cluster.csv');

idx = findClosestCentroids(X, init_centroids);
centroids = findClusterCentroids(X, idx, clusters);
fprintf('Initial accuracy is %.3f...\n', accuracy(idx, y));

[centroids, idx] = runKMeans(X, centroids, clusters, 50);
fprintf('Centroids are %.3f            %.3f\n', centroids, init_centroids);
fprintf('The accuracy... %.3f\n', accuracy(idx, y));


