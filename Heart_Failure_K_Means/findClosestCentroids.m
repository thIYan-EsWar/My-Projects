function cluster_index = findClosestCentroids (X, centroids)
  [m, n] = size(X);
  K = size(centroids, 1);
  
  cluster_index = zeros(m, 1);
  J = zeros(size(centroids), 1);
  
  for i = 1: m,
    for j = 1: K,
      J(j) = sqrt(sum((X(i, :) - centroids(j, :)).^2));
    endfor
    [mini, min_index] = min(J);
    cluster_index(i, 1) = min_index;
  endfor
  
  cluster_index(:);
endfunction
