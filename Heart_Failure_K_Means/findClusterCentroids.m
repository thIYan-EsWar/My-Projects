function centroids = findClusterCentroids (X, idx, K)
  [m, n] = size(X);
  centroids = zeros(K, n);
  
  for k = 1: K,
    centroids(k, :) = mean(X(idx == k, :));
  endfor

endfunction
