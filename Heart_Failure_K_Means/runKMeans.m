function [centroids, idx] = runKMeans (X, cluster_centroids, K, max_iter)
  if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 14;
  endif
  
  for i = 1: max_iter,
    fprintf('Iteration completed %d/%d\n', i, max_iter);
    idx = findClosestCentroids(X, cluster_centroids);
    centroids = findClusterCentroids(X, idx, K);
  endfor

endfunction
