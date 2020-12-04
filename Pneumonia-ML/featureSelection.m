function pBest = featureSelection (X, y,...
                                   nnparams, ...
                                   input_layer, ...
                                   hidden_layer, p)
  accuracies = zeros(p, 1);
  
  for i = 1: p,
    X = X .^ i;
    accuracies(i, 1) = nnAccuracy(nnparams, input_layer, hidden_layer, X, y);
  endfor;
  
  plot(1:p, accuracies, 'r-');
  
  fprintf('%.3f\n', accuracies);
  

endfunction