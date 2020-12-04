function accuracy_val = nnAccuracy (nnparams, ...
                            input_layer, ...
                            hidden_layer, ...
                            X, y)
                            
  f1_score = 0.96;
                            
  theta1 = reshape(nnparams(1: (1 + input_layer) * hidden_layer), (1 + input_layer), hidden_layer);
  theta2 = reshape(nnparams(1 + (1 + input_layer) * hidden_layer: end), 1 + hidden_layer, 1);
  [m, n] = size(X);
  
  a1 = [ones(m, 1) X];
  z2 = a1 * theta1;
  a2 = [ones(m, 1) sigmoid(z2)];
  z3 = a2 * theta2;
  a3 = sigmoid(z3);
  
  %% predictions = zeros(17, 1);
  %% predictions = zeros(17, 1);
  Predict = a3 > f1_score;
  
  %% count = 0;
  %% for index = 1: 17,
  %%   count = count + 0.06;
  %%   predictions(index, 1) = mean((a3 > count) == y);
  %% endfor;
  
  accuracy_val = double(mean(Predict == y));

endfunction
