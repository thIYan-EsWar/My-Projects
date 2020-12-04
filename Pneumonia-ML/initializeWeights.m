function Theta = initializeWeights (m, n)
  
  init_epsilon = 0.12;
  Theta = rand(m, n) * 2 * init_epsilon - init_epsilon;

endfunction
