function theta = randomInitialization (m, n)
  init_epsilon = 0.12;
  theta = rand(m, n) * (2 * init_epsilon) - init_epsilon;

endfunction
