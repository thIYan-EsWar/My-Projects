function theta = random_initialize (L_in, L_out)
  
  episilon = 0.1;
  theta = rand(L_out, 1 + L_in) * 2 * episilon - episilon;
  
  theta = theta';

endfunction
