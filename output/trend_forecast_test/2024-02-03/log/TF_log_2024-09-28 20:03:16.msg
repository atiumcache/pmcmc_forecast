Error in { : task 15 failed - "Lapack routine dgesv: system is exactly singular: U[1,1] = 0" Calls: %dopar% -> <Anonymous> 
Traceback: 
 [1] "3: stop(simpleError(msg, call = expr))"                                       
 [2] "2: e$fun(obj, substitute(ex), parent.frame(), e$data)"                        
 [3] "1: foreach(i_boot = 1:n_boot, .options.snow = opts) %dopar% {"                
 [4] "       library(forecast)"                                                     
 [5] "       library(dplyr)"                                                        
 [6] "       y_boot = boot_sample[, i_boot]"                                        
 [7] "       z_indx = sort(sample(1:ncol(z.t_all), size = n_xprd, replace = FALSE))"
 [8] "       z_boot = z.t_all[, z_indx]"                                            
 [9] "       fct_out = beta_forecast(y = y_boot, z = z_boot, trd.type = \"mean\", " 
[10] "           ic = \"MDL\", h = n_fct, i = i_seed + 10 * (i_boot - 1))"          
[11] "       boot_cpt = fct_out$ga.cpt"                                             
[12] "       boot_val = fct_out$ga.val"                                             
[13] "       boot_fct = fct_out$y_fct"                                              
[14] "       list(boot_ID = i_boot, boot_cpt = boot_cpt, boot_val = boot_val, "     
[15] "           boot_fct = boot_fct)"                                              
[16] "   }"                                                                         

