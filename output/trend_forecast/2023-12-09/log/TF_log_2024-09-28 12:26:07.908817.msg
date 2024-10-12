Error in checkForRemoteErrors(lapply(cl, recvResult)) : 6 nodes produced errors; first error: object '.doSnowGlobals' not found Calls: %dopar% -> <Anonymous> -> clusterCall -> checkForRemoteErrors 
Traceback: 
[1] "5: stop(count, \" nodes produced errors; first error: \", firstmsg)"        
[2] "4: checkForRemoteErrors(lapply(cl, recvResult))"                            
[3] "3: clusterCall(cl, workerInit, xpr, exportenv, pkgname, obj$packages, "     
[4] "       attachExportEnv)"                                                    
[5] "2: e$fun(obj, substitute(ex), parent.frame(), e$data)"                      
[6] "1: foreach(i_boot = 1:5, .options.snow = opts, .packages = c(\"forecast\", "
[7] "       \"dplyr\")) %dopar% {"                                               
[8] "       print(\"test\")"                                                     
[9] "   }"                                                                       

