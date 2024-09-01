#*[-----------------------------------------------------------------------------------------------]*#
#*[ Objective : This R script detects changepoints for logit(beta_t) in a time series regression  ]*#
#*[             model with AR(1) errors using a penalized likelihood framework. The considered    ]*#
#*[             models for structural changes are: piecewise mean shifts (PMS), piecewise linear  ]*#
#*[             trend (PLT), and piecewise cubic spline (PCS). With GA estimated changepoints,    ]*#
#*[             we forecast beta_t for the next 28-days. This method is applied to bootstrap      ]*#
#*[             samples. Our changepoint random forests produce an ensemble forecast on beta_t.   ]*#
#*[ Updated   : August 17, 2024                                                                   ]*#
#*[ Developers: Jaechoul Lee                                                                      ]*#
#*[-----------------------------------------------------------------------------------------------]*#

# Setup directories
WD <- getwd()

WD.inp <- paste( WD, "/Documents/!1Research/Projects/Epidemiology/P05_FluSIR/Analysis/", sep="" ) # on Mac
WD.inp <- paste( WD, "/!1Research/Projects/Epidemiology/P05_FluSIR/Analysis/", sep="" )           # on Windows
WD.inp <- "H:/Documents/!1Research/Projects/Epidemiology/P05_FluSIR/Analysis/"          # on Windows

# Load required packages
library( dplyr )
library( ggplot2 )
library( GGally )
library( forecast )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-1: Read the beta_t average series and its predictors
#*[-----------------------------------------------------------------------------------------------]*#

# Read the estimated beta_t average series for the period from 2023-08-10 to 2023-10-28
df.y_all <- read.csv( file=paste( WD.inp, "data_04_average_beta.csv", sep="" ), header=TRUE )
colnames( df.y_all ) <- c( "time_0", "beta" )

# Read the covariate series for the period from 2023-08-10 to 2023-10-28
df.z_all <- read.csv( file=paste( WD.inp, "data_arizona_factors_test.csv", sep="" ), header=TRUE )
colnames( df.z_all ) <- c( "time_0", "date", "mean_temp", "precip_hours", "max_rel_humidity", 
                           "sun_durn", "wind_speed", "swave_radiat", "google_search" )

# Merge the two data frames
df.yz_all <- merge( df.y_all, df.z_all ) |>
                mutate( time_1 = time_0 + 1 ) |>    # [CAUTION] time_1 starts at 1 instead of 0
                mutate( sun_durn = sun_durn/1000 ) |>
                select( time_1, date, beta, mean_temp, precip_hours, max_rel_humidity,
                        sun_durn, wind_speed, swave_radiat, google_search )
head( df.yz_all )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-2: Preliminary EDA
#*[-----------------------------------------------------------------------------------------------]*#

# Extract beta_t and its predictors
df.yz_ini <- df.yz_all |>
                select( beta, mean_temp, precip_hours, max_rel_humidity, sun_durn, 
                        wind_speed, swave_radiat, google_search )
head( df.yz_ini )

# Correlation between beta_t and its predictors
cor( df.yz_ini )

dev.new( width=6, height=6 )
ggcorr( df.yz_ini )

# Scatterplot matrix
dev.new( width=6, height=6 )
ggpairs( df.yz_ini, upper=list( continuous="points" ), lower=list( continuous="cor" ) ) +
   theme_bw()

# Transform predictors
df.yz_fin <- df.yz_ini |>
                mutate( ln.precip_hours = log( precip_hours+1 ),
                        ln.google_search = log( google_search+1 ) ) |>
                select( beta, mean_temp, ln.precip_hours, max_rel_humidity, sun_durn, 
                        wind_speed, swave_radiat, ln.google_search )
head( df.yz_fin )

# Correlation between beta_t and its predictors
cor( df.yz_fin )

dev.new( width=6, height=6 )
ggcorr( df.yz_fin )

# Scatterplot matrix
dev.new( width=6, height=6 )
ggpairs( df.yz_fin, upper=list( continuous="points" ), lower=list( continuous="cor" ) ) +
   theme_bw()

#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-3: Determine a target beta_t series for the t_bgn:t_end period
#*[-----------------------------------------------------------------------------------------------]*#

# Set up a time period
t_bgn <- 1                                          # first day
t_end <- 80                                         # last day
n_fct <- 28                                         # number of days for forecasting

t_prd <- t_bgn:t_end                 # which( t_bgn <= df.yz_all$time_1 & df.yz_all$time_1 <= t_end )

# Format beta values to a time series
b.t <- ts( df.yz_fin$beta[ t_prd ], start=t_bgn, frequency=1 )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-4: Compute logit(beta_t) for the t_bgn:t_end period via generalized logistic function
#*[-----------------------------------------------------------------------------------------------]*#

# Compute logit(beta.t) with minimum and maximum values of beta.t
b.t_min <- 0.08                                     # mean(b.t)-6*sd(b.t)=0.0785; mean(b.t)-5*sd(b.t)=0.0854
b.t_max <- 0.16                                     # mean(b.t)+6*sd(b.t)=0.1612; mean(b.t)+5*sd(b.t)=0.1543

lb.t <- log( (b.t - b.t_min)/(b.t_max - b.t) )      # generalized logit with b.t_min as min and b.t_max as max

#*[-----------------------------------------------------------------------------------------------]*#
### Step 1-5: EDA for covariate series
#*[-----------------------------------------------------------------------------------------------]*#

# Create a data set for covariate series
z.t_all <- df.yz_fin[t_prd,] |>
              select( mean_temp, ln.precip_hours, max_rel_humidity,
                      sun_durn, wind_speed, swave_radiat, ln.google_search ) |>
              ts( start=t_bgn, frequency=1 )

#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-1: Define the procedures required for changepoint random forests
#*[-----------------------------------------------------------------------------------------------]*#

# Load the R code for GA changepoint detection with changepoint + AR(1) model and GA
source( file=paste( WD.inp, "lib_ga-PLT_v2-3.R", sep="" ) )

# Fit a "changepoints + AR(1)" model with a given changepoint configuration
fit.CPTar <- function( y, trd.type=c("mean","linear","quadratic","cubic"), cp, z=NULL ) {
                                             # y       : a binary series
                                             # trd.type: fitted model order
                                             # cp      : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z       : covariates
  if ( trd.type == "mean"   )    fm = fit.PMSar( y=y, cp=cp, z=z )
  if ( trd.type == "linear" )    fm = fit.PLTar( y=y, cp=cp, z=z )
  if ( trd.type == "quadratic" ) fm = fit.PQSar( y=y, cp=cp, z=z )
  if ( trd.type == "cubic"  )    fm = fit.PCSar( y=y, cp=cp, z=z )

  return( fm )                               # return the fitted model
}

# Estimate the trend with a "changepoints + AR(1)" model
trd.CPTar <- function( y, trd.type=c("mean","linear","quadratic","cubic"), cp, z=NULL ) {
                                             # y       : a binary series
                                             # trd.type: fitted model order
                                             # cp      : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z       : covariates
  if ( trd.type == "mean"   )    y_trd = trd.PMSar( y=y, cp=cp, z=z )
  if ( trd.type == "linear" )    y_trd = trd.PLTar( y=y, cp=cp, z=z )
  if ( trd.type == "quadratic" ) y_trd = trd.PQSar( y=y, cp=cp, z=z )
  if ( trd.type == "cubic"  )    y_trd = trd.PCSar( y=y, cp=cp, z=z )

  return( y_trd )                            # return the estimated trend
}

# Forecast with a "changepoints + AR(1)" model for next 28 days
fct.CPTar <- function( y, trd.type=c("mean","linear","quadratic","cubic"), cp, z=NULL, h=14 ) {
                                             # y       : a binary series
                                             # trd.type: fitted model order
                                             # cp      : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z       : covariates
                                             # h       : number of future values to forecast
  fm = fit.CPTar( y=y, trd.type=trd.type, cp=cp, z=z ) # fm : fitted model

  if ( trd.type == "mean"   )    y_fct = fct.PMSar( y=y, cp=cp, z=z, model=fm, h=h )
  if ( trd.type == "linear" )    y_fct = fct.PLTar( y=y, cp=cp, z=z, model=fm, h=h )
  if ( trd.type == "quadratic" ) y_fct = fct.PQSar( y=y, cp=cp, z=z, model=fm, h=h )
  if ( trd.type == "cubic"  )    y_fct = fct.PCSar( y=y, cp=cp, z=z, model=fm, h=h )

  return( y_fct )                            # return the forecasts
}

# Forecast - all procedures are combined
beta_forecast <- function( y, z=NULL, trd.type=c("mean","linear","quadratic","cubic"),
                           ic=c("BIC","MDL"), h=14, i ) {
                                                     # y       : a binary series
                                                     # z       : covariates
                                                     # trd.type: fitted model order
                                                     # ic      : penalty type for penalized log-likelihood 
                                                     # h       : number of future values to forecast
                                                     # i       : a seed number
  # Step 1: GA changepoint detection
  out_cpt = ga.cpt_ts( y=y, z=z, trd.type=trd.type, ic=ic, gen.size=150, max.itr=100, p.mut=0.05,
                       seed=10*(i-1)+543, is.graphic=FALSE, is.print=FALSE )

  ga.sol = out_cpt$solution                          # GA estimated changepoints (m; tau_1,...,tau_m)
  ga.val = out_cpt$val.sol[length(out_cpt$val.sol)]  # optimized value of penalized likelihood

  # Step 2: Trend prediction
  y_trd = trd.CPTar( y=y, trd.type=trd.type, cp=ga.sol, z=z )

  # Step 3: One-step ahead prediction
  fm = fit.CPTar( y=y, trd.type=trd.type, cp=ga.sol, z=z )
  y_prd = y - resid( fm )

  # Step 4: Forecasting
  y_fct = fct.CPTar( y=y, trd.type=trd.type, cp=ga.sol, z=z, h=h )

  # Return the results
  list( ga.out=out_cpt, ga.cpt=ga.sol, ga.val=ga.val, ic=ic, y_trd=y_trd, y_prd=y_prd, y_fct=y_fct )
}

#*[-----------------------------------------------------------------------------------------------]*#
### Step 2-2: Ensemble forecast via changepoint random forests
#*[-----------------------------------------------------------------------------------------------]*#

# References
# [1] Forecasting: Principles and Practice
#     https://otexts.com/fpp3/bootstrap.html (3rd edn); https://otexts.com/fpp2/bootstrap.html (2nd edn)
# [2] Bergmeir, C., R. J. Hyndman, and J. M. Benitez (2016).
#     Bagging exponential smoothing methods using STL decomposition and Box-Cox transformation.
#     International Journal of Forecasting, 32, 303-312.

# Define random forests settings
n_boot <- 600                                        # number of bootstrap samples
n_xprd <- ceiling( sqrt( ncol(z.t_all) ) )           # number of predictors selected for each tree
i_seed <- 21                                         # seed number for GA

# Generate bootstrap samples
#     The procedure is described in Bergmeir et al. (2016):
#     Box-Cox decomposition is applied, together with STL or Loess (for non-seasonal time series), and 
#     the remainder is bootstrapped using a moving block bootstrap.
set.seed( 100 + i_seed )
boot_sample <- bld.mbb.bootstrap( x=lb.t, num=n_boot, block_size=NULL ) |>
                 as.data.frame() |>
                 ts( start=t_bgn, frequency=1 )
dim( boot_sample )                                   # [1] 80 n_boot

# --------------------------------------------------------------------------------
# [Option 1] Perform changepoint random forests on lb.t using a single core
# --------------------------------------------------------------------------------
boot_cpt <- list()
boot_val <- rep( NA, times=n_boot )
boot_fct <- matrix( NA, nrow=n_fct, ncol=n_boot )

set.seed( 102 + i_seed )
for ( i_boot in 1:n_boot ) {
   print( paste("#-----[ Loop =",i_boot,"has begun at",Sys.time(),"]-----#"), quote=FALSE )

   y_boot = boot_sample[,i_boot]                     # moving-block bootstrap samlple
   z_indx = sort( sample( 1:ncol(z.t_all), size=n_xprd, replace=FALSE ) )  # index for selected predictors
   z_boot = z.t_all[,z_indx]                         # selected predictors

   fct_out = beta_forecast( y=y_boot, z=z_boot, trd.type="mean", ic="MDL", h=n_fct, i=i_seed+10*(i_boot-1) )

   boot_cpt[[i_boot]] = fct_out$ga.cpt               # GA estimated changepoints (m; tau_1,...,tau_m)
   boot_val[i_boot]  = fct_out$ga.val                # optimized value of penalized likelihood

#  boot_trd[,i_boot] = fct_out$y_trd
#  boot_prd[,i_boot] = fct_out$y_prd
   boot_fct[,i_boot] = fct_out$y_fct                 # forecasts
}
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# [Option 2] Ensemble forecast via changepoint random forests using multiple cores
# --------------------------------------------------------------------------------
# Load required packages for parallel computing
library( doParallel )                           # load doParallel with foreach, iterators, parallel
library( doSNOW )

# Set up parallel backend to use multiple cores
cores <- detectCores()                          # [1] 8 on MacBook Air M2
cl <- makeCluster( cores-3 )                    # use 5 cores, makeCluster( cores-2 )
cl <- makeCluster( 20 )                         # use 20 cores
registerDoSNOW( cl )                            # registerDoParallel( cl ) if doSNOW is not used

# Make a progress status bar
pb <- txtProgressBar( max=n_boot, style=3 )
progress <- function( n ) setTxtProgressBar( pb, n )
opts <- list( progress=progress )

# Record run time
print( paste( "#-----[ Changepoint random forests have begun at",Sys.time(),"]-----#" ), quote=FALSE )

# Perform changepoint random forests on lb.t using multiple cores
set.seed( 102 + i_seed )
ls.rf_out <- foreach( i_boot=1:n_boot, .packages='forecast', .options.snow=opts ) %dopar% {
   y_boot = boot_sample[,i_boot]                # moving-block bootstrap samlple
   z_indx = sort( sample( 1:ncol(z.t_all), size=n_xprd, replace=FALSE ) )  # index for selected predictors
   z_boot = z.t_all[,z_indx]                    # selected predictors

   fct_out = beta_forecast( y=y_boot, z=z_boot, trd.type="mean", ic="MDL", h=n_fct, i=i_seed+10*(i_boot-1) )

   boot_cpt = fct_out$ga.cpt                    # GA estimated changepoints (m; tau_1,...,tau_m)
   boot_val = fct_out$ga.val                    # optimized value of penalized likelihood

#  boot_trd = fct_out$y_trd
#  boot_prd = fct_out$y_prd
   boot_fct = fct_out$y_fct                     # forecasts

   list( boot_ID=i_boot, boot_cpt=boot_cpt, boot_val=boot_val, boot_fct=boot_fct )
}

# Record run time
print( paste( "#-----[ Changepoint random forests have ended at",Sys.time(),"]-----#" ), quote=FALSE )

# End parallel backend
close( pb )
stopCluster( cl )

# Save the GA estimated changepoint results of all stations
capture.output( ls.rf_out, file=paste( WD.inp, "prog4_RF-mean_parallel.txt", sep="" ) )
saveRDS( ls.rf_out, file=paste( WD.inp, "prog4_RF-mean_parallel.RDS", sep="" ) )

# Read the model fit results of all stations
ls.rf_out <- readRDS( file=paste( WD.inp, "prog4_RF-mean_parallel.RDS",sep="" ) )

# Extract the random forests results
boot_cpt <- list()
boot_val <- rep( NA, times=n_boot )
boot_fct <- matrix( NA, nrow=n_fct, ncol=n_boot )

for ( i_boot in 1:n_boot ) {
   boot_cpt[[i_boot]] = ls.rf_out[[i_boot]]$boot_cpt # GA estimated changepoints (m; tau_1,...,tau_m)
   boot_val[i_boot]  = ls.rf_out[[i_boot]]$boot_val  # optimized value of penalized likelihood
   boot_fct[,i_boot] = ls.rf_out[[i_boot]]$boot_fct  # forecasts
}
# --------------------------------------------------------------------------------

# Make time series formatted forecasts from bootstrap samples
lb.t_fct.boot <- boot_fct |>
                   as.data.frame() |> ts( start=t_end+1, frequency=1 )
dim( lb.t_fct.boot )                                 # [1] 28 n_boot

# Plot all forecasts
dev.new( width=10.5, height=4 )
autoplot( lb.t ) +
   autolayer( lb.t_fct.boot[,1:n_boot], col=TRUE ) +
   autolayer( lb.t, col=FALSE ) +
   ylab( "log( (b.t - b.t_max)/(b.t_max - b.t) )" ) +
   xlab( "Number of days from 2023-08-10 (Day 1)" ) +
   guides( col="none" ) +
   theme_bw()

# Backtransform via generalized logistic transformation
b.t_fct.boot = transf_logistic( x_logit=lb.t_fct.boot, x_min=b.t_min, x_max=b.t_max )

# Perform ensemble forecast
ens_f95 <- structure( list(
             mean  = ts( apply( b.t_fct.boot, 1, mean ), start=t_end+1 ),            # ensemble forecast
             lower = ts( apply( b.t_fct.boot, 1, quantile, prob=0.025 ), start=t_end+1 ),  # lower limit
             upper = ts( apply( b.t_fct.boot, 1, quantile, prob=0.975 ), start=t_end+1 ),  # upper limit
             level = 0.95),
           class = "forecast" )

dev.new( width=11, height=4 )
autoplot( b.t ) +
   autolayer( ens_f95, series="95% PI" ) +
   autolayer( b.t, col=TRUE, linewidth=0.5, series="PF b.t" ) +
   ylab( "Transmission rates" ) +
   xlab( "Number of days from 2023-08-10 (Day 1)" ) +
   guides( col=guide_legend( title="Forecasts") ) +
   theme_bw()


