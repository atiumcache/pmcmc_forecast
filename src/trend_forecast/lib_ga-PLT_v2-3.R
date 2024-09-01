#*[-----------------------------------------------------------------------------------------------]*#
#*[ Objective : This R library contains the functions to detect changepoints in time series       ]*#
#*[             regressoin models.                                                                ]*#
#*[ Updated   : August 7, 2024                                                                    ]*#
#*[ Authors   : Jaechoul Lee                                                                      ]*#
#*[-----------------------------------------------------------------------------------------------]*#

# --------
# Abbreviations
# --------
#   PMS - piecewise mean shifts
#   PLT - piecewise linear trend
#   PQS - piecewise quadratic spline         [!CAUTION!] Not working or unable to fit correctly
#   PCS - piecewise cubic spline

# --------
# Function to transform logit to logistic
# --------
transf_logistic <- function( x_logit, x_min, x_max ) {
  x_lgst <- x_min + (x_max - x_min)/(1 + exp(-x_logit))

  return( x_lgst )                           # return the generalized logistic transformed values
}

# --------
# Function to fit a "PMS + AR(1)" model with a given changepoint configuration
# --------
fit.PMSar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  t = 1:n                                    # t    : times

  if ( m == 0 ) {
    ar_PMS = arima( y, order=c(1,0,0), xreg=z, include.mean=TRUE, method="ML" )
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = 1*( t >= tau[k] )
    }
    ar_PMS = arima( y, order=c(1,0,0), xreg=cbind(D_nrm,z), include.mean=TRUE, method="ML" )
  }

  return( ar_PMS )                           # return fitted "PMS + AR(1)" model
}

# --------
# Function to fit a "PLT + AR(1)" model with a given changepoint configuration
# --------
fit.PLTar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  c_nrm = 100                                # c_nrm: time normalization constant, 100 days
                                             #      : [!CAUTION!] adjust for other settings
  t = 1:n                                    # t    : times
  t_nrm = t/c_nrm                            # t_nrm: normalized times per 100 days
  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  if ( m == 0 ) {
    ar_PLT = arima( y, order=c(1,0,0), xreg=cbind(t_nrm,z), include.mean=TRUE, method="ML" )
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = pmax( t_nrm-tau_nrm[k], 0 )
    }
    ar_PLT = arima( y, order=c(1,0,0), xreg=cbind(t_nrm,D_nrm,z), include.mean=TRUE, method="ML" )
  }

  return( ar_PLT )                           # return fitted "PLT + AR(1)" model
}

# --------
# Function to fit a "PQS + AR(1)" model with a given changepoint configuration
# --------
fit.PQSar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  c_nrm = 100                                # c_nrm: time normalization constant, 100 days
                                             #      : [!CAUTION!] adjust for other settings
  t = 1:n                                    # t    : times
  t1_nrm = t/c_nrm                           # t1_nrm: normalized times per 100 days
  t2_nrm = t1_nrm^2

  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  if ( m == 0 ) {
    ar_PQS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,z), include.mean=TRUE, method="ML" )
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = pmax( (t1_nrm - tau_nrm[k])^2, 0 )
    }
    ar_PQS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,D_nrm,z), include.mean=TRUE, method="ML" )
  }

  return( ar_PQS )                           # return fitted "PQS + AR(1)" model
}

# --------
# Function to fit a "PCS + AR(1)" model with a given changepoint configuration
# --------
fit.PCSar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  c_nrm = 100                                # c_nrm: time normalization constant, 100 days
                                             #      : [!CAUTION!] adjust for other settings
  t = 1:n                                    # t    : times
  t1_nrm = t/c_nrm                           # t1_nrm: normalized times per 100 days
  t2_nrm = t1_nrm^2
  t3_nrm = t1_nrm^3

  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  if ( m == 0 ) {
    ar_PCS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,t3_nrm,z), include.mean=TRUE, method="ML" )
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = pmax( (t1_nrm - tau_nrm[k])^3, 0 )
    }
    ar_PCS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,t3_nrm,D_nrm,z), include.mean=TRUE, method="ML" )
  }

  return( ar_PCS )                           # return fitted "PCS + AR(1)" model
}

# --------
# Function to compute the mean shifts of a "PMS + AR(1)" model with a given changepoint configuration
# --------
trd.PMSar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  t = 1:n                                    # t    : times

  if ( m == 0 ) {
    ar_PMS = arima( y, order=c(1,0,0), xreg=z, include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PMS )[-1]            # remove AR(1) parameter estimate
    y_trd.m = rep( 0, times=n )
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = 1*( t >= tau[k] )
    }
    ar_PMS = arima( y, order=c(1,0,0), xreg=cbind(D_nrm,z), include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PMS )[-1]            # remove AR(1) parameter estimate

    y_trd.m = rep( 0, times=n )
    for ( k in 1:m ) {
      y_trd.m = y_trd.m + coef_reg[1+k]*D_nrm[,k]
    }
  }

  y_trd = coef_reg[1] + y_trd.m + 
          ifelse( rep( is.null(z), times=n ), 0, z%*%coef_reg[(1+m+1):(1+m+ncol(z))] )

  return( y_trd )                            # return mean shifts of fitted "PMS + AR(1)" model
}

# --------
# Function to compute the trend line of a "PLT + AR(1)" model with a given changepoint configuration
# --------
trd.PLTar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  c_nrm = 100                                # c_nrm: time normalization constant, 100 days
                                             #      : [!CAUTION!] adjust for other settings
  t = 1:n                                    # t    : times
  t_nrm = t/c_nrm                            # t_nrm: normalized times per 100 days
  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  if ( m == 0 ) {
    ar_PLT = arima( y, order=c(1,0,0), xreg=cbind(t_nrm,z), include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PLT )[-1]            # remove AR(1) parameter estimate
    y_trd.m = 0
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = pmax( t_nrm-tau_nrm[k], 0 )
    }
    ar_PLT = arima( y, order=c(1,0,0), xreg=cbind(t_nrm,D_nrm,z), include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PLT )[-1]            # remove AR(1) parameter estimate

    y_trd.m = rep( 0, times=n )
    for ( k in 1:m ) {
      y_trd.m = y_trd.m + coef_reg[2+k]*D_nrm[,k]
    }
  }

  y_trd = coef_reg[1] + coef_reg[2]*t_nrm + y_trd.m +
          ifelse( rep( is.null(z), times=n ), 0, z%*%coef_reg[(2+m+1):(2+m+ncol(z))] )

  return( y_trd )                            # return trend line of fitted "PLT + AR(1)" model
}

# --------
# Function to compute the quadratic spline of a "PQS + AR(1)" model with a given changepoint configuration
# --------
trd.PQSar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  c_nrm = 100                                # c_nrm: time normalization constant, 100 days
                                             #      : [!CAUTION!] adjust for other settings
  t = 1:n                                    # t    : times
  t1_nrm = t/c_nrm                           # t1_nrm: normalized times per 100 days
  t2_nrm = t1_nrm^2

  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  if ( m == 0 ) {
    ar_PQS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,z), include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PQS )[-1]            # remove AR(1) parameter estimate
    y_trd.m = 0
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = pmax( (t1_nrm - tau_nrm[k])^2, 0 )
    }
    ar_PQS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,D_nrm,z), include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PQS )[-1]            # remove AR(1) parameter estimate

    y_trd.m = rep( 0, times=n )
    for ( k in 1:m ) {
      y_trd.m = y_trd.m + coef_reg[3+k]*D_nrm[,k]
    }
  }

  y_trd = coef_reg[1] + coef_reg[2]*t1_nrm + coef_reg[3]*t2_nrm + y_trd.m +
          ifelse( rep( is.null(z), times=n ), 0, z%*%coef_reg[(3+m+1):(3+m+ncol(z))] )

  return( y_trd )                            # return quadratic spline of fitted "PQS + AR(1)" model
}

# --------
# Function to compute the cubic spline of a "PCS + AR(1)" model with a given changepoint configuration
# --------
trd.PCSar <- function( y, cp, z=NULL ) {     # y    : data
                                             # cp   : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z    : covariates
  n = length( y )                            # n    : series length, including missing values
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : time locations of changepoints

  c_nrm = 100                                # c_nrm: time normalization constant, 100 days
                                             #      : [!CAUTION!] adjust for other settings
  t = 1:n                                    # t    : times
  t1_nrm = t/c_nrm                           # t1_nrm: normalized times per 100 days
  t2_nrm = t1_nrm^2
  t3_nrm = t1_nrm^3

  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  if ( m == 0 ) {
    ar_PCS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,t3_nrm,z), include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PCS )[-1]            # remove AR(1) parameter estimate
    y_trd.m = 0
  } else {
    D_nrm = matrix( 0, nrow=n, ncol=m )
    for ( k in 1:m ) {
      D_nrm[,k] = pmax( (t1_nrm - tau_nrm[k])^3, 0 )
    }
    ar_PCS = arima( y, order=c(1,0,0), xreg=cbind(t1_nrm,t2_nrm,t3_nrm,D_nrm,z), include.mean=TRUE, method="ML" )
    coef_reg = coef( ar_PCS )[-1]            # remove AR(1) parameter estimate

    y_trd.m = rep( 0, times=n )
    for ( k in 1:m ) {
      y_trd.m = y_trd.m + coef_reg[4+k]*D_nrm[,k]
    }
  }

  y_trd = coef_reg[1] + coef_reg[2]*t1_nrm + coef_reg[3]*t2_nrm + coef_reg[4]*t3_nrm + y_trd.m +
          ifelse( rep( is.null(z), times=n ), 0, z%*%coef_reg[(4+m+1):(4+m+ncol(z))] )

  return( y_trd )                            # return cubic spline of fitted "PCS + AR(1)" model
}

# --------
# Function to forecast each covariate series using an NNAR model
# --------
nnar_forecast <- function( z, h=14 ) {       # z  : covariates
                                             # h  : number of future values to forecast
  z_fct = NULL
  for ( j in 1:ncol( z ) ) {                 # repeat for each covariate series
  # set.seed( 123 + (j-1)*10 )               # not used for more variability introduced in z_fct
    z.t = z[,j]

    fit.nnar = nnetar( z.t )                 # NNAR and use 'xreg= ' to include numerical covariates for NNAR
    z.t_fct = predict( fit.nnar, h=h )$mean
  # z.t_fct = forecast( fit.nnar, h=n_fct, PI=FALSE )$mean  # produce same result

    z_fct = cbind( z_fct, z.t_fct )
  }

  return( z_fct )                            # return fitted NNAR model
}

# --------
# Function to predict future beta_t with a "PMS + AR(1)" model
# --------
fct.PMSar <- function( y, cp, z=NULL, model, h=14 ) {
                                             # y     : data
                                             # cp    : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z     : covariates
                                             # model : fitted model
                                             # h     : number of future values to forecast
  n = length( y )                            # n     : series length, including missing values
  m = cp[1]                                  # m     : number of changepoints
  tau = cp[-1]                               # tau   : time locations of changepoints

  t = 1:n                                    # t     : times
  t.all = 1:(n+h)

  if ( is.null(z) ) {
    z.fct = NULL
  } else {
    z.fct = nnar_forecast( z=z, h=h )        # forecast each covariate series using NNAR
  }

  if ( m == 0 ) {
    ar_PMS.prd = predict( model, n.ahead=h, newxreg=z.fct )
  } else {
    D_nrm.all = matrix( 0, nrow=n+h, ncol=m )
    for ( k in 1:m ) {
      D_nrm.all[,k] = 1*( t.all >= tau[k] )
    }
    D_nrm = D_nrm.all[1:n,]
    D_nrm.new = D_nrm.all[(n+1):(n+h),]
    ar_PMS.prd = predict( model, n.ahead=h, newxreg=cbind(D_nrm.new,z.fct) )
  }

  return( ar_PMS.prd$pred )                  # return forecasts from the fitted model
}

# --------
# Function to predict future beta_t with a "PLT + AR(1)" model
# --------
fct.PLTar <- function( y, cp, z=NULL, model, h=14 ) {
                                             # y     : data
                                             # cp    : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z     : covariates
                                             # model : fitted model
                                             # h     : number of future values to forecast
  n = length( y )                            # n     : series length, including missing values
  m = cp[1]                                  # m     : number of changepoints
  tau = cp[-1]                               # tau   : time locations of changepoints

  c_nrm = 100                                # c_nrm : time normalization constant, 100 days
                                             #       : [!CAUTION!] adjust for other settings
  t = 1:n                                    # t     : times
  t_nrm = t/c_nrm                            # t_nrm : normalized times per 100 days
  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  t_nrm.all = (1:(n+h))/c_nrm
  t_nrm.new = ((n+1):(n+h))/c_nrm

  if ( is.null(z) ) {
    z.fct = NULL
  } else {
    z.fct = nnar_forecast( z=z, h=h )        # forecast each covariate series using NNAR
  }

  if ( m == 0 ) {
    ar_PLT.prd = predict( model, n.ahead=h, newxreg=cbind(t_nrm.new,z.fct) )
  } else {
    D_nrm.all = matrix( 0, nrow=n+h, ncol=m )
    for ( k in 1:m ) {
      D_nrm.all[,k] = pmax( t_nrm.all-tau_nrm[k], 0 )
    }
    D_nrm = D_nrm.all[1:n,]
    D_nrm.new = D_nrm.all[(n+1):(n+h),]
    ar_PLT.prd = predict( model, n.ahead=h, newxreg=cbind(t_nrm.new,D_nrm.new,z.fct) )
  }

  return( ar_PLT.prd$pred )                  # return forecasts from the fitted model
}

# --------
# Function to predict future beta_t with a "PQS + AR(1)" model
# --------
fct.PQSar <- function( y, cp, z=NULL, model, h=14 ) {
                                             # y     : data
                                             # cp    : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z     : covariates
                                             # model : fitted model
                                             # h     : number of future values to forecast
  n = length( y )                            # n     : series length, including missing values
  m = cp[1]                                  # m     : number of changepoints
  tau = cp[-1]                               # tau   : time locations of changepoints

  c_nrm = 100                                # c_nrm : time normalization constant, 100 days
                                             #       : [!CAUTION!] adjust for other settings
  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  t1_nrm = (1:n)/c_nrm                       # t1_nrm: normalized times per 100 days
  t2_nrm = t1_nrm^2

  t1_nrm.new = ((n+1):(n+h))/c_nrm
  t2_nrm.new = t1_nrm.new^2

  t1_nrm.all = c( t1_nrm, t1_nrm.new )
  t2_nrm.all = c( t2_nrm, t2_nrm.new )

  if ( is.null(z) ) {
    z.fct = NULL
  } else {
    z.fct = nnar_forecast( z=z, h=h )        # forecast each covariate series using NNAR
  }

  if ( m == 0 ) {
    ar_PQS.prd = predict( model, n.ahead=h,
                          newxreg=cbind(t1_nrm.new,t2_nrm.new,z.fct) )
  } else {
    D_nrm.all = matrix( 0, nrow=n+h, ncol=m )
    for ( k in 1:m ) {
      D_nrm.all[,k] = pmax( (t1_nrm.all-tau_nrm[k])^2, 0 )
    }
    D_nrm = D_nrm.all[1:n,]
    D_nrm.new = D_nrm.all[(n+1):(n+h),]
    ar_PQS.prd = predict( model, n.ahead=h,
                          newxreg=cbind(t1_nrm.new,t2_nrm.new,D_nrm.new,z.fct) )
  }

  return( ar_PQS.prd$pred )                  # return forecasts from the fitted model
}

# --------
# Function to predict future beta_t with a "PCS + AR(1)" model
# --------
fct.PCSar <- function( y, cp, z=NULL, model, h=14 ) {
                                             # y     : data
                                             # cp    : changepoint chromosome (m; tau_1,...,tau_m)
                                             # z     : covariates
                                             # model : fitted model
                                             # h     : number of future values to forecast
  n = length( y )                            # n     : series length, including missing values
  m = cp[1]                                  # m     : number of changepoints
  tau = cp[-1]                               # tau   : time locations of changepoints

  c_nrm = 100                                # c_nrm : time normalization constant, 100 days
                                             #       : [!CAUTION!] adjust for other settings
  tau_nrm = tau/c_nrm                        # tau_nrm: normalized changepoint times per 100 days

  t1_nrm = (1:n)/c_nrm                       # t1_nrm: normalized times per 100 days
  t2_nrm = t1_nrm^2
  t3_nrm = t1_nrm^3

  t1_nrm.new = ((n+1):(n+h))/c_nrm
  t2_nrm.new = t1_nrm.new^2
  t3_nrm.new = t1_nrm.new^3

  t1_nrm.all = c( t1_nrm, t1_nrm.new )
  t2_nrm.all = c( t2_nrm, t2_nrm.new )
  t3_nrm.all = c( t3_nrm, t3_nrm.new )

  if ( is.null(z) ) {
    z.fct = NULL
  } else {
    z.fct = nnar_forecast( z=z, h=h )        # forecast each covariate series using NNAR
  }

  if ( m == 0 ) {
    ar_PCS.prd = predict( model, n.ahead=h,
                          newxreg=cbind(t1_nrm.new,t2_nrm.new,t3_nrm.new,z.fct) )
  } else {
    D_nrm.all = matrix( 0, nrow=n+h, ncol=m )
    for ( k in 1:m ) {
      D_nrm.all[,k] = pmax( (t1_nrm.all-tau_nrm[k])^3, 0 )
    }
    D_nrm = D_nrm.all[1:n,]
    D_nrm.new = D_nrm.all[(n+1):(n+h),]
    ar_PCS.prd = predict( model, n.ahead=h,
                          newxreg=cbind(t1_nrm.new,t2_nrm.new,t3_nrm.new,D_nrm.new,z.fct) )
  }

  return( ar_PCS.prd$pred )                  # return forecasts from the fitted model
}


#  if ( m == 0 ) {
#    ar_PLT.prd = predict( model, n.ahead=n.ahead, newxreg=t_nrm.new )
#  } else {
#    D_nrm.all = matrix( 0, nrow=n+n.ahead, ncol=m )
#    for ( k in 1:m ) {
#      D_nrm.all[,k] = pmax( t_nrm.all-tau_nrm[k], 0 )
#    }
#    D_nrm = D_nrm.all[1:n,]
#    D_nrm.new = D_nrm.all[(n+1):(n+n.ahead),]
#    ar_PLT.prd = predict( model, n.ahead=n.ahead, newxreg=cbind(t_nrm.new,D_nrm.new) )
#  }


# --------
# Function to compute the BIC penalty
# --------
penalty_BIC <- function( y, model, cp ) {    # y     : data
                                             # model : fitted model
  n_obs = sum( !is.na( y ) )                 # n_obs : number of non-missing values in y
  m = cp[1]                                  # m     : number of changepoints

  n_par = length( model$coef ) + 1 + m       # n_par : number of parameters in fitted model & changepoint times
# n_par = length( model$coef ) + 1           # NOTE  : length(model$coef)+1 is incorrect if changepoint exists

  pnt = n_par*log( n_obs )                   # BIC penalty (modified for missing values)

  return( pnt )                              # return BIC penalty
}

# --------
# Function to compute the MDL penalty
# --------
penalty_MDL <- function( y, cp ) {           # y   : data
                                             # cp  : changepoint chromosome (m; tau_1,...,tau_m)
  n = length( y )                            # n   : series length, including missing values
  m = cp[1]                                  # m   : number of changepoints

  if ( m == 0 ) {
    pnt = 0                                  # MDL penalty if no changepoints
  } else {
    tau.ext = c( cp[-1], n+1 )               # tau.ext: changepoint times and n+1 (tau_1,...,tau_m,n+1)
    n.r = numeric( length=m )                # n.r    : number of non-missing observations in each regime
    for ( i in 1:m ) {
      n.r[i] = max( sum( !is.na( y[tau.ext[i]:(tau.ext[i+1]-1)] ) ), 1 ) # n.r = 1 if no data
    }
    pnt = log(m+1) + 0.5*sum( log(n.r) ) + sum( log(tau.ext[-1]) ) # MDL penalty if changepoint exists
  }

  return( pnt )                              # return MDL penalty
}

# --------
# Function to check the minimum changepoint distance condition for GA
# --------
check_distance <- function( cp, n, d.min ) { # cp   : changepoint configuration (m; tau_1,...,tau_m)
                                             # n    : series length, including missing values
                                             # d.min: minimum distance between two consecutive changepoint times
  m = cp[1]                                  # m    : number of changepoints
  tau = cp[-1]                               # tau  : changepoint times

  is_dist = ( m == 0 | all( diff( c(1,tau,n+1) ) >= d.min ) ) # min distance check: 2 consecutive times >= d.min

  return( is_dist )                          # return the check result
}

# --------
# Function to produce a child chromosome for GA
# --------
produce_child.cont <- function( ch_dad, ch_mom, tau.min, tau.max, p.mut ) {
                                                  # ch_dad: changepoint configuration of father chromosome
                                                  # ch_mom: changepoint configuration of mother chromosome
                                                  # tau.min: min time location of changepoints
                                                  # tau.max: max time location of changepoints
                                                  # p.mut : probability of a mutation
  # Step 0: Changepoint configuration
  t_inc = 1                                       # t_inc : time location increament used in Step 3

  # Step 1: Combining
  tau_S1 = sort( union( ch_dad[-1],ch_mom[-1] ) ) # discard identical changepoint times
  m_S1 = length( tau_S1 )

  if ( m_S1 == 0 ) {
    # Step 2: Thinning (SKIP!!!)
    # Step 3: Shifting (SKIP!!!)
    # Step 4: Mutation
    m_S4 = rbinom( 1, size=2, prob=p.mut )        # [!CAUTION!] adjust p.mut for other settings
    tau_S4 = sort( runif( m_S4, min=tau.min, max=tau.max ) )
  } else {
    # Step 2: Thinning
    ran.val_S2 = runif( m_S1, min=0, max=1 )
    tau_S2 = tau_S1[ ran.val_S2 <= 0.5 ]
    m_S2 = length( tau_S2 )

    # Step 3: Shifting
    ran.val_S3 = rnorm( m_S2, mean=0, sd=t_inc )  # [!CAUTION!] adjust sd=t_inc for other settings
    tau_S3.tmp = sort( unique( tau_S2 + ran.val_S3 ) )
    tau_S3 = tau_S3.tmp[ tau.min < tau_S3.tmp & tau_S3.tmp < tau.max ] # changepoints must occur in (tau.min, tau.max)
    m_S3 = length( tau_S3 )

    # Step 4: Mutation
    m_mut = rbinom( 1, size=2, prob=p.mut )       # [!CAUTION!] adjust p.mut for other settings
    tau_S4.mut = sort( runif( m_mut, min=tau.min, max=tau.max ) )
    tau_S4 = sort( unique( c( tau_S3, tau_S4.mut ) ) )
    m_S4 = length( tau_S4 )
  }

  ch_new = c( m_S4, tau_S4 )                      # new changepoint configuration (m; tau_1,...,tau_m)
  
  return( ch_new )
}

# --------
# Function to implement GA for trend changepopint detection at continuous time locations
# --------
ga.cpt_ts <- function( y, z=NULL, trd.type=c("mean","linear","quadratic","cubic"), ic=c("BIC","MDL"), 
                       gen.size, max.itr, p.mut, seed, is.graphic=FALSE, is.print=FALSE ) {
                                                  # y       : a binary series
                                                  # z       : covariates
                                                  # trd.type: fitted model order
                                                  # ic      : penalty type for penalized log-likelihood 
                                                  # gen.size: number of combined chromosomes in a generation
                                                  # max.itr : number of iterations (generations)
                                                  # p.mut   : probability of a mutation
                                                  # seed    : a seed number to use for random number generations
  # Changepoint configuration
  n   = length( y )                               # n    : series length, including missing values
  t.1 = time( y )[1]                              # t.1  : first time point of series
  t.n = time( y )[n]                              # t.n  : last time point of series

  d.min = 3                                       # d.min: min distance between 2 consecutive changepoint times, 3 days
  tau.min = t.1 + d.min                           # tau.min: min time location of changepoints
  tau.max = t.n - d.min                           # tau.max: max time location of changepoints
  m.max = min( 8, round(n/14) )                   # m.max: max number of possible changepoints in initial generation

  Confg.cur = list()                              # changepoint configurations in current generation
  Confg.sol = list()                              # best changepoint configuration for each generation
  Confg.all = list()                              # changepoint configurations of all generations

  Pnlik.sol = numeric( length=max.itr )                # optimized penalized likelihood value for each generation
  Pnlik.all = matrix( 0, nrow=max.itr, ncol=gen.size ) # penalized likelihood values for all changepoint configurations

  if ( is.graphic ) {
    dev.new( width=10, height=4 )                 # [!CAUTION!] adjust for other settings
    par( mfrow=c(1,1), mar=c(4.5,4.5,3,1), mex=0.8 )
  }

  # Initial generation
  set.seed( seed )
  for ( g in 1:1 ) {
    if ( is.print ) print( paste("#----------[  Generation =",g,"has begun at",Sys.time()," ]----------#"), quote=FALSE )

    # Include a chromosome of no changepoints
    chrom_zro = 0
    Confg.cur[[1]] = chrom_zro

    # Generate other chromosomes
    j = 2                                         # index for a new unique chromosome
    while ( j <= gen.size ) {                     # index for all repetitions until a set number of unique configurations
      # Generate a random chromosome
      m = rbinom( 1, size=m.max, prob=0.4 )       # [!CAUTION!] adjust prob=0.4 for other settings
      tau = sort( runif( m, min=tau.min, max=tau.max ) )
      chrom_new = c( m, tau )                     # changepoint chromosome (m; tau_1,...,tau_m)

      Confg.cur[[j]] = chrom_new

      # Check uniqueness and minimum distance conditions
      is.unique = ( length( unique( Confg.cur[1:j] ) ) == j )       # uniqueness check: both type (integer or real) and value
      is.distant = check_distance( cp=chrom_new, n=n, d.min=d.min ) # min distance check: 2 consecutive times >= d.min

      # Increase generation size only if (1) a new unique child chromosome is born and (2) the min distance condition is met
      if ( is.unique & is.distant ) {
        j = j+1
      }
    }                                             # ending while loop in j

    # Compute penalized log-likelihood in current generation
    for ( j in 1:gen.size ) {
      chrom = Confg.cur[[j]]

      if ( trd.type == "mean"   )    fm = fit.PMSar( y, cp=chrom, z=z )
      if ( trd.type == "linear" )    fm = fit.PLTar( y, cp=chrom, z=z )
      if ( trd.type == "quadratic" ) fm = fit.PQSar( y, cp=chrom, z=z )
      if ( trd.type == "cubic"  )    fm = fit.PCSar( y, cp=chrom, z=z )

      Pnlik.all[g,j] = ifelse( ic == "BIC",
                         -2*fm$loglik + penalty_BIC( y, model=fm, cp=chrom ),
                         -1*fm$loglik + penalty_MDL( y, cp=chrom ) )

      if ( is.graphic ) {
        plot.ts( y, xlab="Number of days from 2023-08-10 (Day 1)", ylab="Generalized logit of beta_t", col="gray",
                 main=paste("Generation", g, "& Child", j, "(", ic, "=", format(Pnlik.all[g,j],nsmall=4), ")") )
        abline( v=chrom[-1], col="blue", lty=2 )
      }
    }                                             # ending for loop in j

    # Determine the best chromosome in current generation
    loc.sol = which( Pnlik.all[g,] == min(Pnlik.all[g,]) )
    if ( length( loc.sol ) > 1 ) {
      loc.sol = sample( loc.sol, size=1 )         # randomly choose a best chromosome if there are >1 best chromosomes
    }
    chrom.sol = Confg.cur[[loc.sol]]              # best combined changepoint configuration of current generation
    Confg.sol[[g]] = chrom.sol
    Confg.all[[g]] = Confg.cur                    # current generation
    Pnlik.sol[g]   = Pnlik.all[g,loc.sol]         # optimized penalized log-likelihood

    if ( is.print ) {
      print( chrom.sol )
      print( paste( ic, "=", format( Pnlik.sol[g], nsmall=4 ) ), quote=FALSE )
    }
  }                                               # ending for loop in g

  # Next generations from 2 to gen.size
  for ( g in 2:max.itr ) {
    if ( is.print ) print( paste("#----------[  Generation =",g,"has begun at",Sys.time()," ]----------#"), quote=FALSE )

    # Rank chromosomes in previous (g-1)th generation
    gen.rank = rank( -Pnlik.all[g-1,] )           # higher rank is more likely to be selected
    gen.rank_sum = sum( gen.rank )

    # Carry over the fittest chromosome in previous generation to current generation
    Confg.pre      = Confg.all[[g-1]]
    Confg.cur[[1]] = Confg.sol[[g-1]]
    Pnlik.all[g,1] = Pnlik.sol[g-1]

    # Generate other combined chromosomes
    j = 2                                         # index for a new unique chromosome
    while ( j <= gen.size ) {                     # index for all repetitions until a set number of unique configurations
      # Select father and mother chromosomes
      loc.prt = sample( 1:gen.size, size=2, replace=FALSE, prob=gen.rank/gen.rank_sum )
      loc.dad = loc.prt[1]
      loc.mom = loc.prt[2]
      chrom_dad = Confg.pre[[loc.dad]]
      chrom_mom = Confg.pre[[loc.mom]]

      # Produce a child chromosome
      chrom_new = produce_child.cont( ch_dad=chrom_dad, ch_mom=chrom_mom, tau.min=tau.min, tau.max=tau.max, p.mut=p.mut )
      Confg.cur[[j]] = chrom_new

      # Check uniqueness and minimum distance conditions
      is.unique = ( length( unique( Confg.cur[1:j] ) ) == j )       # uniqueness check: both type (integer or real) and value
      is.distant = check_distance( cp=chrom_new, n=n, d.min=d.min ) # min distance check: 2 consecutive times >= d.min

      # Increase generation size only if (1) a new unique child chromosome is born and (2) the min distance condition is met
      if ( is.unique & is.distant ) {
        j = j+1
      }
    }                                             # ending while loop in j

    # Compute penalized log-likelihood in current generation
    for ( j in 1:gen.size ) {
      chrom = Confg.cur[[j]]

      if ( trd.type == "mean"   )    fm = fit.PMSar( y, cp=chrom, z=z )
      if ( trd.type == "linear" )    fm = fit.PLTar( y, cp=chrom, z=z )
      if ( trd.type == "quadratic" ) fm = fit.PQSar( y, cp=chrom, z=z )
      if ( trd.type == "cubic"  )    fm = fit.PCSar( y, cp=chrom, z=z )

      Pnlik.all[g,j] = ifelse( ic == "BIC",
                         -2*fm$loglik + penalty_BIC( y, model=fm, cp=chrom ),
                         -1*fm$loglik + penalty_MDL( y, cp=chrom ) )

      if ( is.graphic ) {
        plot.ts( y, xlab="Number of days from 2023-08-10 (Day 1)", ylab="Generalized logit of beta_t", col="gray",
                 main=paste("Solution in Generation", g-1, "(", ic, "=", format(Pnlik.sol[g-1],nsmall=4), ") vs",
                            "Generation", g, "& Child", j, "(", ic, "=", format(Pnlik.all[g,j],nsmall=4), ")") )
        abline( v=chrom.sol[-1], col="red", lty=1 )
        abline( v=chrom[-1], col="blue", lty=2 )
      }
    }                                             # ending for loop in j

    # Determine the best chromosome in current generation
    loc.sol = which( Pnlik.all[g,] == min(Pnlik.all[g,]) )
    if ( length( loc.sol ) > 1 ) {
      loc.sol = sample( loc.sol, size=1 )         # randomly choose a best chromosome if there are >1 best chromosomes
    }
    chrom.sol = Confg.cur[[loc.sol]]              # best combined changepoint configuration of current generation
    Confg.sol[[g]] = chrom.sol
    Confg.all[[g]] = Confg.cur                    # current generation
    Pnlik.sol[g]   = Pnlik.all[g,loc.sol]         # optimized penalized log-likelihood

    if ( is.print ) {
      print( chrom.sol )
      print( paste( ic, "=", format( Pnlik.sol[g], nsmall=4 ) ), quote=FALSE )
    }
  }                                               # ending for loop in g

  list( gen.all=Confg.all, gen.sol=Confg.sol, val.all=Pnlik.all, val.sol=Pnlik.sol, solution=chrom.sol )
}
