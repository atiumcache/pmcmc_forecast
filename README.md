# PMCMC --- Forecasting Flu Hospitalizations
The goal of this project is to predict new flu cases using PMCMC (Particle Markov Chain Monte Carlo) and trend forecasting. We model the state of our system using an SIRH 
model, and we infer the transmission rate using PMCMC.

This repository implements an automated pipeline to:
- Collect new hospitalization data.
- Run the hospitalization data through a PMCMC algorithm to estimate the transmission rate.
- Forecast future transmission rates. 
- Use the forecasted transmission rates to predict future hospitalizations. 

We utilize bash scripts to automate and parallelize 
most of this process on an HPC cluster. 

## Implementation Details
See `/notebooks/usage_example_20241116.ipynb` for a detailed look at various components of the forecast pipeline. 

## Determine Accuracy
Use Weighted Interval Scores (WIS) to determine the accuracy of our forecasts. This is performed in the [Flu Forecast Accuracy repository](https://github.com/atiumcache/flu-forecast-accuracy). We also compare this method with MCMC forecasting.

More information on WIS can be found here:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7880475/

## Particle Filter Credits
Particle filter code derived from:   

C. Gentner, S. Zhang, and T. Jost, “Log-PF: particle filtering in logarithm domain,” Journal of Electrical and Computer Engineering, vol. 2018, Article ID 5763461, 11 pages, 2018.

D. Calvetti, A. Hoover, J. Rose, and E. Somersalo. Bayesian particle filter algorithm for learning epidemic dynamics. Inverse Problems, 2021



