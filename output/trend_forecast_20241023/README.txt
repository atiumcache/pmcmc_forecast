This test was run on 2024.10.23.

We just fixed an error in the PMCMC transition function:
the new hospitalizations were not being calculated quite right.
The result was that stochastic effect was growing over time,
even as hospitalizations were going down.

So, we fixed that issue and reran the PMCMC.

And then fed those new results into this Trend Forecast test.
