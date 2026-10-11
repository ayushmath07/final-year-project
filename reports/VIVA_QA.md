# Viva questions and answers

## Why is ground truth synthetic?

Real company connections are incomplete and definitions vary. Synthetic graphs make edge recovery measurable while keeping the real-data demonstration exploratory.

## What makes SRCS different from correlation?

Correlation uses all observations. SRCS conditions on source-company shock events, searches only the following lag window and rewards response consistency across repeated events.

## Why multiply average response by consistency?

A single large response can inflate a mean. The consistency factor reduces scores when only a small fraction of source events produce a meaningful response.

## Why use separate calibration and test seeds?

Choosing a threshold on the same graphs used for final metrics would overstate performance. Separate seeds make the reported test metrics out of sample with respect to threshold selection.

## Why count only the upper triangle?

The graph is undirected, so entries `(i,j)` and `(j,i)` represent one edge. Counting both doubles every confusion count.

## Does Granger-style prediction prove causality?

No. It measures whether source lags improve target prediction under a specified linear model. Omitted common drivers and model misspecification can create predictive associations.

## Why remove the market component?

A broad market move can make unrelated companies move together. Regressing company returns on a market index reduces, but does not eliminate, that confounding.

## What happens to constant series?

Their standard deviation is zero, so the implementation maps their standardized values to zero and avoids division by zero.

## How are incomplete windows handled?

Events too close to the end of the sample are excluded from scoring and detection so every retained event has the configured number of future observations.

## What is the main limitation?

The strongest limitation is the gap between a controlled static linear simulator and real financial markets with shared news, changing volatility and time-varying relationships.
