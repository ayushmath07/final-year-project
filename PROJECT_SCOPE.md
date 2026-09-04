# Step 1 scope freeze

## Title

**Inferring Hidden Economic Connections from Financial Shock Propagation: A
Synthetic and Real-Data Prototype**

## Research question

Can repeated lagged responses after company-specific financial shocks provide
useful evidence for reconstructing hidden economic connections between
companies?

## Objectives

1. Generate a synthetic company network with known connections.
2. Simulate company-specific shocks that travel through that network.
3. Later, score shock responses and reconstruct the hidden network.
4. Compare the proposed method with simple baselines.
5. Demonstrate the same pipeline on a small cached real-data example.

## Step 1 deliverables

- A small, reproducible synthetic environment.
- A true-network figure, a shock-propagation figure, and an architecture figure.
- Saved synthetic observations and a structured shock-event list.
- Tests that prove basic simulator correctness.
- Initial black-book prose and chapter skeleton (kept out of GitHub).

## Assumptions for the prototype

- The hidden network is static, sparse, binary, and undirected.
- A shock affects a company directly, then can affect connected companies later.
- The simulation uses Gaussian noise and conservative coefficients to stay stable.
- The true adjacency matrix is retained for evaluation only; later inference code
  must receive observations and events, not the adjacency matrix.

## Explicit exclusions until later milestones

- SRCS scoring and inferred-network thresholding
- correlation, lagged-correlation, and Granger-style baselines
- robustness experiments, real-data analysis, and Streamlit deployment
- claims of causal direction or confirmed supplier/customer relationships

## Architecture

`hidden graph -> normalized adjacency -> shocks + noise -> company states -> saved observations -> later inference`

## Why correlation alone is insufficient

Two companies can move together because the overall market or a shared news
event affects both.  This project instead looks for repeated delayed responses
after company-specific events, while treating any real-data result as ranked
evidence rather than causal proof.

