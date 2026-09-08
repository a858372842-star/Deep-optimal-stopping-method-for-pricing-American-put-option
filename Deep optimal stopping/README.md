# Deep Learning Methods for American Put Option Pricing
This repository contains the Python source code used for the numerical experiments in my MSc dissertation.

## Contents
- section_4_1_one_dimensional_pricing.py is the code used in Section 4.1 
- run_4_1_five_seed_sensitivity.py is the code used in Section 4.2 and 4.3 Main results
- postprocess_7case_value_learning.py is the code used in Section 4.2 and 4.3 to calculate the Learning curves and Value functions
- local_boundary_learning_postprocess.py is the code used in Section 4.2 and 4.3 to generate the learned exercise boundary figures and stopping region
- analyse_boundary_blocks.py is the code used in Section 4.2 and 4.3 to calculate the MAE and RMSE
- becker_basket_engine.py is the code used in Section 4.4 acting as the High dimensional valuation engine
- run_high_dim_basket.py is the code used in Section 4.4 acting as the Experiment driver
- Section 4.5.py is the code used in Section 4.5 to complete Stress testing under high volatility and high interest rates

## Requirements
- Python
- TensorFlow
- NumPy
- pandas
- SciPy
- Matplotlib

## Numerical method
The implementation applies the deep optimal stopping method of Becker, Cheridito and Jentzen (2019). Continuous-time prices and exercise boundaries for the one-dimensional cases are calculated using a free-boundary integral method.

## Availability note

The high-dimensional experiment driver imports an auxiliary script, `summarise_results.py`, which was used to aggregate the case-level results after the numerical calculations had been completed. This file was stored on a temporary AutoDL instance that has expired and is no longer available. It did not perform neural-network training or calculate the lower bound, upper bound, point estimate, or duality gap. These calculations are implemented in `becker_basket_engine.py` and saved separately for each experimental case. Therefore, the original Python files are retained here without modification, but the automatic summary step cannot be reproduced from this repository alone.

## Author
Jiale Du