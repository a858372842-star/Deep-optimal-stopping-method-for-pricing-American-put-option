# Deep Learning Methods for American Put Option Pricing
This repository contains the Python source code used for the numerical experiments in my MSc dissertation.

## Contents
- Section 4.1 One-dimensional American put pricing
- Section 4.2 Sensitivity analysis for the number of exercise intervals and training batch size
- Section 4.3 Learned value-function and exercise-boundary analysis
- Section 4.4 High-dimensional American basket put pricing
- Section 4.5 Stress testing under high volatility and high interest rates

## Requirements
- Python
- TensorFlow
- NumPy
- pandas
- SciPy
- Matplotlib

## Numerical method
The implementation applies the deep optimal stopping method of Becker, Cheridito and Jentzen (2019). Continuous-time prices and exercise boundaries for the one-dimensional cases are calculated using a free-boundary integral method.

## Author
Jiale Du