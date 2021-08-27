# HT-PINN: Hydraulic Tomography Physics-Informed Neural Network 
Author: Quan G

## About this notebook

This notebook applies PINN to hydraulic tomography inverse modeling to estimate spatially distributed hydraulic conductivity (inverse problem) as well as approximating relative hydraulic heads in pumping tests (forward problem).

Please note this work:
* Assumes the reader is comfortable with Python, especially, python notebook and pytorch.
* Google Cloud is recommended as the computing platform.

## Data

**heads/heads_pump<id>**: Hydraulic heads under each pumping test (solved with FEM)
**logK_field**: natural log hydraulic conductivity field (lnK)
**alpha_vector**: hidden random variables used to generated logK field with PCA realization generation method
**K_measure_id_61**: idx of direct measurement on hydraulic conductivity on domain mesh
**K_measure_id_25**: idx of pumping wells on domain mesh
   
## How to use

1) Clone.

2) For forward problem: forward_example.ipynb
   For inverse problem: inverse_exampel.ipynb
  
3) Tune hyper-parameter

4) train and save results

## Citation Format
