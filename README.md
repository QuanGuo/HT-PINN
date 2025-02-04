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
   
 
## Model Training (model_coeff)
   
recommended hyper-parameters are saved in **hyper_parameters.txt**
   
coefficients (weights and biases) of example trained forward model are save in **model_u_12.txt**
   
   
## How to use

1) Clone.

2) For forward problem: **HT_PINN_forward.ipynb**; 
   For inverse problem: **HT_PINN_inverse.ipynb**
  
3) Tune hyper-parameters

4) Train and save results

## Citations
```
@article{GUO2023128828,
      title = {High-dimensional inverse modeling of hydraulic tomography by physics informed neural network (HT-PINN)},
      journal = {Journal of Hydrology},
      volume = {616},
      pages = {128828},
      year = {2023},
      issn = {0022-1694},
      doi = {https://doi.org/10.1016/j.jhydrol.2022.128828},
      url = {https://www.sciencedirect.com/science/article/pii/S0022169422013981},
      author = {Quan Guo and Yue Zhao and Chunhui Lu and Jian Luo}
}
```
