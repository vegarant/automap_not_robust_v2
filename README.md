# On non-robustness, hallucinations and unpredictability in AI for imaging

Code related to the paper *"On non-robustness, hallucinations and unpredictability in AI for imaging"*.

## Setup
The data used in the paper can be downloaded from [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/storage_automap_not_robust.zip), and the AUTOMAP network weights can be downloaded [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/cs_poisson_for_vegard.h5) (3.4 GB). After downloading the data, modify the paths in the file `adv_tools_PNAS/automap_config.py` to link all relevant paths to the data. To run the stability test for the LASSO experiment, add the [UiO-CS/optimization](https://github.com/UiO-CS/optimization) and [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) packages to your Python path. 

## Overview of the different files
----------------------------

* Figure 1: Demo_test_automap_stability.py and Demo_test_automap_non_zero_mean_noise.py.
* Figure 2: Demo_test_automap_error_map.py, Demo_test_lasso_error_map.py and Demo_test_read_and_plot_error_maps.py
* Extended Data Figure 1: Demo_test_automap_stability.py and Demo_test_lasso_stability.py
* Extended Data Table 1: Demo_test_automap_compute_norms.py
* SI Figure 3: See https://github.com/vegarant/new_experiment_zero_mean
* SI Figure 4: Demo_test_automap_stability_knee.py, Demo_test_automap_non_zero_mean_noise.py and Demo_test_lasso_non_zero_mean_noise.py
* SI Table 1: Use scripts in matlab folder.

---------------------------

All scripts have been executed with Tensorflow version 1.14.0.



