"""
This script reads the worst-case perturbations produced by the script
'Demo_test_automap_stability.py', sample these perturbations and use the
sampled perturbations as the mean for randomly drawn gaussian noise vectors.
These noise vectors are then added to the measurements of other images than the
worst-case perturbations where computed with respect to.
"""


import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path
from PIL import Image

from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, scale_to_01
from adv_tools_PNAS.Runner import Runner;
from adv_tools_PNAS.Automap_Runner import Automap_Runner;
from adv_tools_PNAS.automap_tools import load_runner, read_automap_k_space_mask, compile_network, hand_f, sample_image;
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

runner_id_automap = 5;

dest_plots = 'plots_non_zero_mean';
dest_data = 'data_non_zero_mean';

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);
if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);

N = 128

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

runner = load_runner(runner_id_automap);

pert_nbr = 2
HCP_nbr = 1002
data = loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));
mri_data = data['im'];
im_nbrs = [37, 50, 76];

batch_size = 1;
mri_data.shape[0];

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

sample = lambda im: sample_image(im, k_mask_idx1, k_mask_idx2)


fID = open(join(dest_data, 'magnitude_e.txt'), 'w')
data_dict = {}
for r_value in range(0,5):

    rr = runner.r[r_value];
    rr = rr[pert_nbr, :, :];
    rr = np.expand_dims(rr, 0)
    e = sample(rr);
    if r_value == 0:
        e_random = np.random.normal(loc=0, scale=0.01, size=e.shape)
    else:
        e_random = np.random.normal(loc=e, scale=0.01, size=e.shape)
    im_nbr= im_nbrs[0];
    image = mri_data[im_nbr];
    image = np.expand_dims(image, 0);
    Ax = sample(image)
    n_diff_e = l2_norm_of_tensor(e - e_random)
    n_e = l2_norm_of_tensor(e);
    n_y = l2_norm_of_tensor(Ax)
    str1 = f"r: {r_value}, |e-e_rand|/|e|: {n_diff_e/n_e}";

    data_dict[f"e{r_value}"] = e_random

    print(str1)
    fID.write(str1 + "\n")

    for i in range(len(im_nbrs)):
        im_nbr= im_nbrs[i];
        image = mri_data[im_nbr];
        image = np.expand_dims(image, 0);

        Ax = sample(image)

        im_rec = np.uint8(np.squeeze(255*scale_to_01(raw_f(Ax+e_random))));

        pil_im_rec = Image.fromarray(im_rec);
        pil_im_rec.save(join(dest_plots, f'im_rec_automap_HCP_{HCP_nbr}_im_nbr_{im_nbr}_random_non_zero_mean_r_idx_{r_value}.png'))

savemat(join(dest_data, f"automap_random_pert.mat"), data_dict);
fID.close()




