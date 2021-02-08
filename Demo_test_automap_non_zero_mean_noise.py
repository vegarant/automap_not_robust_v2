import time

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
from scipy.io import loadmat
import matplotlib.pyplot as plt

runner_id_automap = 5;

dest_plots = 'plots_automap_non_zero_mean';

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);

N = 128

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

runner = load_runner(runner_id_automap);

im_nbr = 2;
mri_data = runner.x0[0] 
print('mri_data.shape: ', mri_data.shape )
#print('samp.shape: ', samp.shape )

batch_size = 1;
mri_data.shape[0];
start = time.time()

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

f = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)
g = lambda x: scale_to_01(f(x))

sample = lambda im: sample_image(im, k_mask_idx1, k_mask_idx2)

im_nbr = 2
image = mri_data[im_nbr];
image = np.expand_dims(image, 0);

Ax = sample(image)

for r_value in range(0,5):
    rr = runner.r[r_value];
    rr = rr[im_nbr, :, :];
    rr = np.expand_dims(rr, 0)
    e = sample(rr);
    if r_value == 0:
        e_random = np.random.normal(loc=0, scale=0.01, size=e.shape)
    else:
        e_random = np.random.normal(loc=e, scale=0.01, size=e.shape)

    im_rec = np.uint8(np.squeeze(255*scale_to_01(raw_f(Ax+e_random))));
    
    pil_im_rec = Image.fromarray(im_rec);
    pil_im_rec.save(join(dest_plots, f'im_rec_automap_random_non_zero_mean_r_idx_{r_value}.png'))
    #compute_norms_v2_lp(f, image, image+rr, r_value, p=2);

