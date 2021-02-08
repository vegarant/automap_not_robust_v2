"""
This script add random noise in two images, sample them and compute the AUTOMAP
reconstruction. The resulting images are stored as png images.
"""

import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg;
import numpy as np;
from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.automap_tools import read_automap_k_space_mask, compile_network, hand_f, hand_dQ;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, scale_to_01
from PIL import Image
from scipy.io import loadmat, savemat

use_gpu = False
compute_node = 3
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    print('Compute node: {}'.format(compute_node))
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

# Turn on soft memory allocation
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
sess = tf.compat.v1.Session(config=tf_config)


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

N = 128
size_zoom = 80

HCP_nbr = 1004 # Use HCP_nbr 1002 to genereate the other dataset
data = scipy.io.loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));

mri_data = data['im'];
#new_im1 = mpimg.imread(join(src_data, 'brain1_128_anonymous.png'))
#new_im2 = mpimg.imread(join(src_data, 'brain2_128_anonymous.png'))

batch_size = mri_data.shape[0]

for i in range(batch_size):
    mri_data[i, :,:] = scale_to_01(mri_data[i,:,:]);

plot_dest = './plots_error_map'
data_dest = './data_error_map'

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest)
if not (os.path.isdir(data_dest)):
    os.mkdir(data_dest)

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)

print('mri_data.shape: ', mri_data.shape);


fx_no_noise = f(mri_data)

im_rec = np.zeros(mri_data.shape, mri_data.dtype);


for i in range(batch_size):
    im_rec[i,:,:]  = scale_to_01(fx_no_noise[i]);
    
savemat(join(data_dest, f'im_rec_automap_HCP_{HCP_nbr}.mat'), {'mri_data': mri_data, 'im_rec': im_rec});

sess.close();



