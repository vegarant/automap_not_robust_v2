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
from scipy.io import loadmat

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

im_nbr_gauss1 = 0;
im_nbr_gauss2 = 1;

N = 128
size_zoom = 80

HCP_nbr = 1002
data = scipy.io.loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));

mri_data = data['im'];
new_im1 = mpimg.imread(join(src_data, 'brain1_128_anonymous.png'))
#new_im2 = mpimg.imread(join(src_data, 'brain2_128_anonymous.png'))

mri_data = np.zeros([2, N,N], dtype='float32')
mri_data[0, :, :] = new_im1
mri_data[1, :, :] = data['im'][36]

batch_size = mri_data.shape[0]

plot_dest = './plots_random'
data_dest = './data_random'

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest)
if not (os.path.isdir(data_dest)):
    os.mkdir(data_dest)

sess = tf.compat.v1.Session()

raw_f, _ = compile_network(sess, batch_size)

f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)

# Create noise, mri_data.shape = [2,N,N]
noise_gauss1 = np.float32(np.random.normal(loc=0, scale=1, size=mri_data.shape))
noise_gauss2 = np.float32(np.random.normal(loc=0, scale=1, size=mri_data.shape))

# Scale the noise
norm_mri_data_gauss1 = l2_norm_of_tensor(mri_data[im_nbr_gauss1])
norm_mri_data_gauss2 = l2_norm_of_tensor(mri_data[im_nbr_gauss2])
norm_noise_gauss1 = l2_norm_of_tensor(noise_gauss1[im_nbr_gauss1])
norm_noise_gauss2 = l2_norm_of_tensor(noise_gauss2[im_nbr_gauss2])

print('mri_data.shape: ', mri_data.shape);

p = 0.04;
noise_gauss1 *= (p*norm_mri_data_gauss1/norm_noise_gauss1);
noise_gauss2 *= (p*norm_mri_data_gauss2/norm_noise_gauss2);

# Save noise
fname_data = 'noise_%d_automap.mat' % (round(1000*p));

scipy.io.savemat(join(data_dest, fname_data), {'noise_gauss1': noise_gauss1, 'noise_gauss2': noise_gauss2});

image_no_noise = mri_data
image_noisy_gauss1 = mri_data + noise_gauss1
image_noisy_gauss2 = mri_data + noise_gauss2

fx_no_noise = f(mri_data)
fx_noise_gauss1 = f(image_noisy_gauss1)
fx_noise_gauss2 = f(image_noisy_gauss2)

for i in range(batch_size):
    # Save reconstruction with noise
    image_data_no_noise = np.uint8(255*scale_to_01(fx_no_noise[i]));
    image_data_gauss1 = np.uint8(255*scale_to_01(fx_noise_gauss1[i]));
    image_data_gauss2 = np.uint8(255*scale_to_01(fx_noise_gauss2[i]));

    image_rec_no_noise = Image.fromarray(image_data_no_noise);
    image_rec_gauss1 = Image.fromarray(image_data_gauss1);
    image_rec_gauss2 = Image.fromarray(image_data_gauss2);

    image_rec_no_noise.save(join(plot_dest, 'im_no_noise_rec_nbr_%d.png' % (i)));
    image_rec_gauss1.save(join(plot_dest, 'im_gauss1_rec_p_%d_nbr_%d.png' % (round(p*1000), i)));
    image_rec_gauss2.save(join(plot_dest, 'im_gauss2_rec_p_%d_nbr_%d.png' % (round(p*1000), i)));

    # Save original image with noise
    image_orig_no_noise = Image.fromarray(np.uint8(255*(scale_to_01(image_no_noise[i]))));
    image_orig_gauss1 = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_gauss1[i]))));
    image_orig_gauss2 = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_gauss2[i]))));

    image_orig_no_noise.save(join(plot_dest, 'im_no_noise_nbr_%d.png' % (i)));
    image_orig_gauss1.save(join(plot_dest, 'im_gauss1_noise_p_%d_nbr_%d.png' % (round(p*1000), i)));
    image_orig_gauss2.save(join(plot_dest, 'im_gauss2_noise_p_%d_nbr_%d.png' % (round(p*1000), i)));

    # Create zoomed crops, reconstructions 
    image_rec_no_noise_zoom1 = Image.fromarray(image_data_no_noise[:size_zoom, -size_zoom:]);
    image_rec_no_noise_zoom2 = Image.fromarray(image_data_no_noise[-size_zoom:, -size_zoom:]);
    image_rec_gauss1_zoom = Image.fromarray(image_data_gauss1[:size_zoom, -size_zoom:]);
    image_rec_gauss2_zoom = Image.fromarray(image_data_gauss2[-size_zoom:, -size_zoom:]);

    image_rec_no_noise_zoom1.save(join(plot_dest, 'im_no_noise_rec_nbr_%d_zoom1.png' % (i)));
    image_rec_no_noise_zoom2.save(join(plot_dest, 'im_no_noise_rec_nbr_%d_zoom2.png' % (i)));
    image_rec_gauss1_zoom.save(join(plot_dest, 'im_gauss1_rec_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));
    image_rec_gauss2_zoom.save(join(plot_dest, 'im_gauss2_rec_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));

    # Create zoomed crops, images with noise 
    image_orig_no_noise_zoom1 = Image.fromarray(np.uint8(255*(scale_to_01(image_no_noise[i, :size_zoom, -size_zoom:]))));
    image_orig_no_noise_zoom2 = Image.fromarray(np.uint8(255*(scale_to_01(image_no_noise[i, -size_zoom:, -size_zoom:]))));
    image_orig_gauss1_zoom = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_gauss1[i, :size_zoom, -size_zoom:]))));
    image_orig_gauss2_zoom = Image.fromarray(np.uint8(255*(scale_to_01(image_noisy_gauss2[i, -size_zoom:, -size_zoom:]))));

    image_orig_no_noise_zoom1.save(join(plot_dest, 'im_no_noise_nbr_%d_zoom1.png' % (i)));
    image_orig_no_noise_zoom2.save(join(plot_dest, 'im_no_noise_nbr_%d_zoom2.png' % (i)));
    image_orig_gauss1_zoom.save(join(plot_dest, 'im_gauss1_noise_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));
    image_orig_gauss2_zoom.save(join(plot_dest, 'im_gauss2_noise_p_%d_nbr_%d_zoom.png' % (round(p*1000), i)));

sess.close();

