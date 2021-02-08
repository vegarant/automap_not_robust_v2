"""
This script reads the random noise generated in the script
'Demo_test_automap_random_noise.py'. It samples the noisy images, and recover
an approximation to these using the LASSO method.
"""

import time
import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path

from optimization.gpu.operators import MRIOperator
from optimization.gpu.proximal import WeightedL1Prox, SQLassoProx2
from optimization.gpu.algorithms import SquareRootLASSO
from optimization.utils import estimate_sparsity, generate_weight_matrix
from tfwavelets.dwtcoeffs import get_wavelet
from tfwavelets.nodes import idwt2d
from PIL import Image
import matplotlib.image as mpimg;

from adv_tools_PNAS.automap_config import src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor
from adv_tools_PNAS.Runner import Runner;

src_noise = 'data_random';

N = 128
wavname = 'db2'
levels = 3
use_gpu = True
compute_node = 1
dtype = tf.float64;
sdtype = 'float64';
scdtype = 'complex128';
cdtype = tf.complex128
wav = get_wavelet(wavname, dtype=dtype);
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


dest_data = 'data_random';
dest_plots = 'plots_random';

if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);

# Parameters for the CS-algorithm
n_iter = 1000
tau = 0.6
sigma = 0.6
lam = 0.001

# Parameters for CS algorithm
pl_sigma = tf.compat.v1.placeholder(dtype, shape=(), name='sigma')
pl_tau   = tf.compat.v1.placeholder(dtype, shape=(), name='tau')
pl_lam   = tf.compat.v1.placeholder(dtype, shape=(), name='lambda')

# Build Primal-dual graph
tf_im = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='image')
tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

# For the weighted l^1-norm
pl_weights = tf.compat.v1.placeholder(dtype, shape=[N,N,1], name='weights')

tf_input = tf_im

op = MRIOperator(tf_samp_patt, wav, levels, dtype=dtype)
measurements = op.sample(tf_input)

tf_adjoint_coeffs = op(measurements, adjoint=True)
adj_real_idwt = idwt2d(tf.math.real(tf_adjoint_coeffs), wav, levels)
adj_imag_idwt = idwt2d(tf.math.imag(tf_adjoint_coeffs), wav, levels)
tf_adjoint = tf.complex(adj_real_idwt, adj_imag_idwt)

prox1 = WeightedL1Prox(pl_weights, pl_lam*pl_tau, dtype=dtype)
prox2 = SQLassoProx2(dtype=dtype)

alg = SquareRootLASSO(op, prox1, prox2, measurements, sigma=pl_sigma, tau=pl_tau, lam=pl_lam, dtype=dtype)

initial_x = op(measurements, adjoint=True)

result_coeffs = alg.run(initial_x)

real_idwt = idwt2d(tf.math.real(result_coeffs), wav, levels)
imag_idwt = idwt2d(tf.math.imag(result_coeffs), wav, levels)
tf_recovery = tf.complex(real_idwt, imag_idwt)

samp = np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool))
samp = np.expand_dims(samp, -1)

# Read the data.
# Images:
im1 = np.asarray(Image.open(join(src_data, 'brain1_128_anonymous.png')), dtype=sdtype)/255;
im2 = np.asarray(Image.open(join(src_data, 'brain2_128_anonymous.png')), dtype=sdtype)/255;
print('np.amax(im1): ', np.amax(im1));
print('np.amin(im1): ', np.amin(im1));
print('im1.dtype: ', im1.dtype);


HCP_nbr = 1002
data = scipy.io.loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));

mri_data = data['im'];
new_im1 = mpimg.imread(join(src_data, 'brain1_128_anonymous.png'))
#new_im2 = mpimg.imread(join(src_data, 'brain2_128_anonymous.png'))

mri_data = np.zeros([6, N,N], dtype=sdtype)
mri_data[0, :, :] = new_im1
mri_data[1, :, :] = data['im'][36]
mri_data[2, :, :] = new_im1
mri_data[3, :, :] = data['im'][36]
mri_data[4, :, :] = new_im1
mri_data[5, :, :] = data['im'][36]

p = 0.04

noise_dict    = scipy.io.loadmat(join(src_noise, f'noise_{round(1000*p)}_automap.mat')) 
noise_gauss1   = noise_dict['noise_gauss1'];
noise_gauss2 = noise_dict['noise_gauss2'];

mri_data[0:2, :, :] = mri_data[0:2, :, :] 
mri_data[2:4, :, :] = mri_data[2:4, :, :] + noise_gauss1
mri_data[4:6, :, :] = mri_data[4:6, :, :] + noise_gauss2

batch_size = mri_data.shape[0];
zoom_size = 80;
with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
    weights = np.ones([128,128,1], dtype=sdtype);

    np_im_rec = np.zeros([batch_size, N, N], dtype=scdtype);

    for i in range(batch_size):

        _image = mri_data[i,:,:];
        _image = np.expand_dims(_image, -1)
        _rec = sess.run(tf_recovery, feed_dict={ 'tau:0': tau,
                                                 'lambda:0': lam,
                                                 'sigma:0': sigma,
                                                 'weights:0': weights,
                                                 'n_iter:0': n_iter,
                                                 'image:0': _image,
                                                 'sampling_pattern:0': samp})
        np_im_rec[i,:,:] = _rec[:,:,0];


    np_im_rec = np.abs(np_im_rec);
    np_im_rec[np_im_rec > 1] = 1;

    im1_no_noise = np_im_rec[0,:,:];
    im2_no_noise = np_im_rec[1,:,:];
    im1_gauss1   = np_im_rec[2,:,:];
    im2_gauss1   = np_im_rec[3,:,:];
    im1_gauss2   = np_im_rec[4,:,:];
    im2_gauss2   = np_im_rec[5,:,:];

    fname1_no_noise = f'im_no_noise_lasso_rec_nbr_0';
    fname2_no_noise = f'im_no_noise_lasso_rec_nbr_1';
    fname1_gauss1   = f'im_gauss1_lasso_rec_p_{round(1000*p)}_nbr_0';
    fname2_gauss1   = f'im_gauss1_lasso_rec_p_{round(1000*p)}_nbr_1';
    fname1_gauss2   = f'im_gauss2_lasso_rec_p_{round(1000*p)}_nbr_0';
    fname2_gauss2   = f'im_gauss2_lasso_rec_p_{round(1000*p)}_nbr_1';

    Image_im1_no_noise = Image.fromarray(np.uint8(255*np.abs(im1_no_noise)));
    Image_im2_no_noise = Image.fromarray(np.uint8(255*np.abs(im2_no_noise)));
    Image_im1_gauss1 = Image.fromarray(np.uint8(255*np.abs(im1_gauss1)));
    Image_im2_gauss1 = Image.fromarray(np.uint8(255*np.abs(im2_gauss1)));
    Image_im1_gauss2 = Image.fromarray(np.uint8(255*np.abs(im1_gauss2)));
    Image_im2_gauss2 = Image.fromarray(np.uint8(255*np.abs(im2_gauss2)));
    
    Image_im1_no_noise_zoom = Image.fromarray(np.uint8(255*np.abs(im1_no_noise[:zoom_size, -zoom_size:])));
    Image_im2_no_noise_zoom = Image.fromarray(np.uint8(255*np.abs(im2_no_noise[-zoom_size:, -zoom_size:])));
    Image_im1_gauss1_zoom = Image.fromarray(np.uint8(255*np.abs(im1_gauss1[:zoom_size, -zoom_size:])));
    Image_im2_gauss1_zoom = Image.fromarray(np.uint8(255*np.abs(im2_gauss1[-zoom_size:, -zoom_size:])));
    Image_im1_gauss2_zoom = Image.fromarray(np.uint8(255*np.abs(im1_gauss2[:zoom_size, -zoom_size:])));
    Image_im2_gauss2_zoom = Image.fromarray(np.uint8(255*np.abs(im2_gauss2[-zoom_size:, -zoom_size:])));

    Image_im1_no_noise.save(join(dest_plots, fname1_no_noise + '.png'));
    Image_im2_no_noise.save(join(dest_plots, fname2_no_noise + '.png'));
    Image_im1_gauss1.save(join(dest_plots, fname1_gauss1 + '.png'));
    Image_im2_gauss1.save(join(dest_plots, fname2_gauss1 + '.png'));
    Image_im1_gauss2.save(join(dest_plots, fname1_gauss2 + '.png'));
    Image_im2_gauss2.save(join(dest_plots, fname2_gauss2 + '.png'));
    
    Image_im1_no_noise_zoom.save(join(dest_plots, fname1_no_noise + '_zoom.png'));
    Image_im2_no_noise_zoom.save(join(dest_plots, fname2_no_noise + '_zoom.png'));
    Image_im1_gauss1_zoom.save(join(dest_plots, fname1_gauss1 + '_zoom.png'));
    Image_im2_gauss1_zoom.save(join(dest_plots, fname2_gauss1 + '_zoom.png'));
    Image_im1_gauss2_zoom.save(join(dest_plots, fname1_gauss2 + '_zoom.png'));
    Image_im2_gauss2_zoom.save(join(dest_plots, fname2_gauss2 + '_zoom.png'));



