from PIL import Image
from scipy.io import loadmat
import numpy as np
from os.path import join
import os

N = 128
p = 0.06;

src_data = 'data_random'
dest_plots = 'plots_random'

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots)

data_auto  = loadmat(join(src_data, f'noise_gauss_{round(1000*p):d}_automap.mat'))
data_lasso  = loadmat(join(src_data, f'noise_gauss_{round(1000*p):d}_lasso.mat'))


true_im = np.squeeze(data_auto['mri_data'])
rec_auto_no_noise = np.squeeze(data_auto['rec_no_noise']);
rec_auto_gauss_noise = np.squeeze(data_auto['rec_gauss_noise']);
rec_lasso_no_noise = np.squeeze(data_lasso['rec_no_noise']);
rec_lasso_gauss_noise = np.squeeze(data_lasso['rec_gauss_noise']);


recs = np.zeros([4,N,N]);
recs[0, :, :] = rec_auto_no_noise
recs[1, :, :] = rec_auto_gauss_noise
recs[2, :, :] = rec_lasso_no_noise.astype(np.float64)
recs[3, :, :] = rec_lasso_gauss_noise.astype(np.float64)


max_diff = []
for i in range(4):
    diff = np.abs(true_im - recs[i,:,:])
    max_diff.append(np.amax(diff))

max_err = max(max_diff)
print(max_err)

diff_auto_no_noise = np.abs(true_im - rec_auto_no_noise)
diff_auto_gauss_noise = np.abs(true_im - rec_auto_gauss_noise)
diff_lasso_no_noise = np.abs(true_im - rec_lasso_no_noise)
diff_lasso_gauss_noise = np.abs(true_im - rec_lasso_gauss_noise)

pil_diff_auto_no_noise = Image.fromarray(np.uint8(255*(1-diff_auto_no_noise/max_err)))
pil_diff_auto_gauss_noise = Image.fromarray(np.uint8(255*(1-diff_auto_gauss_noise/max_err)))
pil_diff_lasso_no_noise = Image.fromarray(np.uint8(255*(1-diff_lasso_no_noise/max_err)))
pil_diff_lasso_gauss_noise = Image.fromarray(np.uint8(255*(1-diff_lasso_gauss_noise/max_err)))

pil_diff_auto_no_noise.save(join(dest_plots, f'err_auto_p_{round(1000*p):d}_no_noise.png'))
pil_diff_auto_gauss_noise.save(join(dest_plots, f'err_auto_p_{round(1000*p):d}_gauss_noise.png'))
pil_diff_lasso_no_noise.save(join(dest_plots, f'err_lasso_p_{round(1000*p):d}_no_noise.png'))
pil_diff_lasso_gauss_noise.save(join(dest_plots, f'err_lasso_p_{round(1000*p):d}_gauss_noise.png'))


