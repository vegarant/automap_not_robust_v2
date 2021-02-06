from PIL import Image
from scipy.io import loadmat
import numpy as np
from os.path import join
import os

HCP_nbr1 = 1002
HCP_nbr2 = 1033

im_nbrs1 = [37, 39, 43, 49]
im_nbrs2 = [2]

HCP = [HCP_nbr1]*len(im_nbrs1) + [HCP_nbr2]*len(im_nbrs2) 
im_nbrs = im_nbrs1 + im_nbrs2

N = 128

src_data = 'data_error_map'
dest_plots = 'plots_error_map'

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots)

data1_automap = loadmat(join(src_data, f'im_rec_automap_HCP_{HCP_nbr1}.mat'))
data2_automap = loadmat(join(src_data, f'im_rec_automap_HCP_{HCP_nbr2}.mat'))
data1_lasso = loadmat(join(src_data, f'im_rec_lasso_HCP_{HCP_nbr1}.mat'))
data2_lasso = loadmat(join(src_data, f'im_rec_lasso_HCP_{HCP_nbr2}.mat'))

mri_data1 =  data1_automap['mri_data']
mri_data2 =  data2_automap['mri_data']

im_rec_auto1 = data1_automap['im_rec']
im_rec_auto2 = data2_automap['im_rec']
print(im_rec_auto2.shape)

im_rec_lasso1 = data1_lasso['lasso_im_rec']
im_rec_lasso2 = data2_lasso['lasso_im_rec']

number_of_images = len(im_nbrs1) + len(im_nbrs2)
all_images = np.zeros([number_of_images, N, N])
all_recs_lasso = np.zeros([number_of_images, N, N])
all_recs_auto = np.zeros([number_of_images, N, N])

for i in range(len(im_nbrs1)):
    print(np.amax(mri_data1[im_nbrs1[i]]))
    all_images[i, :, :] = mri_data1[im_nbrs1[i], :,:]
    all_recs_lasso[i, :, :] = np.abs(im_rec_lasso1[im_nbrs1[i], :,:]).astype('float32')
    all_recs_auto[i, :, :] = im_rec_auto1[im_nbrs1[i], :,:]

for i in range(len(im_nbrs2)):
    print(np.amax(mri_data2[im_nbrs2[i]]))
    all_images[i+len(im_nbrs1), :, :] = mri_data2[im_nbrs2[i], :,:]
    all_recs_lasso[i+len(im_nbrs1), :, :] = np.abs(im_rec_lasso2[im_nbrs2[i], :, :]).astype('float32')
    all_recs_auto[i+len(im_nbrs1), :, :] = im_rec_auto2[im_nbrs2[i], :,:]

max_diff = []
for i in range(number_of_images):
    diff_automap = np.abs(all_recs_auto[i] - all_images[i])
    diff_lasso   = np.abs(all_recs_lasso[i] - all_images[i])

    max_diff_auto = np.amax(diff_automap)
    max_diff_lasso = np.amax(diff_lasso);
    max_diff.append(max_diff_auto)
    max_diff.append(max_diff_lasso)

print(max_diff)
max_err = max(max_diff)
for i in range(number_of_images):
    print(np.amax(all_recs_auto[i]))
    diff_automap = np.abs(all_recs_auto[i] - all_images[i])
    diff_lasso   = np.abs(all_recs_lasso[i] - all_images[i])

    diff_im_automap = 1 - (diff_automap/max_err);
    diff_im_lasso   = 1 - (diff_lasso/max_err);

    pil_diff_im_auto  = Image.fromarray(np.uint8(255*diff_im_automap));
    pil_diff_im_lasso = Image.fromarray(np.uint8(255*diff_im_lasso));
    
    
    pil_diff_im_auto.save(join(dest_plots, f'error_map_automap_HCP_{HCP[i]}_{im_nbrs[i]}.png'));
    pil_diff_im_lasso.save(join(dest_plots, f'error_map_lasso_HCP_{HCP[i]}_{im_nbrs[i]}.png'));


#        bd = 5
#        im_out = np.ones([2*N+bd, 2*N+bd]);
#
#        diff_automap = np.abs(automap_im_rec[i] - mri_data[i])
#        diff_lasso = np.abs(lasso_im_rec[i] - mri_data[i])
#
#        max_diff_auto = np.amax(diff_automap)
#        max_diff_lasso = np.amax(diff_lasso);
#        max_err = max(max_diff_auto, max_diff_lasso);
#        
#        diff_im_automap = 1 - (diff_automap/max_err);
#        diff_im_lasso   = 1 - (diff_lasso/max_err);
#
#        im_out[:N,:N] = automap_im_rec[i]
#        im_out[:N,N+bd:] = lasso_im_rec[i]
#        im_out[N+bd:,:N] = diff_im_automap
#        im_out[N+bd:,N+bd:] = diff_im_lasso
#
#        pil_im = Image.fromarray(np.uint8(255*im_out));
#        pil_im.save(join(dest_plots, f'error_map_HCP_{HCP_nbr}_{i:03d}.png'));


