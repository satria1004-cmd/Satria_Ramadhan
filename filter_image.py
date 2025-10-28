import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.util import random_noise
from skimage.filters import rank
from skimage.morphology import disk
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import imageio
from skimage.color import rgb2gray

image_path = 'D:/FOLDER DHAN/TUGAS KULIAH/SEMESTER 5/CITRA DIGITAL/PROJECT UTS/monalisa.jpg'  
image = img_as_float(rgb2gray(imageio.imread(image_path)))  

noisy = random_noise(image, mode='s&p', amount=0.1)  # Salt & pepper noise

# Rank filters membutuhkan input citra 8-bit
noisy_ubyte = img_as_ubyte(noisy)

mean_filtered = rank.mean(noisy_ubyte, footprint=disk(3))
min_filtered = rank.minimum(noisy_ubyte, footprint=disk(3))
median_filtered = rank.median(noisy_ubyte, footprint=disk(3))
max_filtered = rank.maximum(noisy_ubyte, footprint=disk(3))

# Kembalikan ke float [0,1] untuk evaluasi
mean_filtered_float = img_as_float(mean_filtered)
min_filtered_float = img_as_float(min_filtered)
median_filtered_float = img_as_float(median_filtered)
max_filtered_float = img_as_float(max_filtered)

metrics = {
   "Noisy": {
       "PSNR": psnr(image, noisy),
       "SSIM": ssim(image, noisy, data_range=1.0)
   },
   "Mean Filtered": {
       "PSNR": psnr(image, mean_filtered_float),
       "SSIM": ssim(image, mean_filtered_float, data_range=1.0)
   },
   "Min Filtered": {
       "PSNR": psnr(image, min_filtered_float),
       "SSIM": ssim(image, min_filtered_float, data_range=1.0)
   },
   "Median Filtered": {
       "PSNR": psnr(image, median_filtered_float),
       "SSIM": ssim(image, median_filtered_float, data_range=1.0)
   },
   "Max Filtered": {
       "PSNR": psnr(image, max_filtered_float),
       "SSIM": ssim(image, max_filtered_float, data_range=1.0)
   }
}

fig, axes = plt.subplots(1, 6, figsize=(15, 5))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')

ax[1].imshow(noisy, cmap='gray')
ax[1].set_title("Noisy")
ax[1].axis('off')

ax[2].imshow(mean_filtered, cmap='gray')
ax[2].set_title("Mean Filter")
ax[2].axis('off')

ax[3].imshow(min_filtered, cmap='gray')
ax[3].set_title("Min Filter")
ax[3].axis('off')

ax[4].imshow(median_filtered, cmap='gray')
ax[4].set_title("Median Filter")
ax[4].axis('off')

ax[5].imshow(max_filtered, cmap='gray')
ax[5].set_title("Max Filter")
ax[5].axis('off')
plt.tight_layout()
plt.show()

for name, vals in metrics.items():
   print(f"{name} -> PSNR: {vals['PSNR']:.2f}, SSIM: {vals['SSIM']:.4f}")
