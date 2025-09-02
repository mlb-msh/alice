'''
# Sobel operator

import cv2
import numpy as np
from matplotlib import pyplot as plt

sample_img = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/optimus.jpg'

img = cv2.imread(sample_img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_img_array = np.array(gray_img)

gx_mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
gy_mask = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

img_rows, img_cols = gray_img.shape
mask_rows, mask_cols = gx_mask.shape


output_rows = img_rows - mask_rows + 1
output_cols = img_cols - mask_cols + 1


gx = np.zeros((output_rows, output_cols))


for i in range(output_rows):
    for j in range(output_cols):
        # Extract the region of interest (ROI) from the input array
        roi = gray_img[i:i+mask_rows, j:j+mask_cols]
        # Perform element-wise multiplication and sum
        gx[i, j] = np.sum(roi * gx_mask)

print(gx)


gy = np.zeros((output_rows, output_cols))
for i in range(output_rows):
    for j in range(output_cols):
        # Extract the region of interest (ROI) from the input array
        roi = gray_img[i:i+mask_rows, j:j+mask_cols]
        # Perform element-wise multiplication and sum
        gy[i, j] = np.sum(roi * gy_mask)

print(gy)


#Normalize
gx = gx / np.max(np.abs(gx))*255
gy = gy / np.max(np.abs(gy))*255

# Magnitude of the gradient
magnitude = np.sqrt(gx**2 + gy**2)
print("\nMagnitude of the gradient:")
print(magnitude)

#Thresholding
threshold =np.array(magnitude)
threshold_value = 75
threshold[threshold < threshold_value] = 0
threshold[threshold >= threshold_value] = 255
print("\nThresholding:")
print(threshold)

# Plotting the results
plt.figure(figsize=(90, 90))
plt.subplot(1, 5, 1)
plt.imshow(gray_img,cmap='gray')
plt.title('Original')
plt.axis('off')


plt.subplot(1, 5, 2)
plt.imshow(gx,cmap='gray')
plt.title('Gradient in x-direction (Gx)')
plt.axis('off')


plt.subplot(1, 5, 3)
plt.imshow(gy,cmap='gray')
plt.title('Gradient in y-direction (Gy)')
plt.axis('off')


plt.subplot(1, 5, 4)
plt.imshow(magnitude,cmap='gray')
plt.title('Edge Magnitude')
plt.axis('off')



plt.subplot(1, 5, 5)
plt.imshow(threshold,cmap='gray')
plt.title('Threshold')
plt.axis('off')

plt.tight_layout()
plt.show()
'''