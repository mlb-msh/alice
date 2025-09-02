import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = Image.open("/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/sample_image.jpg")

red_img,green_img,blue_img = img.split()
red = np.array(red_img)
green = np.array(green_img)
blue = np.array(blue_img)

l = red.shape[0]
b = red.shape[1]

red_new = np.zeros((l,2*b+1))
green_new = np.zeros((l,2*b+1))
blue_new = np.zeros((l,2*b+1))

for i in range(l):
  for j in range(b):
    red_new[i,2*j+1] = red[i,j]
    green_new[i,2*j+1] = green[i,j]
    blue_new[i,2*j+1] = blue[i,j]

    if 2*j+2 == 2*b:
      break
    else:
      red_new[i,2*j+2] = red[i,j]
      green_new[i,2*j+2] = green[i,j]
      blue_new[i,2*j+2] = blue[i,j]
print(red_new)

b = red_new.shape[1]
red_new_col = np.zeros((2*l+1,b))
green_new_col = np.zeros((2*l+1,b))
blue_new_col = np.zeros((2*l+1,b))

for i in range(l):
  for j in range(b):
    red_new_col[2*i+1,j] = red_new[i,j]
    green_new_col[2*i+1,j] = green_new[i,j]
    blue_new_col[2*i+1,j] = blue_new[i,j]
    if 2*i+2 == 2*l:
      break
    else:
      red_new_col[2*i+2,j] = red_new[i,j]
      green_new_col[2*i+2,j] = green_new[i,j]
      blue_new_col[2*i+2,j] = blue_new[i,j]

print(red_new_col)

modified_image = Image.merge('RGB', (Image.fromarray(red_new_col.astype(np.uint8)),
                                     Image.fromarray(green_new_col.astype(np.uint8)),
                                     Image.fromarray(blue_new_col.astype(np.uint8))))
plt.figure()
plt.axis('off')
plt.imshow(modified_image)
modified_image.save('/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/modified1.jpg')