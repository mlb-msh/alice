'''
# image convert from color to gray

from PIL import Image

img = Image.open("/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg")
graying = img.convert('L')
plt.axis('off')
plt.imshow(graying)
graying.show()
graying.save('grayimg.png')
print(img.height,img.width,img.mode,img.format,type(img))
img.show()

newimg = img.point(lambda i:i*20)
newimg.show()

'''


'''
#display image

from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt

img = image.imread('/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg')
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(img)

'''

'''
#seperate image and increase red component

from PIL import Image

img = Image.open("/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg")

red,green,blue = img.split()

#red.show()
#green.show()
#blue.show()


new_red = red.point(lambda i:i*4 )
#new_red.show()


modified_image = Image.merge('RGB', (new_red,green,blue))
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(modified_image)
modified_image.save('modified_red.jpg')

'''

'''
#resize image 

from PIL import Image

# Load an image from a file
img = Image.open('/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg')
print(f"Image size (width, height): {img.size}")
img_resized = img.resize((300,400))

plt.imshow(img_resized)


#flipping

img = Image.open('/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg')

flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)

plt.axis("off")
plt.imshow(flipped_img)

#blend image

blended_img = Image.blend(img,flipped_img,alpha = 0.5)
plt.axis('off')
plt.imshow(blended_img)

'''


'''
#combine image

from PIL import Image

# Get dimensions
width1, height1 = img.size
width2, height2 = flipped_img.size

# Create a new image with combined width and maximum height
combined_width = width1 + width2
combined_height = max(height1, height2)
merged_image = Image.new("RGB", (combined_width, combined_height), (255, 255, 255)) # White background

# Paste the flipped images
merged_image.paste(img, (width1, 0))
merged_image.paste(flipped_img, (0, 0))

plt.axis('off')
plt.imshow(merged_image)

'''

'''
#all basic operations

!pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
else:
    # Display the original image (OpenCV uses BGR color space)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 3, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Convert image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
    plt.title('HSV Image')
    plt.axis('off')

    # Basic operations: Resize
    resized_img = cv2.resize(img, (300, 200)) # width, height
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image (300x200)')
    plt.axis('off')

    # Basic operations: Flip
    flipped_img = cv2.flip(img, 1) # 0 for vertical, 1 for horizontal, -1 for both
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB))
    plt.title('Horizontally Flipped Image')
    plt.axis('off')

    # Basic operations: Rotate (using getRotationMatrix2D and warpAffine)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0) # rotate 90 degrees, scale 1.0
    rotated_img = cv2.warpAffine(img, M, (w, h))
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
    plt.title('Rotated Image (45 deg)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Saving converted images
    cv2.imwrite('grayscale_img.png', gray_img)
    cv2.imwrite('resized_img.jpg', resized_img)

    print("Image conversion and basic operations performed.")
    print("Grayscale image saved as 'grayscale_img.png'")
    print("Resized image saved as 'resized_img.jpg'") 

'''

'''
#crop image

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/sample_image.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
else:
    # Define the cropping region (x, y, width, height)
    x = 100  # Top-left corner x-coordinate
    y = 50   # Top-left corner y-coordinate
    w = 500  # Width of the cropped region
    h = 400  # Height of the cropped region

    # Crop the image
    # Note: OpenCV uses y:y+h, x:x+w for slicing
    cropped_img = img[y:y+h, x:x+w]

    # Display the original and cropped images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Original image shape: {img.shape}")
    print(f"Cropped image shape: {cropped_img.shape}")

    # Optionally save the cropped image
    # cv2.imwrite('cropped_image.jpg', cropped_img)
    # print("Cropped image saved as 'cropped_image.jpg'")
    
'''


'''
#Apply translation and rotation to an image.

import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
img = cv2.imread("/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/sample_image.jpg")

(h, w) = img.shape[:2]
tx, ty = 50, 30
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
translated_img = cv2.warpAffine(img, M_translate, (w, h))

center = (w // 2, h // 2)
angle = 45
M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_img = cv2.warpAffine(img, M_rotate, (w, h))


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('Off')

plt.show()

plt.imshow(cv2.cvtColor(translated_img, cv2.COLOR_BGR2RGB))
plt.title(f'Translated Image ({tx}, {ty})')
plt.axis('Off')

plt.show()

plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
plt.title(f'Rotated Image ({angle} degrees)')
plt.axis('Off')

plt.show()

'''

'''
# Load an image

# Convert it to grey scale

# create a mask image and the mask image all the pixels will contain the same value

# for each pixel in the original image you have to mask the last three bits should be 111 or 000 and apply(and/or operation)

# what is the quality of the image and also increase the bit from 3 bits


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load an image
img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg'
img = cv2.imread(img_path)


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask=np.full_like(gray_img,0b11111000)


masked_and_img = cv2.bitwise_and(gray_img, mask)
masked_or_img = cv2.bitwise_or(gray_img, mask)  #Not suitable for these operations because it doesn't give much information in higher orderbits

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(masked_and_img, cmap='gray')
plt.title('Image after Bitwise AND with Mask (keeping last 3 bits)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(masked_or_img, cmap='gray')
plt.title('Image after Bitwise OR with Mask (setting last 3 bits to 1)')
plt.axis('off')


plt.tight_layout()
plt.show()




print("Original grayscale pixel values (top-left corner):")
print(gray_img[:5, :5])
print("Masked pixel values (top-left corner):")
print(mask[:5, :5])
print("\nMasked (AND) pixel values (top-left corner):")
print(masked_and_img[:5, :5])
print("\nMasked (OR) pixel values (top-left corner):")
print(masked_or_img[:5, :5])

masked_and_img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/masked_and_img.png'
cv2.imwrite(masked_and_img_path, masked_and_img)
grey_img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/grey_img.png'
cv2.imwrite(grey_img_path, gray_img)
print(f"Masked image saved as '{masked_and_img_path}'")

def get_image_bytes(file_path):
    try:
        with open(file_path, "rb") as file:
            image_bytes = file.read()
        return image_bytes
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

image_data_bytes = get_image_bytes(grey_img_path)
masked_data_bytes = get_image_bytes(masked_and_img_path)

if image_data_bytes:
    print(f"Successfully read {len(image_data_bytes)} bytes from the image.")

if masked_data_bytes:
    print(f"Successfully read {len(masked_data_bytes)} bytes from the masked image.")

print(len(image_data_bytes)>len(masked_data_bytes))

'''


'''

# Image multiplication & Division

# Logical Operation (masking)

# Filters (mean, median, emboss)

# Quantization

# Basic Thresholding

# Isodata


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img1_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/modified1.jpg'
img1 = cv2.imread(img1_path)

img2_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/modified.jpg'
img2 = cv2.imread(img2_path)

gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#multiplication
multiplied_img = cv2.multiply(img1, img2)
plt.figure(figsize=(18, 12))
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(multiplied_img,cv2.COLOR_BGR2RGB))
plt.title('Image Multiplication')
plt.axis('off')

#division
divided_img = cv2.divide(multiplied_img, img2)
plt.subplot(3, 4, 2)
plt.imshow(cv2.cvtColor(divided_img,cv2.COLOR_BGR2RGB))
plt.title('Image Division')
plt.axis('off')

# Create a simple mask (a white rectangle on a black background)
mask = np.zeros_like(gray_img)
cv2.rectangle(mask, (100, 50), (600, 450), 255, -1) # Draw a filled white rectangle
masked_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)
plt.subplot(3, 4, 3)
plt.imshow(cv2.cvtColor(masked_img,cv2.COLOR_BGR2RGB))
plt.title('Masked Image (Rectangle Mask)')
plt.axis('off')

#Filters (Mean)
mean_filtered_img = cv2.blur(img1, (15, 15)) # Kernel size 15x15
plt.subplot(3, 4, 5)
plt.imshow(cv2.cvtColor(mean_filtered_img,cv2.COLOR_BGR2RGB))
plt.title('Mean Filter (15x15)')
plt.axis('off')

#Filters (Median)
median_filtered_img = cv2.medianBlur(img1, 15) # Kernel size 15 (must be odd)
plt.subplot(3, 4, 6)
plt.imshow(cv2.cvtColor(median_filtered_img,cv2.COLOR_BGR2RGB))
plt.title('Median Filter (15x15)')
plt.axis('off')

#Filters (Emboss)
emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])
embossed_img = cv2.filter2D(img1, -1, emboss_kernel)
plt.subplot(3, 4, 7)
plt.imshow(cv2.cvtColor(embossed_img,cv2.COLOR_BGR2RGB))
plt.title('Emboss Filter')
plt.axis('off')


#Quantization
num_levels = 16
quantization_step = 256 // num_levels
quantized_img = (gray_img // quantization_step) * quantization_step
plt.subplot(3, 4, 9)
plt.imshow(gray_img,cmap = 'gray',vmin=0,vmax=255)
plt.title(f'Quantized Image ({num_levels} levels)')
plt.axis('off')


# Basic Thresholding
threshold_value = 127
ret, basic_threshold_img = cv2.threshold(img1, threshold_value, 255, cv2.THRESH_BINARY)
plt.subplot(3, 4, 10)
plt.imshow(cv2.cvtColor(basic_threshold_img,cv2.COLOR_BGR2RGB))
plt.title(f'Basic Thresholding (>{threshold_value})')
plt.axis('off')

# --- Isodata Thresholding (Otsu's Binarization) ---
# OpenCV's THRESH_OTSU automatically finds the optimal threshold value
ret_otsu, otsu_threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.subplot(3, 4, 11)
plt.imshow(otsu_threshold_img,cmap = 'gray',vmin=0,vmax=255)
plt.title(f'Otsu Thresholding (Threshold: {ret_otsu:.2f})')
plt.axis('off')


plt.tight_layout()
plt.show()


'''