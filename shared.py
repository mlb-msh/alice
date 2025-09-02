'''
#Basic Operations

# prompt: Use opencv to convert one image to another type and basic operations as the image in the location ('/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/img.jpg')

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
#zero and first order hold

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to convert RGB to Grayscale using cv2
def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function for Zero Order Hold
def zero_order_hold(gray_image, scale_factor):
    h, w = gray_image.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    zoh_image = np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            zoh_image[i, j] = gray_image[i // scale_factor, j // scale_factor]

    return zoh_image

# Function for First Order Hold
def first_order_hold(gray_image, scale_factor):
    h, w = gray_image.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    foh_image = np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            x = i / scale_factor
            y = j / scale_factor
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, h - 1), min(y1 + 1, w - 1)

            # Linear interpolation
            foh_image[i, j] = (
                gray_image[x1, y1] * (x2 - x) * (y2 - y) +
                gray_image[x2, y1] * (x - x1) * (y2 - y) +
                gray_image[x1, y2] * (x2 - x) * (y - y1) +
                gray_image[x2, y2] * (x - x1) * (y - y1)
            )

    return foh_image

# Load the image using cv2
image_path = 'D:/sem9/cv/sunflower.jpeg'  # Replace with your image path
rgb_image = cv2.imread(image_path)

# Convert to grayscale
gray_image = rgb_to_grayscale(rgb_image)

# Define scale factor
scale_factor = 4

# Apply ZOH and FOH
zoh_image = zero_order_hold(gray_image, scale_factor)
foh_image = first_order_hold(gray_image, scale_factor)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Grayscale")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Zero Order Hold")
plt.imshow(zoh_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("First Order Hold")
plt.imshow(foh_image, cmap='gray')
plt.axis('off')

plt.show()

'''

'''
#histogram

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Step 1: Load grayscale image ----------------
img = cv2.imread("/content/drive/MyDrive/images.jpeg", cv2.IMREAD_GRAYSCALE)

# ---------------- Step 2: Compute histogram (frequency) ----------------
freq = np.zeros(256, dtype=int)   # for 8-bit grayscale
rows, cols = img.shape

for i in range(rows):
    for j in range(cols):
        freq[img[i, j]] += 1

# ---------------- Step 3: PMF (Probability Mass Function) ----------------
total_pixels = rows * cols
pmf = freq / total_pixels

# ---------------- Step 4: CDF (Cumulative Distribution Function) ----------------
cdf = np.cumsum(pmf)

# ---------------- Step 5: Mapping function ----------------
L = 256  # total gray levels
mapping = np.round(cdf * (L - 1)).astype(np.uint8)
print(mapping)
# ---------------- Step 6: Apply mapping ----------------
equalized_img = np.zeros_like(img, dtype=np.uint8)
print(equalized_img)
for i in range(rows):
    for j in range(cols):
        equalized_img[i, j] = mapping[img[i, j]]

# ---------------- Step 7: Visualization ----------------
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")

plt.subplot(2, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_img, cmap="gray")

plt.subplot(2, 2, 3)
plt.title("Original Histogram")
plt.hist(img.flatten(), bins=256, range=[0, 256], color='blue')

plt.subplot(2, 2, 4)
plt.title("Equalized Histogram")
plt.hist(equalized_img.flatten(), bins=256, range=[0, 256], color='green')

plt.tight_layout()
plt.show()

'''