'''
#canny edge detection


import cv2
import numpy as np
from matplotlib import pyplot as plt

sample_img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/rolls.jpg'

input_image = cv2.imread(sample_img_path)

if input_image is None:
    print(f"Error: Could not load image from {sample_img_path}")
else:
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy_mask = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_rows, img_cols = grayscale_image.shape
    mask_rows, mask_cols = gx_mask.shape

    output_rows = img_rows - mask_rows + 1
    output_cols = img_cols - mask_cols + 1

    gradient_x = np.zeros((output_rows, output_cols))
    gradient_y = np.zeros((output_rows, output_cols))

    for i in range(output_rows):
        for j in range(output_cols):
            roi = grayscale_image[i:i + mask_rows, j:j + mask_cols]
            gradient_x[i, j] = np.sum(roi * gx_mask)
            gradient_y[i, j] = np.sum(roi * gy_mask)


    gradient_strength = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_orientation = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    gradient_orientation[gradient_orientation < 0] += 180

    rows, cols = gradient_strength.shape
    suppressed_output = np.zeros_like(gradient_strength)

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            angle_deg = gradient_orientation[r, c]

            # calculating the gradient direction
            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                if gradient_strength[r, c] >= gradient_strength[r, c + 1] and gradient_strength[r, c] >= gradient_strength[r, c - 1]:
                    suppressed_output[r, c] = gradient_strength[r, c]
            elif (22.5 <= angle_deg < 67.5):
                if gradient_strength[r, c] >= gradient_strength[r - 1, c + 1] and gradient_strength[r, c] >= gradient_strength[r + 1, c - 1]:
                    suppressed_output[r, c] = gradient_strength[r, c]
            elif (67.5 <= angle_deg < 112.5):
                if gradient_strength[r, c] >= gradient_strength[r - 1, c] and gradient_strength[r, c] >= gradient_strength[r + 1, c]:
                    suppressed_output[r, c] = gradient_strength[r, c]
            else: # 112.5 <= angle_deg < 157.5
                if gradient_strength[r, c] >= gradient_strength[r - 1, c - 1] and gradient_strength[r, c] >= gradient_strength[r + 1, c + 1]:
                    suppressed_output[r, c] = gradient_strength[r, c]

    high_threshold_ratio_val = 0.25
    low_threshold_ratio_val = 0.05

    high_threshold_val = np.max(suppressed_output) * high_threshold_ratio_val
    low_threshold_val = high_threshold_val * low_threshold_ratio_val

    final_edges = np.zeros_like(suppressed_output, dtype=np.uint8)


    strong_edge_rows, strong_edge_cols = np.where(suppressed_output >= high_threshold_val)
    weak_edge_rows, weak_edge_cols = np.where((suppressed_output >= low_threshold_val) & (suppressed_output < high_threshold_val))

    final_edges[strong_edge_rows, strong_edge_cols] = 255


    for row, col in zip(weak_edge_rows, weak_edge_cols):
        if 255 in final_edges[row-1:row+2, col-1:col+2]:
            final_edges[row, col] = 255


    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.imshow(final_edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.tight_layout()
'''