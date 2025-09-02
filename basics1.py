'''
#imports

import numpy as np
import matplotlib.pyplot as plt
import math

'''


'''
#black image with a white dot and sample the row

img = np.zeros([256,256])
img[128,128] = 255
plt.imshow(img, cmap = 'gray',vmin=0, vmax=256)

img = np.zeros([256,256])
img[128,128] = 255
rowToDisplay = img[128,:]
plt.plot(rowToDisplay)

'''


'''
#random_value_img

img = np.zeros([256,256])
for i in range(256):
    for j in range(256):
        img[i,j] = np.random.randint(0,256)

plt.imshow(img,vmin=0, vmax=256)  #prints the image
rowToDisplay = img[128,:]
#plt.plot(rowToDisplay)  #samples a row

'''

'''
#white diagonal x mark


img = np.zeros([256,256])
for i in range(256):
    for j in range(256):
        if i == j or i+j == 255:
            img[i,j] = 255
plt.imshow(img, cmap = 'gray',vmin=0, vmax=256)  #prints the image
rowToDisplay = img[120,:]
#plt.plot(rowToDisplay)
'''

'''
#create a black image with a square in the centre with side of 10 units


img = np.ones([256,256])
tp = 128 - 10
bt = 128+10
lt = 128 - 10
rt = 128+10

for i in range(256):
    for j in range(256):
        if (i<=rt and i>=lt) and (j<=bt and j>=tp):
            img[i,j] = 0
        else:
            img[i,j] = 255
plt.imshow(img, cmap = 'gray',vmin=0, vmax=256)

'''


'''
#create a black image with a circle in the centre with radius of 10 units


img = np.ones([256,256])

for i in range(256):
    for j in range(256):
        if math.sqrt((128-i)**2 + (128-j)**2)<=10:
            img[i,j] = 0
        else:
            img[i,j] = 255
plt.imshow(img, cmap = 'gray',vmin=0, vmax=256)
'''

'''
#create a square with all 256 shades in it

side = int(math.sqrt(256))
print(side)

img = np.zeros([side,side])
intensity = 0
for i in range(side):
    for j in range(side):
        img[i,j] = intensity
        intensity+=1

plt.imshow(img, cmap = 'gray',vmin = 0,vmax=255)

'''


'''
#create a elipse of formula ((x-xc)/a)^2 + ((y-yc)/b)^2 = r^2 with r = 10

a = 40
b = 20
img = np.zeros([256,256])
for i in range(256):
    for j in range(256):
        if (((i-128)/a)**2 + ((j-128)/b)**2) <=1:
            img[i,j] = 0
        else:
            img[i,j] = 255
plt.imshow(img, cmap = 'gray',vmin = 0,vmax=255)

'''


'''
#Black and white color gradient

img = np.zeros([256,256])
img[: ,0:256] = np.arange(0,256,1)
plt.imshow(img,cmap='gray')

rowToDisplay = img[0,:]
plt.plot(rowToDisplay)

'''

'''
#Generate a gradient image (dark - light -dark)

img = np.zeros([256,256])
img[: ,128:256] = np.arange(128,0,-1)
img[: ,0:128] = np.arange(0,128,1)
plt.imshow(img,cmap='gray')

rowToDisplay = img[0,:]
plt.plot(rowToDisplay)
'''


'''
#mesh multiple circles


x = np.linspace(-128,128,256)
y = np.linspace(-128,128,256)
x,y = np.meshgrid(x,y)

img = np.cos(256*np.pi*(x**2 + y**2))
plt.imshow(img, cmap='gray')
'''