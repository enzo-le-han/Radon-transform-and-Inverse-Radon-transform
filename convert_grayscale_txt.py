import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

"""
This code allows you to convert a jpg or png image into grayscale in a .txt file.

Use a square image
Don't forget to input the link to your image below before running the program.
Don't forget to input the link to the folder where the grayscale image will be saved before running the program.
Don't forget to select the format of your image below.
"""

path_image = '' ##################### path of the image
save_folder_link = ''

# Image format (1 = yes, 0 = no)
jpg = 1
png = 0


if jpg == 1 :
    img = Image.open(path_image)
    img_array = np.array(img)
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

if png == 1 : 
    import matplotlib.image as mpimg
    img = mpimg.imread(path_image)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]


imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

# Visualize the grayscale image.
# plt.imshow(imgGray,cmap=plt.cm.Greys_r)
# plt.show()

# Save the grayscale image in .txt format.
np.savetxt(save_folder_link + "/ImgGray.txt",imgGray)



 
