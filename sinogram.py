import numpy as np
import matplotlib.pyplot as plt

"""
This program calculates the Radon transform of a grayscale square image in .txt format.  
You can adjust the number of projections with the variable "number_projection".

Don't forget to input the link to your image below [txt format] before running the program.
"""

number_projection = 180
path_image = '' ##################### path of the image 'image.txt'


# import image
with open(path_image, 'r') as file: # lien de l'image
    data = [float(value) for value in file.read().split()]

# dimension
square_length = int(np.sqrt(len(data)))

# Create a 2D array of the grayscale values of the image.
image = np.array(data).reshape((square_length, square_length))

# # exemple with square
# image = np.zeros((100, 100))
# image[10:50, 20:60] = 1

# # circle
# image = np.zeros((100, 100))
# y, x = np.ogrid[:100, :100]
# cercle = (x - 30) ** 2 + (y - 30) ** 2 <= 5 ** 2
# image[cercle] = 1


def radon(image, Nangles):
    M, N = image.shape # length image
    Nprojection = int(np.sqrt(M**2+N**2))+1 # length of diagonal
    theta = np.linspace(0, 180, Nangles, endpoint=False) # Projection angles vector

    sinogram = np.zeros((Nprojection, Nangles), dtype=np.float64) # Sinogram table filled with zeros

    cos = np.cos(theta*np.pi/180) # Tables of cosines and sines of the angles
    sin = np.sin(theta*np.pi/180)

    q_valeurs = np.arange(0,Nprojection) # Vector of points along the trajectory (maximum length = diagonal size)

    Xcenter = M/2 # centre of image
    Ycenter = N/2

    factor = M*np.sqrt(0.5) # only M because the image is square

    for t in range(Nangles): # Loop over all angles

        cost = cos[t] # Cosine and sine for angle t
        sint = sin[t]

        x0 = -cost*factor + Xcenter # Coordinates of the initial projection point
        y0 = -sint*factor + Ycenter

        for p in range(-int(Nprojection/2),int(Nprojection/2)): # loop for all projection
            x_val = np.round(x0 + q_valeurs*cost - p*sint).astype(int) # Transformation of the axes as functions of theta (vectors)
            y_val = np.round(y0 + p*cost + q_valeurs*sint).astype(int)

            indices = (x_val >= 0) & (x_val < M) & (y_val >= 0) & (y_val < N) # Only the pixels in the image

            # For each p, we add the grayscale value of each pixel along the trajectory (vector)
            sinogram[p+int(Nprojection/2), t] += np.sum(image[y_val[indices], x_val[indices]])
    
    sinogram /= M # normalisation

    return sinogram, theta




# Calculate the Radon transform
sinogram, vector_theta = radon(image, number_projection)

# Display the image and its sinogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title("Original image ")
ax1.imshow(image, cmap=plt.cm.Greys_r)

ax2.set_title("Sinogram (Radon Transform)")
ax2.set_xlabel("Angle (°)")
ax2.set_ylabel("Projections")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

plt.show()

