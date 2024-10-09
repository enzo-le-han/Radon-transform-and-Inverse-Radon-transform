import numpy as np
import matplotlib.pyplot as plt

"""
This program calculates the Radon transform of a square grayscale image in .txt format.  
You can adjust the number of projections with the variable "number_projection."  
Then, the filtered back projection of the Radon transform is calculated. You can change the filter using the variable "filter."  
The Fourier transforms used in the reconstruction part are computed with the `scipy.fft` module.

Available filters: "square," "ramp," "shepp-logan," "hamming," "hann," "cosine."  
The "square" filter produces a simple back projection, meaning there is no filter applied.

Don't forget to input the link to your image below [txt format] before running the program.
"""

number_projection = 180
filter = 'hamming'
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






"""
Backpropagation / Inverse Radon Transform
"""
from scipy.fft import fft, ifft, fftfreq, fftshift # Fast Fourier Transform algorithm


def filter_function(length,filter):

    n = np.concatenate(
        (
            # Generate a sequence of odd numbers ranging from 1 to (length/2 + 1) (exclusive), with a step of 2. This sequence represents the low spatial frequencies.
            np.arange(1, length / 2 + 1, 2, dtype=int),
            # Generate a sequence of odd numbers ranging from (length/2 - 1) to 1 (exclusive), with a step of -2. This sequence represents the high spatial frequencies.
            np.arange(length / 2 - 1, 0, -2, dtype=int),
        ))

    f = np.zeros(length)
    f[0] = 0.25 # Smooth the transition to zero of the filter's frequency response.

    # create the ramp filter from its Fourier transform.
    f[1::2] = -1 / (np.pi * n)**2

    fourier_filter = 2 * np.real(fft(f))

    if filter == "ramp":
        pass

    elif filter == "shepp-logan":
        # Start with the first element to avoid division by zero.
        omega = np.pi * fftfreq(length)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega

    elif filter == "cosine":
        freq = np.linspace(0, np.pi, length, endpoint=False)
        cosine_filter = fftshift(np.sin(freq))
        fourier_filter *= cosine_filter

    elif filter == "hamming":
        fourier_filter *= fftshift(np.hamming(length))

    elif filter == "hann":
        fourier_filter *= fftshift(np.hanning(length))

    elif filter == 'square': # simple backprojection 
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]



def inverse_radon(image_radon,vector_theta,filter):

    image_radon_length = image_radon.shape[0]
    dtype = image_radon.dtype

    # Ensure that the resulting size will be a power of two, which is optimal for the FFT.
    length_projection = max(64, int(2 ** np.ceil(np.log2(2 * image_radon_length))))
    width = ((0, length_projection - image_radon_length), (0, 0)) # Number of rows to add until the specified size is reached.
    img = np.pad(image_radon, width, mode='constant', constant_values=0) # Fill the new rows with zeros to reach the power of two.

    # Apply filter
    fourier_filter = filter_function(length_projection,filter)
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:image_radon_length, :])
    
    # The rest of the code is similar to the radon() function, except for the inversion of the last line.
    new_length = int(image_radon_length/np.sqrt(2))
    reconstructed_image = np.zeros((new_length,new_length), dtype=dtype)

    cos = np.cos(vector_theta*np.pi/180)
    sin = np.sin(vector_theta*np.pi/180)

    q_valeurs = np.arange(0,image_radon_length)

    Xcentre = new_length/2
    Ycentre = new_length/2

    facteur = new_length*np.sqrt(0.5)

    for t in range(len(vector_theta)):

        cost = cos[t]
        sint = sin[t]

        x0 = -cost*facteur + Xcentre
        y0 = -sint*facteur + Ycentre

        for p in range(-int(image_radon_length/2),int(image_radon_length/2)):         
            x_val = np.round(x0 + q_valeurs*cost - p*sint).astype(int)
            y_val = np.round(y0 + p*cost + q_valeurs*sint).astype(int)

            indices_valides = (x_val >= 0) & (x_val < new_length) & (y_val >= 0) & (y_val < new_length)

            reconstructed_image[y_val[indices_valides], x_val[indices_valides]] += radon_filtered[p+int(image_radon_length/2), t]

    return reconstructed_image  * np.pi / (2 * len(vector_theta)) # normalised


# Calculate the inverse Radon transform
reconstructed_image = inverse_radon(sinogram,vector_theta,filter)

# Display the original and reconstructed images.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title("Original image")
ax1.imshow(image, cmap=plt.cm.Greys_r)

ax2.set_title("Reconstructed image")
ax2.imshow(reconstructed_image, cmap=plt.cm.Greys_r)

plt.show()
