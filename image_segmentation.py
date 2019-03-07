# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from imageio import imread
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    m,n = np.shape(A)
    D = np.zeros(n**2).reshape((n,n))
    A_sums = A.sum(axis=1)

    for i in range(n):
        D[i,i] = A_sums[i]

    return D - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    lap = laplacian(A)
    eigs = la.eig(lap)
    real_eigvals = np.real(eigs[0])
    count = 0

    for i in range(len(np.real(eigs[0]))):
        if abs(np.real(eigs[0])[i]) < tol:
            count += 1
    return count

# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        #Read image and scale for matplotlib
        self.image = imread(filename)
        self.scaled = self.image / 255

        #Check if color, if so change to grayscale
        if self.scaled.ndim == 3:
            self.brightness = self.scaled.mean(axis = 2)
            self.color = True
        else:
            self.brightness = self.scaled
            self.color = False

        #Unravel brightness into 1-D array
        self.M,self.N = self.brightness.shape
        self.flat_brightness = np.ravel(self.brightness)

        #Ensure flat brightness is correct shape, store as attribute
        if self.flat_brightness.size != self.M*self.N:
            raise ValueError("Flat brightness incorrect shape!")

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self.color == True:
            plt.imshow(self.image)
        elif self.color == False:
            plt.imshow(self.image, cmap="gray")

        plt.axis("off")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        A = sparse.lil_matrix((self.M*self.N, self.M*self.N))
        for i in range(self.M*self.N):
            neighbors, distances = get_neighbors(i, r, self.M, self.N)
            W = [np.exp(-1*abs(self.flat_brightness[i]-self.flat_brightness[neighbors[j]])/sigma_B2-abs(distances[j])/sigma_X2) for j in range(len(neighbors))]
            A[i, neighbors] = W

        D = np.array(A.sum(axis = 0))[0]
        return A.tocsc(), D


    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        lap = sparse.csgraph.laplacian(A)
        D_12 = sparse.diags(1/np.sqrt(D))

        DLD = D_12 @ lap @ D_12
        eig_vals, eig_vecs = sparse.linalg.eigsh(DLD, which="SM", k=2)
        X = eig_vecs[:,1].reshape(self.M,self.N)
        mask = X > 0
        return mask


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A,D)

        if self.color:
            mask = np.dstack((mask, mask, mask))
        pos_image = self.image*mask
        neg_image = self.image*~mask
        ax1 = plt.subplot(131)
        if self.color == False:
            ax1.imshow(self.image, cmap = "gray")
            plt.axis("off")
            ax2 = plt.subplot(132)
            ax2.imshow(pos_image, cmap = "gray")
            plt.axis("off")
            ax3 = plt.subplot(133)
            ax3.imshow(neg_image, cmap = "gray")
            plt.axis("off")

        else:
            ax1.imshow(self.image)
            plt.axis("off")
            ax2 = plt.subplot(132)
            ax2.imshow(pos_image)
            plt.axis("off")
            ax3 = plt.subplot(133)
            ax3.imshow(neg_image)
            plt.axis("off")
        plt.show()


#if __name__ == '__main__':
    #ImageSegmenter("dream_gray.png").segment()
    #ImageSegmenter("dream.png").segment()
    #ImageSegmenter("monument_gray.png").segment()
    #ImageSegmenter("monument.png").segment()
