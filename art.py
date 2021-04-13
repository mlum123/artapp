# contains all functions used to convert an image to different styles of art

import cv2
# print (cv2. __version__)                           # check OpenCV version

def oil_painting(img):
    """
    converts passed-in image to oil painting
    returns oil painting version of image
    """
    res = cv2.xphoto.oilPainting(img, 7, 1)              # make image look like oil painting
    return res

def watercolor(img):
    """
    converts passed-in image to watercolor
    returns watercolor version of image
    """
    # make image look like watercolor painting
    # sigma_s controls the size of the neighborhood: range 1 - 200
    # sigma_r controls how dissimilar colors within the neighborhood will be averaged
    # a larger sigma_r results in large regions of constant color: range 0 - 1
    res = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return res

def pencil_sketch_bw(img):
    """
    converts passed-in image to pencil sketch
    returns black-and-white pencil sketch version of image
    """
    # make image look like pencil sketch
    # sigma_s controls the size of the neighborhood: range 1 - 200
    # sigma_r controls how dissimilar colors within the neighborhood will be averaged
    # a larger sigma_r results in large regions of constant color: range 0 - 1
    # shade_factor is a simple scaling of the output image intensity,
    # the higher the shade_factor, the brighter the result: range 0 - 0.1
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return dst_gray

def pencil_sketch_color(img):
    """
    converts passed-in image to pencil sketch
    returns color pencil sketch version of image
    """
    # make image look like pencil sketch
    # sigma_s controls the size of the neighborhood: range 1 - 200
    # sigma_r controls how dissimilar colors within the neighborhood will be averaged
    # a larger sigma_r results in large regions of constant color: range 0 - 1
    # shade_factor is a simple scaling of the output image intensity,
    # the higher the shade_factor, the brighter the result: range 0 - 0.1
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return dst_color

#####################
# Pointillist Painting
import scipy.spatial
import numpy as np
import random
import math
from sklearn.cluster import KMeans

def compute_color_probabilities(pixels, palette):
    """
    compute_color_probabilities calculates and returns the probabilities of similarities of colors
    """
    distances = scipy.spatial.distance.cdist(pixels, palette)
    maxima = np.amax(distances, axis=1)
    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    return distances

def get_color_from_prob(probabilities, palette):
    """
    get_color_from_prob returns the most similar color based on probabilities passed in
    """
    probs = np.argsort(probabilities)
    i = probs[-1]
    return palette[i]

def randomized_grid(h, w, scale):
    """
    randomized_grid creates a grid of positions for dots by sampling uniformly
    and shuffles it to be in random order

    returns the resulting grid
    """
    assert (scale > 0)
    r = scale//2

    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid

def get_color_palette(img, n=20):
    """
    get_color_palette uses KMeans to return the n most used colors in img
    default: color palette of 20 colors
    """
    clt = KMeans(n_clusters=n)
    clt.fit(img.reshape(-1, 3))
    return clt.cluster_centers_

def complement(colors):
    """
    complement returns a palette that's the complement of the colors palette passed in
    """
    return 255 - colors

def pointillize(img, primary_colors=20):
    """
    pointillize is a function that takes in an image and the num of colors we want to use in our palette (default: 20)
    returns pointillized version of image
    """
    radius_width = int(math.ceil(max(img.shape) / 1000))                        # calculate dot radius size from image size
    palette = get_color_palette(img, primary_colors)                            # get a palette of primary_colors num of most used colors in image
    complements = complement(palette)                                           # get complement colors
    palette = np.vstack((palette, complements))
    canvas = img.copy()

    grid = randomized_grid(img.shape[0], img.shape[1], scale=3)                 # create a randomized grid of positions where to paint dots

    pixel_colors = np.array([img[x[0], x[1]] for x in grid])                    # create numpy array of colors for each pixel in img
    
    color_probabilities = compute_color_probabilities(pixel_colors, palette)    # calculate probabilities of similarities of colors

    # color each pixel so it's similar to the pixel in original image
    for i, (y, x) in enumerate(grid):
        # compute probabilities of similarities of colors, take the most similar color
        color = get_color_from_prob(color_probabilities[i], palette)

        # draw a dot of that color in this position
        cv2.ellipse(canvas, (x, y), (radius_width, radius_width), 0, 0, 360, color, -1, cv2.LINE_AA)
    
    return canvas