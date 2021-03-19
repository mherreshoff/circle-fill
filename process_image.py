#!/usr/bin/env python3

# To run this you need to install some python3 packages:
# pip3 install networkx numpy scipy pillow
# Should get them all for you.

import sys

import networkx as nx
import numpy as np
import scipy
import scipy.optimize
from PIL import Image, ImageDraw


def image_to_array(image):
    """
    Converts an image to a 3 column matrix whose columns are the
    * the x coordinate,
    * y coordinate, and 
    * the brightness of each pixel.
    """
    width, height = image.size
    ycoords, xcoords = np.mgrid[0:height,0:width]
    greys = np.asarray(image.convert("L"), dtype=np.float32) / 255
    labeled_matrix = np.dstack([xcoords, ycoords, greys])
    return labeled_matrix.reshape(-1, labeled_matrix.shape[2])

def consecutive(arr):
    """ Given an array of integers, return the subset whose right neighbors are one higher.

    For example, if arr is [1,2,3,7,8,100], the result will be [1,2,7]
    """
    consecutive_idx = np.concatenate([arr[:-1] + 1 == arr[1:], [False]])
    return arr[consecutive_idx]

def horizontal_edges(coords):
    """
    Given a two column matrix of coordinates, find all the pairs of coordinates
    that are horizontally next to eachother.
    """
    xs, ys = coords.T
    width = np.max(xs) + 2 # The +2 prevents wraps from being adjacencies.
    ids = ys*width + xs
    ids.sort()
    edges = consecutive(ids)
    edge_xs = edges % width
    edge_ys = edges // width
    return (np.column_stack([edge_xs, edge_ys]),
            np.column_stack([edge_xs+1, edge_ys]))

def vertical_edges(coords):
    """Same idea as horizontal_edges, but for vertical adjacency. """
    def flip(a):
        return np.column_stack([a[:, 1], a[:, 0]])
    sources, targets = horizontal_edges(flip(coords))
    return (flip(sources), flip(targets))

def connected_components(coords):
    """
    Given a list of pixel coordinates, separate then into connected chunks.
    (We use horizontal and vertical adjacency to determine connectedness.)
    """
    if len(coords) == 0: return
    coords = coords.astype(int)
    num_points = len(coords)

    xs, ys = coords.T
    width = np.max(xs) + 1
    height = np.max(ys) + 1

    # Note: the ID of a pixel (x,y) is w*y+x.  The index of that pixels is where it occurs in `coords`.
    id_to_index = np.zeros(width*height)
    id_to_index[ys*width+xs] = np.arange(len(xs))
    def as_index(coordinates):
        xs, ys = coordinates.T
        return id_to_index[ys*width + xs]

    h_src, h_trg = horizontal_edges(coords)
    v_src, v_trg = vertical_edges(coords)


    src = np.concatenate([h_src, v_src])
    trg = np.concatenate([h_trg, v_trg])

    adjacency = scipy.sparse.coo_matrix(
            (np.ones(src.shape[0]), (as_index(src), as_index(trg))),
            shape=(num_points, num_points))

    n_components, labels = scipy.sparse.csgraph.connected_components(
        adjacency, directed=False, return_labels=True)

    label_sort = np.argsort(labels)
    label_sorted = labels[label_sort]
    label_starts = np.nonzero(np.concatenate([[1], np.ediff1d(label_sorted)]))[0]
    label_ends = np.concatenate([label_starts[1:], [num_points]])
    for start, end in zip(label_starts, label_ends):
        yield coords[label_sort[start:end]]


def left_edge(arr):
    """Given an array of integers, find all the elements that
    aren't one heigher than their left neighbor."""
    left_edge_idx = np.concatenate([[True], arr[:-1] + 1 != arr[1:]])
    return arr[left_edge_idx]

def right_edge(arr):
    """Given an array of integers, find all the elements that
    aren't one lower than their right neighbor."""
    right_edge_idx = np.concatenate([arr[:-1] + 1 != arr[1:], [True]])
    return arr[right_edge_idx]

def horizontal_border(coords):
    """Find all pixel locations that are to the left or right or pixels in `coords`
    but aren't present in coords."""
    xs, ys = coords.T
    width = np.max(xs) + 2 # The +2 prevents wraps from being adjacencies.
    ids = ys*width + xs
    ids.sort()

    def as_coords(ids):
        return np.column_stack([ids % width, ids // width])

    h_shift = np.array([[1, 0]])
    left_border = as_coords(left_edge(ids)) - h_shift
    right_border = as_coords(right_edge(ids)) + h_shift

    return np.concatenate([left_border, right_border])

def vertical_border(coords):
    """Same idea as horizontal_border, but verical."""
    def flip(a):
        return np.column_stack([a[:, 1], a[:, 0]])
    return flip(horizontal_border(flip(coords)))


def border(coords):
    """Returns pixels that are either in the horizontal or vertical border."""
    coords = coords.astype(int)
    border = np.concatenate([horizontal_border(coords), vertical_border(coords)])
    return np.unique(border, axis=0)


def inscribe_circle(coords, border):
    """Fit a big circle inside the region designated by coords, trying to make it as big as possible.
    `border` should be the border or the region as calculated by the `border` function.
    """
    coord_set = set((int(x), int(y)) for x,y in coords)
    def objective(v):
        if (int(v[0]), int(v[1])) not in coord_set:
            return 0
            # Objective function is zero outside the region.
        distances = np.linalg.norm(border - [v], axis=1)
        return -np.min(distances)
    guess = coords[np.random.randint(coords.shape[0])]
    result = scipy.optimize.minimize(objective, guess, method='Nelder-Mead')
    return (result.x, -result.fun)


def circle_coords(center, r, innerR=None):
    """Given a center, a radius, and an optional inner radius, return the list of
    pixels inside the circle (or ring if an inner radius was specified).
    """
    if innerR is None: innerR = 0
    ri = np.ceil(r)
    ys, xs = np.reshape(np.mgrid[-ri:(ri+2), -ri:(ri+2)], (2, -1)).astype(int)
    coords = np.column_stack([xs, ys])
    coords += np.array([np.floor(center)], dtype=int)
    radii = np.linalg.norm(coords - [center], axis=1) 
    return coords[np.logical_and(radii < r, radii > innerR)]


def paint(destination, coords, color):
    """Given a `destination` matrix representing an image, and a matrix `coords`
    of pixel coordinates, set all of the coresponding pixels to `color`.
    """
    height, width, chan = destination.shape
    xs, ys = coords.T
    xs = np.clip(xs, 0, width-1)
    ys = np.clip(ys, 0, height-1)
    destination[ys, xs] = color


class Region:
    def __init__(self, coords, color=None):
        """Make a region.  Unspecified color defaults to random."""
        if color is None:
            color = np.random.randint(255, size=3, dtype='uint8')
        self.coords = coords
        self.border = border(self.coords)
        self.color = color

    def size(self):
        """How many pixels are in the region?"""
        return self.coords.shape[0]

    def split_circle(self):
        """Find a circle in the region, and use the circle to break the region apart into new subregions.
        Regurns the circle, and the remaining pieces.
        """
        center, radius = inscribe_circle(self.coords, self.border)
        radius += 1
        outside_of_circle = self.coords[np.linalg.norm(self.coords - [center], axis=1) > radius]
        components = connected_components(outside_of_circle)
        pieces = [Region(c, self.color) for c in components]
        return ((center, radius, self.color), pieces)



image = Image.open(sys.argv[1])

labeled_greys = image_to_array(image)
non_ink_pixels = labeled_greys[labeled_greys[:,2] > .8, :2]

regions = [Region(c) for c in connected_components(non_ink_pixels)]
circles = []

while len(regions) > 0:
    region = regions.pop()
    if region.size() < 10: continue

    circle, newRegions = region.split_circle()
    circles.append(circle)
    regions += newRegions

output = np.copy(np.asarray(image.convert("RGB"), dtype='uint8'))
for c in circles:
    center, radius, color = c
    paint(output, circle_coords(center, radius, radius*.9), color)

output_image = Image.fromarray(output)
output_image.show()


