from PIL import Image
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# import pandas as pd
from scipy.spatial.distance import cdist

def pixel_accuracy(groundarray, imagearray):
    """find error rate of accurately segmented pixels in imagearray given the ground truth groundarray

    Arguments:
        groundarray -- ground truth image, as a numpy array
        imagearray -- image to find the error of, as a numpy array

    Returns:
        segmentation pixel accuracy (false negatives) - how many pixels were correct out of total number of pixels classified as a cell
    """
    # masks displayed by distinct non-zero integers
    max = np.max(groundarray) + 1
    # total number of mask pixels
    total_mask_pixels = groundarray[groundarray > 0].size
    tot_correct = 0
    previous_values = []
    for i in range(1, max):
        # array indices of cell masks in ground truth array
        coords = np.where(groundarray == i)
        # equivalent region in image being tested
        test_area = imagearray[coords]
        equivalent_value = stats.mode(test_area)[0][0]
        if equivalent_value not in previous_values:
            previous_values.append(equivalent_value)
        # get the count of the mode number in the region 
        # (since masks don't necessarily have the same number)
            test_area = test_area[test_area > 0]
            if len(test_area) > 0:
                tot_correct += stats.mode(test_area)[1][0]
    return tot_correct/total_mask_pixels

def id_accuracy(groundarray, imagearray, image_info=False):
    """return identification accuracy of image compared to ground truth image

    Arguments:
        groundarray -- ground truth image, as a numpy array
        imagearray -- image to find the error of, as a numpy array

    Returns:
        out[0] - num_identified/num_cells < 1
        out[1] - num_identified/num_cells = 1
        out[2] - num_cells/num_identified < 1
    """
    out = np.zeros(3)
    num_cells = np.max(groundarray)
    if image_info:
        num_identified = len(imagearray[0])
    else:
        num_identified = np.max(imagearray)
    if num_identified < num_cells:
        out[0] = num_identified/num_cells
    elif num_identified == num_cells:
        out[1] = 1
    elif num_identified > num_cells:
        out[2] = num_cells/num_identified
    return out

def find_cell_info(array):
    num_cells = np.max(array)
    cell_areas = np.zeros(num_cells)
    cell_centroids = np.zeros([num_cells,2])
    for i in range(1, num_cells+1):
        coords = np.where(array == i)
        cell_areas[i-1] = len(array[coords])
        y_bar = np.mean(coords[0])
        x_bar = np.mean(coords[1])
        cell_centroids[i-1] = [y_bar, x_bar]
    return cell_centroids, cell_areas

def find_nearest_centres(ground_centroids, cell_centroids):
    # create array of distances from each cell centroid to each ground centroid
    distances = cdist(cell_centroids, ground_centroids)
    # create vector, same length as cell_centroids, containing the nearest ground centroid to the equivalent 
    nearest_ground = ground_centroids[np.argmin(distances,axis=1)]
    dist_vector = np.min(distances, axis=1)
    return nearest_ground, dist_vector #, distances

def compare_cells(ground_centroids, ground_areas, array, min_dist, lb, ub, images_info):
    num_correct = 0
    if images_info:
        cell_centroids, cell_areas = array[1:].T, array[0]
    else:
        cell_centroids, cell_areas = find_cell_info(array)
    nearest_ground, distances = find_nearest_centres(ground_centroids, cell_centroids)
    unique_nearest_ground, nearest_ground_counts = np.unique(nearest_ground, return_counts=True,axis=0)
    for i, distance in enumerate(distances):
        if distance <= min_dist:
            near_ground = nearest_ground[i]
            ground_area = ground_areas[np.sum(ground_centroids == near_ground, axis=1) == 2]
            if lb < ground_area/cell_areas[i] < ub:
                ind = np.argmax(np.sum(unique_nearest_ground == near_ground, axis=1) == 2)
                if nearest_ground_counts[ind] > 0:
                    nearest_ground_counts[ind] = 0
                    num_correct += 1
    return num_correct/len(ground_areas)

def segmentation_accuracy(groundarray, imagearrays, images_info=False, min_dist=5, lb=0.7, ub=1.5):
    # what percent of masks identified are reasonable representations of true cells
    ground_centroids, ground_areas = find_cell_info(groundarray)
    segmentation_accuracy = []
    for imagearray in imagearrays:
        accuracy = compare_cells(ground_centroids, ground_areas, imagearray, min_dist, lb, ub, images_info)
        segmentation_accuracy.append(accuracy)
    return segmentation_accuracy

def andoveror(a,b):
    acs = np.array([a[0],a[1]]).T
    bcs = np.array([b[0],b[1]]).T
    tupa = [tuple(item) for item in acs]
    tupb = [tuple(item) for item in bcs]
    return [len(set(tupa).intersection(tupb)), len(set(tupa).union(tupb))]

def IoU(groundarray, imagearray):
    """find error rate of accurately segmented pixels in imagearray given the ground truth groundarray

    Arguments:
        groundarray -- ground truth image, as a numpy array
        imagearray -- image to find the error of, as a numpy array

    Returns:
        segmentation pixel accuracy (false negatives) - how many pixels were correct out of total number of pixels classified as a cell
    """
    # masks displayed by distinct non-zero integers
    max = int(np.max(groundarray) + 1)
    # total number of mask pixels
    previous_values = []
    iou_vals = []
    sizes = []
    for i in range(1, max):
        if np.isin(i,groundarray):
            # array indices of cell masks in ground truth array
            coords = np.where(groundarray == i)
            sizes.append(len(coords[0]))
            # equivalent region in image being tested
            test_area = imagearray[coords]
            equivalent_value = stats.mode(test_area)[0][0]
            previous_values.append(equivalent_value)
            image_coords = np.where((imagearray == equivalent_value))
            if len(coords) > 0:
                iou_vals.append(andoveror(coords,image_coords))
    unq, counts = np.unique(previous_values,return_counts=True)
    iou_vals = np.array(iou_vals)
    pv = np.array(previous_values)
    out = np.array(iou_vals)
    for i, val in enumerate(unq):
        if counts[i] > 1:
            dodgy_vals = iou_vals[np.where(pv==val)]
            perc = dodgy_vals[:,0]/dodgy_vals[:,1]
            fak = dodgy_vals[np.argsort(perc)][:-1]
            for fake in fak:
                out[np.all(np.equal(out,fake),axis=-1)] = [0,sizes[i]]
    return out