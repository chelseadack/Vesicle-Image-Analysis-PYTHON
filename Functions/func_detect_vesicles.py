## ----------------- FUNCTION TO DETECT VESICLES IN A MARKER DYE IMAGE ----------------------------------------------------------------

# This function finds every vesicle in the marker dye image, using one of 2 detection methods:
#   - 'HOUGH' = circular Hough transform, the same as MATLAB's imfindcircles (the original method).
#   - 'CELLPOSE' = Cellpose deep learning segmentation. Each vesicle is then measured as the circle with the same area.

# Each image is only detected once: the result is remembered and reused if the same image is detected again with the same settings
# (eg. the test image in TEST PARAMETERS is not detected again in DETECT & MEASURE).
# Cellpose is slow, so its results are also saved in a 'Saved vesicle labels' folder (inside the working folder),
# so they are reused even after restarting. If an image is changed (even if the file name stays the same), it is detected again
# automatically and its old saved results are deleted.
# Delete the folder at any time to free up space.

# Run this function with:
# centers, radii = func_detect_vesicles(condition_folder, image_name, grey_DYE_image, settings)

import glob
import hashlib
import os
import re

import numpy as np
from skimage.measure import regionprops

from matlab_helpers import imfindcircles, matlab_round

detection_methods = ['HOUGH', 'CELLPOSE']
labels_folder = 'Saved vesicle labels'
_detections = {}  # vesicles already detected in this session
_cellpose_model = None  # loaded once, the first time Cellpose is used


def func_check_detection_method(detection_method):
    # Gives a clear error if the detection method is misspelt
    if detection_method not in detection_methods:
        raise ValueError("detection_method is '%s', but must be one of: %s" % (detection_method, ', '.join(detection_methods)))


def func_detect_vesicles(condition_folder, image_name, grey_DYE_image, settings):
    #   Input parameters:
    #       - condition_folder, image_name = which image this is, used to save the Cellpose results.
    #       - grey_DYE_image = Greyscale marker dye image used to detect the vesicles.
    #       - settings = the detection settings defined in INPUT PARAMETERS:
    #           - detection_method = 'HOUGH' or 'CELLPOSE'.
    #           - sensitivity_factor = How circular vesicles have to be for detection (0 = perfect circle, 1 = barely circular). Only used by 'HOUGH'.
    #           - pixel_size_micron = Size of one pixel in microns.
    #           - min_diameter_micron = Minimum vesicle diameter to detect in microns.
    #           - max_diameter_micron = Maximum vesicle diameter to detect in microns.
    #   Output:
    #       - centers = [x, y] of each vesicle centre, in MATLAB 1-based pixel positions.
    #       - radii = radius of each vesicle in pixels.

    detection_method = settings['detection_method']
    func_check_detection_method(detection_method)
    min_diameter = settings['min_diameter_micron'] / settings['pixel_size_micron']  # in pixels
    max_diameter = settings['max_diameter_micron'] / settings['pixel_size_micron']

    image_id = hashlib.md5(grey_DYE_image.tobytes()).hexdigest()[:8]  # changes if the image changes
    key = (condition_folder, image_name, image_id, detection_method, settings['sensitivity_factor'], min_diameter, max_diameter)
    if key in _detections:
        return _detections[key]

    if detection_method == 'HOUGH':
        # Converts the min and max diameters into radii in pixels,
        # then uses a circular Hough transform to find circle edges in the greyscale image
        radiusRange = [matlab_round(min_diameter / 2), matlab_round(max_diameter / 2)]
        centers, radii = imfindcircles(grey_DYE_image, radiusRange, settings['sensitivity_factor'])

    else:  # 'CELLPOSE': find every vesicle outline, then keep those within the diameter range
        labels = func_get_cellpose_labels(condition_folder, image_name, image_id, grey_DYE_image)
        vesicles = regionprops(labels)
        diameters = np.array([v.equivalent_diameter_area for v in vesicles])  # diameter of the circle with the same area
        keep = (diameters >= min_diameter) & (diameters <= max_diameter)

        centers = np.array([[v.centroid[1] + 1, v.centroid[0] + 1] for v in vesicles]).reshape(-1, 2)[keep]  # MATLAB 1-based positions
        radii = diameters[keep] / 2

    _detections[key] = centers, radii
    return centers, radii


def func_get_cellpose_labels(condition_folder, image_name, image_id, grey_DYE_image):
    # Returns the Cellpose label image (background 0, vesicle 1 = 1, vesicle 2 = 2, etc.), loading it if already saved
    global _cellpose_model

    labels_file = os.path.join(labels_folder, condition_folder, '%s_%s.npz' % (image_name, image_id))
    if os.path.isfile(labels_file):
        return np.load(labels_file)['labels']

    if _cellpose_model is None:
        from cellpose import models  # only imported when needed, so 'HOUGH' works without Cellpose installed
        _cellpose_model = models.CellposeModel(gpu=True)  # uses the GPU if there is one, otherwise the CPU
    print('   running Cellpose on %s (about 1 minute per image, only done once)' % image_name)
    labels = _cellpose_model.eval(grey_DYE_image)[0]

    # delete any old results for this image (from before the image was changed), then save
    old_pattern = re.compile(re.escape(image_name) + r'_[0-9a-f]{8}\.npz')
    for old_file in glob.glob(os.path.join(glob.escape(os.path.join(labels_folder, condition_folder)), '*.npz')):
        if old_pattern.fullmatch(os.path.basename(old_file)):
            os.remove(old_file)
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    np.savez_compressed(labels_file, labels=labels)
    return labels
