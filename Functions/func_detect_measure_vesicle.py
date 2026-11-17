## ----------------- FUNCTION TO DETECT VESICLES & QUANTIFY FLUORESCENCE ----------------------------------------------------------------

# This function detects the vesicles in each image then calculates their average fluorescence.
# It can calculate whole vesicle fluorescence or just membrane fluorescence.
# Average fluorescence is normalised by subtracting the background fluorescence (or dividing by the inner vesicle fluorescence,
# in which case no background images are needed).
# The result is one table with one row per vesicle, for every image in every condition.
# The check plots (func_plot_MES_for_sample_vesicle & func_line_profile_vesicle) use this table, so vesicles are only detected once.

# Run this function with:
# vesicleData, imageList = func_detect_measure_vesicle(condition_folders, settings)

import numpy as np
import pandas as pd

from func_detect_vesicles import func_detect_vesicles
from func_load_images_vesicle import func_find_images_vesicle, func_load_images_vesicle
from matlab_helpers import bwperim, matlab_round, trimmean

inner_circle_radius = 10  # radius in pixels of the inner circle used for 'MEMBRANE-INNER'

# the measurement types, and the axis label used for each on the plots
measure_labels = {
    'WHOLE': 'Average Vesicle Fluorescence (A.U.)',
    'MEMBRANE-BACKGROUND': 'Average Membrane Fluorescence (A.U.)',
    'MEMBRANE-INNER': 'Membrane/Lumen Fluorescence Ratio',
}


def func_check_measure_type(measure_fluorescence_type):
    # Gives a clear error if the measurement type is misspelt
    if measure_fluorescence_type not in measure_labels:
        raise ValueError("measure_fluorescence_type is '%s', but must be one of: %s" % (measure_fluorescence_type, ', '.join(measure_labels)))


def func_check_up_to_date(vesicleData, settings):
    # Gives a clear error if vesicleData was measured with different settings to the current ones,
    # eg. if the detection method was changed without re-running the DETECT & MEASURE section
    changed = ['%s: %r -> %r' % (name, vesicleData.attrs['settings'].get(name), value)
               for name, value in settings.items() if vesicleData.attrs['settings'].get(name) != value]
    if changed:
        raise ValueError('vesicleData was measured with different settings to the current ones (%s).\n'
                         'Re-run the DETECT & MEASURE section first.' % '; '.join(changed))


def func_distance_from(center, image_shape):
    # Distance of every pixel from the vesicle centre (MATLAB 1-based positions)
    rows, cols = image_shape
    x, y = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    return np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)


def func_membrane_mask(centers, radii, image_shape, inner_boundary, outer_boundary, outline=False):
    # Returns a mask of the membrane region searched around every vesicle (between inner & outer boundary x radius),
    # or just the outline of each vesicle's membrane region if outline=True
    mask = np.zeros(image_shape, dtype=bool)
    for center, radius in zip(centers, radii):
        dist = func_distance_from(center, image_shape)
        membrane_ring = (dist >= inner_boundary * radius) & (dist <= outer_boundary * radius)
        mask |= bwperim(membrane_ring) if outline else membrane_ring
    return mask


def func_membrane_peaks(img, center, radius, inner_boundary, outer_boundary):
    # Takes 4 line profiles (right, left, down, up) from the vesicle centre, out to 1.3 x the radius,
    # and returns the peak fluorescence of each within the membrane region (NaN if the profile is too short, eg. at the image edge).
    # Also returns the line profiles and where the peak was searched for, for func_line_profile_vesicle.
    rows, cols = img.shape
    cx = int(matlab_round(center[0]))  # MATLAB 1-based positions
    cy = int(matlab_round(center[1]))

    x_right = np.arange(cx, matlab_round(cx + radius * 1.3) + 1)
    x_left = np.arange(matlab_round(cx - radius * 1.3), cx + 1)
    y_down = np.arange(cy, matlab_round(cy + radius * 1.3) + 1)
    y_up = np.arange(matlab_round(cy - radius * 1.3), cy + 1)
    x_right = x_right[(x_right >= 1) & (x_right <= cols)].astype(int)
    x_left = x_left[(x_left >= 1) & (x_left <= cols)].astype(int)
    y_down = y_down[(y_down >= 1) & (y_down <= rows)].astype(int)
    y_up = y_up[(y_up >= 1) & (y_up <= rows)].astype(int)

    # left & up are flipped so every profile starts at the centre
    # (the original MATLAB code didn't flip 'up', so it searched the wrong end of that profile)
    lineProfiles = [
        img[cy - 1, x_right - 1].astype(np.float64),
        img[cy - 1, x_left - 1][::-1].astype(np.float64),
        img[y_down - 1, cx - 1].astype(np.float64),
        img[y_up - 1, cx - 1][::-1].astype(np.float64),
    ]

    # search membrane region for peak (search_min & search_max are MATLAB 1-based positions along the profile,
    # profile position i is i-1 pixels from the centre)
    search_min = int(matlab_round(radius * inner_boundary))
    search_max = int(matlab_round(radius * outer_boundary))

    peakVals = np.full(4, np.nan)
    for d, lineProfile in enumerate(lineProfiles):
        if search_min >= len(lineProfile):
            continue  # stays NaN

        peakVals[d] = lineProfile[search_min - 1:min(search_max, len(lineProfile))].max()

    return peakVals, lineProfiles, search_min, search_max


def func_measure_vesicle(img, center, radius, Background, settings):
    # Returns the measured fluorescence (MES) of one vesicle (infinite or NaN if the inner circle is 0, see func_detect_measure_vesicle)
    measure_fluorescence_type = settings['measure_fluorescence_type']
    dist = func_distance_from(center, img.shape)

    ############## MEASURE WHOLE VESICLE #############
    if measure_fluorescence_type == 'WHOLE':
        return np.mean(img[dist <= radius]) - Background

    ############## MEASURE MEMBRANE LINE PROFILES & AVERAGE PEAK #############
    peakVals = func_membrane_peaks(img, center, radius, settings['inner_boundary'], settings['outer_boundary'])[0]
    memMean = trimmean(peakVals[~np.isnan(peakVals)], 50)  # removes top & bottom (NaNs at the image edge removed first), averages the 2 middle values

    if measure_fluorescence_type == 'MEMBRANE-BACKGROUND':
        return memMean - Background  # subtract background fluorescence

    # 'MEMBRANE-INNER'
    innerMean = np.mean(img[dist <= inner_circle_radius])  # average of the inner circle
    with np.errstate(divide='ignore', invalid='ignore'):
        return memMean / innerMean  # ratio membrane : inner


def func_detect_measure_vesicle(condition_folders, settings):
    #   Input parameters:
    #       - condition_folders = list of condition folders to process, in the order they will be plotted.
    #       - settings = the measurement & detection settings defined in INPUT PARAMETERS.
    #   Output:
    #       - vesicleData = table with one row per vesicle: Condition, Image, Repeat, Vesicle, X, Y, Radius, MES
    #               (Vesicle = detection number, shown by func_plot_MES_for_sample_vesicle & used by func_line_profile_vesicle;
    #                X & Y are MATLAB 1-based pixel positions; removed vesicles are left out, so some numbers can be skipped)
    #       - imageList = table of every image processed (including any where no vesicles were found)
    #   For 'MEMBRANE-INNER', vesicles are removed if the inner circle is completely dark (ratio divided by 0),
    #   or the ratio is above settings['max_ratio'] (a near-black inner circle gives a meaningless, extreme ratio).

    func_check_measure_type(settings['measure_fluorescence_type'])
    needs_background = settings['measure_fluorescence_type'] != 'MEMBRANE-INNER'

    vesicleRows = []
    imageList = []
    n_unmeasurable = 0  # inner circle of 0
    n_above_max = 0  # ratio above max_ratio
    for condition_folder in condition_folders:
        print('started detecting vesicles in %s' % condition_folder)
        image_names = func_find_images_vesicle(condition_folder, needs_background)

        for image_name in image_names:
            grey_DYE_image, img, grey_background_image = func_load_images_vesicle(condition_folder, image_name)
            centers, radii = func_detect_vesicles(condition_folder, image_name, grey_DYE_image, settings)
            Background = np.mean(grey_background_image) if needs_background else None

            for k, (center, radius) in enumerate(zip(centers, radii), start=1):
                MES = func_measure_vesicle(img, center, radius, Background, settings)
                if not np.isfinite(MES):
                    n_unmeasurable += 1
                elif not needs_background and MES > settings['max_ratio']:
                    n_above_max += 1
                else:
                    vesicleRows.append((condition_folder, image_name, k, center[0], center[1], radius, MES))

            imageList.append((condition_folder, image_name))

        print('finished %d images from %s' % (len(image_names), condition_folder))

    if not needs_background:
        print('removed %d vesicles with ratio above max_ratio (%g) and %d with a completely dark inner circle'
              % (n_above_max, settings['max_ratio'], n_unmeasurable))

    vesicleData = pd.DataFrame(vesicleRows, columns=['Condition', 'Image', 'Vesicle', 'X', 'Y', 'Radius', 'MES'])
    imageList = pd.DataFrame(imageList, columns=['Condition', 'Image'])

    # the repeat is the start of each image name, before the first '.' (eg. 'A' in 'A. noNTPs_t0')
    for table in [vesicleData, imageList]:
        table.insert(2, 'Repeat', table['Image'].str.split('.').str[0])
        # keep the conditions in the order given, for tables and plots
        table['Condition'] = pd.Categorical(table['Condition'], categories=condition_folders)

    vesicleData.attrs['settings'] = dict(settings)  # remembers the settings used, so the check plots can confirm they are up to date

    return vesicleData, imageList
