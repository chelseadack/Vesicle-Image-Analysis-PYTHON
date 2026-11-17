## ----------------- FUNCTION TO CHECK LINE PROFILE PERFORMANCE ----------------------------------------------------------------

# This function plots the line profile data for an individual vesicle in a specified sample.
# It uses the vesicle detected by func_detect_measure_vesicle (no detection is repeated) and the same line profile code,
# so the values shown are exactly the values measured.

# Run this function with:
# fig = func_line_profile_vesicle(test_condition, test_sample, k, vesicleData, settings)

import matplotlib.pyplot as plt
import numpy as np

from func_detect_measure_vesicle import func_check_up_to_date, func_distance_from, func_membrane_peaks, inner_circle_radius, measure_labels
from func_load_images_vesicle import func_load_images_vesicle
from matlab_helpers import imshow_scaled, matlab_round, trimmean, viscircles


def func_line_profile_vesicle(test_condition, test_sample, k, vesicleData, settings):
    #   Input parameters:
    #       - test_condition = Name of the condition folder containing the test sample eg. 'noNTPs_t0'.
    #       - test_sample = specified sample to test (file name without '_DYE.png') eg. 'A. noNTPs_t0'.
    #       - k = number of the vesicle to plot (the 'Vesicle' column of vesicleData, also shown by func_plot_MES_for_sample_vesicle).
    #       - vesicleData = table returned by func_detect_measure_vesicle.
    #       - settings = the measurement & detection settings defined in INPUT PARAMETERS (must be the ones vesicleData was measured with).
    #   Output:
    #       - Left: the image with the chosen vesicle's 4 line profiles, in the same colours as the graph,
    #               the membrane region searched (red dashed circles) and the inner region (green circle).
    #       - Right: raw line profile data:
    #               - Red dashed vertical lines indicate the membrane region searched for peak fluorescence.
    #               - Red horizontal line is the average (of the two middle values) peak membrane fluorescence.
    #               - Green dashed vertical line indicates the inner vesicle region used.
    #               - Green horizontal line is the average inner vesicle fluorescence value.

    func_check_up_to_date(vesicleData, settings)

    ########## FIND THE VESICLE & LOAD ITS IMAGE ###########
    vesicles = vesicleData[(vesicleData['Condition'] == test_condition) & (vesicleData['Image'] == test_sample)]
    vesicle = vesicles[vesicles['Vesicle'] == k]
    if vesicle.empty:
        raise ValueError("There is no vesicle %d in '%s' (condition '%s'). Choose one of the vesicle numbers shown by the section above "
                         "(numbers can be skipped where a vesicle was removed, eg. ratio above max_ratio)." % (k, test_sample, test_condition))
    center = vesicle[['X', 'Y']].to_numpy()[0]
    radius = vesicle['Radius'].iloc[0]

    grey_MES_image = func_load_images_vesicle(test_condition, test_sample)[1]

    ############## LINE PROFILES & PEAK MEMBRANE FLUORESCENCE ##############
    # profiles all start at the vesicle centre (position 0 on the graph); search_min & search_max are MATLAB 1-based positions
    peakValues, allProfiles, search_min, search_max = func_membrane_peaks(grey_MES_image, center, radius, settings['inner_boundary'], settings['outer_boundary'])
    meanMembraneIntensity = trimmean(peakValues[~np.isnan(peakValues)], 50)  # removes top & bottom, averages the 2 middle values

    ############## MEASURE INNER VESICLE AVERAGE FLUORESCENCE #############
    innerMean = np.mean(grey_MES_image[func_distance_from(center, grey_MES_image.shape) <= inner_circle_radius].astype(np.float64))

    fig, (ax_image, ax_graph) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ['r', 'g', 'b', 'k']

    ############## PLOT VESICLE IDENTITY ONTO IMAGE (LEFT) ############
    imshow_scaled(ax_image, grey_MES_image)

    # the 4 line profiles, drawn from the centre outwards (plotting positions are 0-based)
    cx = matlab_round(center[0]) - 1
    cy = matlab_round(center[1]) - 1
    for d, (dx, dy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):  # right, left, down, up
        length = len(allProfiles[d]) - 1
        ax_image.plot([cx, cx + dx * length], [cy, cy + dy * length], color=colors[d], linewidth=1)

    # membrane bounds (the distances from the centre searched for the peak) & inner region
    viscircles(ax_image, [center], search_min - 1, 'r', linestyle='--', linewidth=1)
    viscircles(ax_image, [center], search_max - 1, 'r', linestyle='--', linewidth=1)
    viscircles(ax_image, [center], inner_circle_radius, 'g', linewidth=1)

    ax_image.set_title('Vesicle %d in %s' % (k, test_sample))

    ############## PLOT LINE PROFILES (RIGHT) ##############
    for d, label in enumerate(['Right', 'Left', 'Down', 'Up']):
        ax_graph.plot(np.arange(len(allProfiles[d])), allProfiles[d], color=colors[d], linewidth=1.5, label=label)

    # horizontal red line for average peak membrane intensity
    ax_graph.axhline(meanMembraneIntensity, color='r', linestyle='--', linewidth=2)
    # horizontal green line for average inner intensity
    ax_graph.axhline(innerMean, color=(0, 0.5, 0), linestyle='--', linewidth=2)

    # vertical red dashed lines for membrane region searched (in pixels from the centre)
    ax_graph.axvline(search_min - 1, color='r', linestyle='--', linewidth=0.5)
    ax_graph.axvline(search_max - 1, color='r', linestyle='--', linewidth=0.5)
    # vertical green dashed line for inner region
    ax_graph.axvline(inner_circle_radius, color=(0, 0.5, 0), linestyle='--', linewidth=0.5)

    ax_graph.legend()
    ax_graph.set_xlabel('Distance from vesicle centre (pixels)')
    ax_graph.set_ylabel('Intensity')
    ax_graph.set_title('%s: %.4g' % (measure_labels[settings['measure_fluorescence_type']], vesicle['MES'].iloc[0]))

    fig.tight_layout()
    return fig
