## ----------------- FUNCTION TO MANUALLY TEST VESICLE DETECTION PARAMETERS ------------------------------------------------------------------------------

# This function is to manually test the vesicle detection parameters, to check they are suitable for the samples and images.
# For the specified test sample, it will load the images then create a figure showing where it detected vesicles and where it will measure fluorescence.

# Run this function with:
# fig = func_test_detection_vesicle(test_condition, test_sample, settings)

import matplotlib.pyplot as plt
import numpy as np

from func_detect_measure_vesicle import func_check_measure_type, func_membrane_mask, inner_circle_radius
from func_detect_vesicles import func_detect_vesicles
from func_load_images_vesicle import func_load_images_vesicle
from matlab_helpers import imshow, viscircles


def func_test_detection_vesicle(test_condition, test_sample, settings):
    #   Input parameters:
    #       - test_condition = Name of the condition folder containing the test sample eg. 'noNTPs_t0'.
    #       - test_sample = Name of specific sample to test (file name without '_DYE.png') eg. 'A. noNTPs_t0'.
    #       - settings = the measurement & detection settings defined in INPUT PARAMETERS.
    #   Output:
    #       - Left: image of vesicle marker dye, with the vesicles it detected (cyan).
    #       - Right: image of fluorescence to measure, with where it will measure fluorescence:
    #               - 'WHOLE': inside the cyan circles.
    #               - membrane types: between the red rings (the membrane region), green circle = inner region.

    measure_fluorescence_type = settings['measure_fluorescence_type']
    func_check_measure_type(measure_fluorescence_type)

    ############# LOAD IMAGES & DETECT VESICLES FOR SPECIFIED SAMPLE ############
    grey_DYE_image, grey_MES_image = func_load_images_vesicle(test_condition, test_sample)[:2]
    centers, radii = func_detect_vesicles(test_condition, test_sample, grey_DYE_image, settings)

    ############### PLOT FIGURE OF DETECTED VESICLES #############
    fig, (ax_DYE, ax_MES) = plt.subplots(1, 2, figsize=(11, 5))

    imshow(ax_DYE, grey_DYE_image)
    viscircles(ax_DYE, centers, radii, 'c', linewidth=1)
    ax_DYE.set_title('%d vesicles detected in marker (DYE) image' % len(radii), fontsize=16)

    # shows the location of the vesicles on the fluorescence image
    imshow(ax_MES, grey_MES_image)
    ax_MES.set_title('Regions measured in fluorescence (MES) image', fontsize=16)

    if measure_fluorescence_type == 'WHOLE':
        viscircles(ax_MES, centers, radii, 'c', linestyle='--')

    else:
        # Overlay the outline of the membrane region searched around each vesicle (red)
        membrane_edges = func_membrane_mask(centers, radii, grey_DYE_image.shape, settings['inner_boundary'], settings['outer_boundary'], outline=True)
        edge_overlay = np.zeros(membrane_edges.shape + (4,))
        edge_overlay[..., 0] = 1  # red
        edge_overlay[..., 3] = membrane_edges  # transparent except the outlines
        ax_MES.imshow(edge_overlay)
        viscircles(ax_MES, centers, inner_circle_radius, 'g')  # green ring for the inner circle

    fig.tight_layout()
    return fig
