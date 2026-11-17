## ----------------- FUNCTION TO CHECK MEASURED FLUORESCENCE FOR ONE SAMPLE ----------------------------------------------------------------

# Overlays a coloured spot onto each vesicle in the specified image, indicating its measured fluorescence.
# Uses the vesicles already detected & measured by func_detect_measure_vesicle, so no detection is repeated.
# Each vesicle is labelled with its number, for use with func_line_profile_vesicle.

# Run this function with:
# fig = func_plot_MES_for_sample_vesicle(test_condition, test_sample, vesicleData, settings, maxyvalue, minyvalue)

import matplotlib.pyplot as plt
import numpy as np

from func_detect_measure_vesicle import func_check_up_to_date, func_membrane_mask, measure_labels
from func_load_images_vesicle import func_load_images_vesicle
from matlab_helpers import imshow_scaled, viscircles


def func_plot_MES_for_sample_vesicle(test_condition, test_sample, vesicleData, settings, maxyvalue, minyvalue):
    #   Input parameters:
    #       - test_condition = name of the condition folder containing the sample eg. 'noNTPs_t0'.
    #       - test_sample = name of the sample (file name without '_DYE.png') eg. 'A. noNTPs_t0'.
    #       - vesicleData = table returned by func_detect_measure_vesicle.
    #       - settings = the measurement & detection settings defined in INPUT PARAMETERS (must be the ones vesicleData was measured with).
    #       - maxyvalue = measured fluorescence value at the top of the colour scale.
    #       - minyvalue = measured fluorescence value at the bottom of the colour scale.
    #   Output:
    #       - The fluorescence (MES) image with a coloured spot on each vesicle indicating its measured fluorescence,
    #         over the membrane region measured (red) or the vesicle outline (blue) for 'WHOLE'.

    func_check_up_to_date(vesicleData, settings)

    # Find the vesicles in this sample
    vesicles = vesicleData[(vesicleData['Condition'] == test_condition) & (vesicleData['Image'] == test_sample)]
    if vesicles.empty:
        raise ValueError("No vesicles found for '%s' in condition '%s'. Check the names match the image file & folder names." % (test_sample, test_condition))
    centers = vesicles[['X', 'Y']].to_numpy()
    radii = vesicles['Radius'].to_numpy()

    grey_MES_image = func_load_images_vesicle(test_condition, test_sample)[1]

    ############# OVERLAY COLOUR-CODED VALUES ONTO IMAGE #############
    fig, ax = plt.subplots(figsize=(10, 8))
    imshow_scaled(ax, grey_MES_image)

    # If measuring membrane, will overlay membrane region used onto image (semi-transparent red)
    # If measuring whole vesicle, will overlay vesicle boundary onto image
    if settings['measure_fluorescence_type'] == 'WHOLE':
        viscircles(ax, centers, radii, 'b', linewidth=1)
    else:
        membrane_mask = func_membrane_mask(centers, radii, grey_MES_image.shape, settings['inner_boundary'], settings['outer_boundary'])
        edge_overlay = np.zeros(membrane_mask.shape + (4,))
        edge_overlay[..., 0] = 1  # red
        edge_overlay[..., 3] = 0.25 * membrane_mask
        ax.imshow(edge_overlay)

    # plot coloured spots onto image (values outside minyvalue-maxyvalue get the end colours)
    spots = ax.scatter(vesicles['X'] - 1, vesicles['Y'] - 1, s=10**2, c=vesicles['MES'], cmap='turbo', vmin=minyvalue, vmax=maxyvalue, edgecolors='k')
    for x, y, number in zip(vesicles['X'], vesicles['Y'], vesicles['Vesicle']):
        ax.text(x - 1 + 15, y - 1, str(number), color='w', fontsize=9)  # vesicle number
    fig.colorbar(spots, ax=ax, label=measure_labels[settings['measure_fluorescence_type']])

    ax.set_title('Measured fluorescence of each vesicle: ' + test_sample)

    return fig
