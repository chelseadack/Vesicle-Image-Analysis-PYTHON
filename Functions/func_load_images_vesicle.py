## ----------------- FUNCTIONS TO FIND AND LOAD THE IMAGES FOR ONE CONDITION FOLDER ------------------------------------------------------------------------------------

# Images are loaded one sample at a time, so only one set of images is in memory at once.

# Data must be in this format:
#	- One folder per condition, containing all of the repeats for that condition.
#	- Colour or greyscale images must be saved as '.png'.
#	- Each unique image must have these images with these naming patterns:
#		Eg. 'A. noNTPs_t0_DYE.png'        = Image of vesicle marker dye, used to detect the vesicles.
#		    'A. noNTPs_t0_MES.png'        = Image of fluorescence to measure.
#		    'A. noNTPs_t0_Background.png' = Image of representative negative background for normalisation.
#		                                    Not needed for 'MEMBRANE-INNER' (normalised to the inside of each vesicle instead).

# Run these functions with:
# image_names = func_find_images_vesicle(condition_folder, needs_background)
# grey_DYE_image, grey_MES_image, grey_background_image = func_load_images_vesicle(condition_folder, image_name)

import os

from matlab_helpers import im2gray, imread


def func_find_images_vesicle(condition_folder, needs_background=True):
    # Returns the names of all samples in the folder (file names without '_DYE.png'),
    # checking each has a MES image, and a Background image if needs_background
    files = sorted(name for name in os.listdir(condition_folder) if name.endswith('_DYE.png'))  # Find all marker dye images in the folder
    image_names = [name.replace('_DYE.png', '') for name in files]

    for image_name in image_names:
        for image_type in ['_MES.png'] + (['_Background.png'] if needs_background else []):
            if not os.path.isfile(os.path.join(condition_folder, image_name + image_type)):
                raise FileNotFoundError('No matching %s image found for %s' % (image_type, os.path.join(condition_folder, image_name + '_DYE.png')))

    return image_names


def func_load_images_vesicle(condition_folder, image_name):
    # Loads the marker dye, fluorescence & background images for one sample and converts them to greyscale
    # (grey_background_image is None if there is no background image)
    grey_DYE_image = im2gray(imread(os.path.join(condition_folder, image_name + '_DYE.png')))
    grey_MES_image = im2gray(imread(os.path.join(condition_folder, image_name + '_MES.png')))
    background_file = os.path.join(condition_folder, image_name + '_Background.png')
    grey_background_image = im2gray(imread(background_file)) if os.path.isfile(background_file) else None

    return grey_DYE_image, grey_MES_image, grey_background_image
