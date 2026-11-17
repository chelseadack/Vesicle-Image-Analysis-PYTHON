# Vesicle (GUV) fluorescence quantification (Python)

Automated detection of giant unilamellar vesicles (GUVs) in fluorescence microscope images, with per-vesicle measurement of whole-vesicle or membrane-localised fluorescence.

The pipeline finds every vesicle in a marker dye image, measures the fluorescence of each one in a second channel, and compares conditions using:

- per-vesicle violin plots (violin, data points, box, median notch and mean) for all repeats pooled, and for each biological repeat side by side
- tables of the number of vesicles and the mean & median fluorescence for each condition and each image
- for membrane/lumen ratios, the number and % of **localised** vesicles (ratio > 1)

It is a Python version of MATLAB GUV analysis scripts. The circle detection reproduces MATLAB's `imfindcircles`, so results match the original MATLAB method. Cellpose deep learning segmentation is available as an alternative detection method.

## Vesicle detection

Vesicles are detected in the marker dye (`_DYE`) image, using one of 2 methods (`detection_method`):

- **`'HOUGH'`:** a circular Hough transform. It re-implements MATLAB's `imfindcircles(image, radiusRange, 'Sensitivity', s)` (see `Functions/matlab_helpers.py`). `sensitivity_factor` sets how circular a vesicle has to be (0 = perfect circle, 1 = barely a circle).
- **`'CELLPOSE'`:** [Cellpose](https://github.com/MouseLand/cellpose) segmentation. Each vesicle is measured as the circle with the same area (centre = centroid). It finds vesicles with a dark lumen and a bright membrane, which the Hough transform can miss. It is slower (about 1 minute per image), so its results are saved in a `Saved vesicle labels` folder inside the working folder and reused. If an image changes, it is detected again automatically.

Both methods keep only vesicles between `min_diameter_micron` and `max_diameter_micron`.

Each image is only detected once. The test, the measurement and the check figures all use the same detected vesicles.

## Fluorescence measurement

The fluorescence (`_MES`) image is measured in one of 3 ways (`measure_fluorescence_type`):

| Type | Measurement | Normalisation |
|---|---|---|
| `'WHOLE'` | Mean fluorescence inside the vesicle | minus the mean of the `_Background` image |
| `'MEMBRANE-BACKGROUND'` | Peak membrane fluorescence (see below) | minus the mean of the `_Background` image |
| `'MEMBRANE-INNER'` | Peak membrane fluorescence (see below) | divided by the mean of a 10 pixel radius circle at the vesicle centre (membrane/lumen ratio). No background image is needed. |

**Peak membrane fluorescence:** 4 line profiles (right, left, down, up) are taken from the vesicle centre out to 1.3 × the radius. The peak of each is found between `inner_boundary` × radius and `outer_boundary` × radius, where the membrane should be. The highest and lowest of the 4 peaks are removed and the 2 middle values are averaged.

**Removed vesicles (`'MEMBRANE-INNER'` only):** vesicles with a completely dark centre (division by 0), or a ratio above `max_ratio` (default 30), are removed from all results. A near-black lumen gives a meaningless, extreme ratio (hundreds to thousands), while real, strongly localised vesicles in the test dataset reached ratios of about 26. The number removed is printed.

## Repository structure

```
PYTHON_CHELS_vesicle_quantification/
├── MAIN_SCRIPT_VESICLE.ipynb   # main analysis notebook, run it cell by cell
├── Functions/                  # functions called by the notebook
├── README.md
└── LICENSE
```

| File | Purpose |
|---|---|
| `MAIN_SCRIPT_VESICLE.ipynb` | Main analysis notebook. Run it cell by cell. |
| `Functions/func_load_images_vesicle.py` | Finds the samples in a condition folder and loads their images |
| `Functions/func_detect_vesicles.py` | Vesicle detection (Hough or Cellpose, see above) |
| `Functions/func_detect_measure_vesicle.py` | Detects the vesicles in every image and measures each one, giving one table of every vesicle |
| `Functions/func_summary_tables_vesicle.py` | Number, mean, median and % localised vesicles for each condition and each image |
| `Functions/func_violin_plot.py` | Violin plots and axis styling |
| `Functions/func_test_detection_vesicle.py` | QC figure: detected vesicles and the regions measured, for one image |
| `Functions/func_plot_MES_for_sample_vesicle.py` | QC figure: each vesicle coloured by its measured fluorescence, and numbered |
| `Functions/func_line_profile_vesicle.py` | QC figure: the 4 line profiles of one vesicle, next to the image |
| `Functions/matlab_helpers.py` | Python versions of the MATLAB functions used (`imfindcircles`, `imread`, `im2gray`, `round`, `trimmean`, `bwperim`, ...) |

## Data format

The images are kept in a separate working folder, with one folder per condition:

```
working_folder/
├── noNTPs_t0/                          # one folder per condition
│   ├── A. noNTPs_t0_DYE.png            # repeat A, marker dye (used for detection)
│   ├── A. noNTPs_t0_MES.png            # repeat A, fluorescence to measure
│   ├── A. noNTPs_t0_Background.png     # repeat A, negative background (not needed for 'MEMBRANE-INNER')
│   ├── B. noNTPs_t0_DYE.png
│   └── ...
├── noNTPs_t2/
├── withNTPs_t0/
└── withNTPs_t2/
```

- The images can be colour or greyscale `.png` files (8 or 16 bit).
- Images are named `repeat. condition`. The text before the first `.` is the biological repeat.
- The condition folder names are used as the graph labels (these can be changed in the notebook).
- Set `pixel_size_micron` in the first cell to match your microscope (default: 124.5 µm across 1920 pixels), as the vesicle diameter limits are given in microns.

## How to run

1. Requirements: Python 3.10 or later. Install the packages once with:
   ```
   pip install numpy scipy scikit-image matplotlib pandas pillow seaborn ipykernel ipympl
   ```
   To use Cellpose detection, also install `pip install cellpose` (it uses the GPU if there is one, including Apple silicon).  
   Tested with Python 3.13, numpy 2.4, scipy 1.17, scikit-image 0.26, matplotlib 3.10, pandas 3.0, seaborn 0.13 and Cellpose 4.2, in VS Code on macOS.
2. Open `MAIN_SCRIPT_VESICLE.ipynb` in VS Code (or Jupyter) and choose the Python environment as the kernel.
3. In the first cell, set `working_folder` to the folder containing your condition folders (a full path, or a path relative to the notebook's folder). The `Functions` folder is found automatically.
4. Run the cells in order (Shift + Enter):
   1. **Input parameters:** the conditions, measurement type, detection method and settings.
   2. **Test parameters:** check the detection and the measured regions on one image, and adjust the settings if needed.
   3. **Detect & measure:** processes every image, giving the `vesicleData` table (one row per vesicle). Re-run this whenever you change the input parameters. The check figures give an error if you forget.
   4. **Summary tables:** number, mean, median and % localised vesicles for each condition (repeats pooled) and each image.
   5. **Violin plots:** all repeats pooled, then each repeat side by side. The y axis limits can be lowered to hide sparse high values. They are still included in the violins and means, and the number hidden is printed.
   6. **Check measured fluorescence:** colours and numbers each vesicle in one image by its measured value.
   7. **Check line profiles:** the line profiles of one numbered vesicle, with the membrane search region and the averages used.

Figures are interactive (zoom and pan, like MATLAB figures) using `ipympl`. Change `%matplotlib widget` to `%matplotlib inline` in the first cell for static images. Tables and figures can be saved with the commented lines in each cell.

## Acknowledgements

Cellpose: Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods* 18, 100–106 (2021).

## License

MIT License, see [LICENSE](LICENSE).
