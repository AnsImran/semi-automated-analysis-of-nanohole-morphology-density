## Semi-Automatic Nanohole Morphology Pipeline

This repository contains the tooling I built during my internship at the Institut des Nanosciences de Paris (INSP), Sorbonne University, Paris, for the Coupled Quantum Dots for Quantum Computation project. The goal is to turn raw AFM height maps of nanoholes (fabricated using molecular-beam-epitaxy (MBE) & Local Droplet Etching (LDE)) into reliable depth, rim-area, and shape statistics so we can correlate growth/fabrication parameters with resulting quantum-dot-ready morphologies. The full internship report that motivates the design choices is available at https://drive.google.com/file/d/1wkk0U2tV2VnF8PwFoKwcEoMVh8tzEAGN/view.

The pipeline is semi-automatic on purpose: after experimenting with fully automated approaches (documented in the report), the mixed manual/automatic flow turned out to be the most robust and least error-prone for large AFM datasets. It reduces weeks of manual analysis down to a day or two, even for hundreds of scans, while still letting the researcher make the critical visual decisions that algorithms routinely miss.

### Technical highlights

- Built in Python with NumPy, Pandas, SciPy, Matplotlib, Plotly, and related scientific data analysis stacks to keep the workflow reproducible.
- Designed as a resilient data analysis pipeline (yes, a deliberate data analysis pipeline as well) for AFM height maps captured during Atomic Force Microscopy Scans.
- Focused on high-throughput yet human-aware scientific data analysis so researchers can iterate growth parameters quickly while maintaining traceability.

---

### Repository layout

- `1_click_hole_coordinates.py` - interactive tool that cycles through AFM `.npy` height maps and records nanohole centers that you click on.
- `2_automatic_hole_depth_&_size.py` - loads the click log, zooms into every nanohole, lets you select rim contours, fits ellipses, and computes depths, diameters, eccentricities, and additional statistics.
- `data_analysis.ipynb` - convenience notebook to visualize the aggregated outputs and run extra analyses (histograms, scatter plots vs. process parameters, etc.).
- `files/` - demo AFM scans (`*.npy`) plus sample outputs (`clicked_points_20250630_124212.json` and `dict_clicked_points_20250630_124212.json`).
- `pyproject.toml` / `requirements.txt` - dependency definitions (Matplotlib, NumPy, SciPy, scikit-image, Plotly, etc.).

---

### Getting started

1. **Create a virtual environment**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

   (Alternatively, use `uv sync` if you prefer `uv` for lock-file aware installs.)

3. **Add AFM scans**

   Place your exported AFM height maps (saved as `.npy` arrays) inside `./files`. The sample scans shipped here are 5 um x 5 um windows on InAs nanoholes.

---

### Workflow

1. **Log nanohole centers**
   - Run `python 1_click_hole_coordinates.py`.
   - The script lists all `.npy` files inside `files/`; type the filename you want to inspect.
   - A Matplotlib viewer opens so you can visually locate nanoholes and click their approximate centers. Each click is written immediately to `files/clicked_points_<timestamp>.json`.
   - Close the figure to move on to another scan or exit. The JSON file stores, per scan, the list of `(x, y)` pixel indices you validated plus a count of how many nanoholes were tagged.

2. **Measure morphology**
   - Run `python 2_automatic_hole_depth_&_size.py` and point it to the JSON produced in step 1 (the default filenames in the script already match the sample data).
   - For each clicked nanohole the script:
     - Loads the correct AFM matrix and crops a zoomed region around the click.
     - Lets you interactively pick the outer rim by selecting a contour level that hugs the lip of the nanohole.
     - Fits an ellipse to the rim, converts pixel distances to nanometers (make sure `pixels_to_nano_meters` matches your scan size), and logs depth, rim area, eccentricity, rim height, and distance between rim center and lowest point.
     - Stores per-hole metrics plus per-scan aggregates (means, standard deviations, density estimates) in `files/dict_clicked_points_<timestamp>.json`.

3. **Explore the dataset**
   - Open `data_analysis.ipynb` to visualize distributions, correlate morphologies with MBE growth parameters, and prepare plots for reporting.

This flow intentionally keeps a human-in-the-loop for the two steps that most affect accuracy: identifying nanoholes in complex AFM landscapes and setting the rim level. Everything else (statistics, ellipse fitting, derived metrics) is automated and repeatable.

---

### Output formats

#### `clicked_points_<timestamp>.json`

```json
{
  "20250320_22R033__250321_Z Height_Forward_015.npy": {
    "clicked_points": [[48, 190], [655, 500], ...],
    "total_number_of_clicked_points": 6
  }
}
```

Each scan name maps to the pixel coordinates of every nanohole center you approved.

#### `dict_clicked_points_<timestamp>.json`

This file is the heart of the pipeline. For every scan it records:

- `hole_number_<i>` - measurements for each nanohole:
  - `distance_h_e`: offset (nm) between the hole center and fitted ellipse center.
  - `major_axis`, `minor_axis`, `minor_over_major`, `theta`: rim geometry derived from ellipse fitting.
  - `selected_contour_height`: height level used for the rim (in the data's native height units).
  - `depth`: vertical distance between the rim level and the lowest point inside the selected contour.
  - `all_contour_heights`: the sweep of contour levels you inspected while locking in the rim.
  - `useable`: whether the measurement passed manual QC.
- Scan-level rollups such as `depths_std`, `average_hole_depth_for_<scan>`, `density_of_holes_for_<scan>`, and lists of each metric to make plotting easy.

The sample file in `files/dict_clicked_points_20250630_124212.json` is a good reference if you want to integrate this pipeline with downstream analysis tools.

---

### Why semi-automatic?

- AFM scans of local droplet-etched nanoholes routinely contain drift, debris, and partial features - to name a few -that confuse fully automatic segmentation. Manual confirmation of center points and rim placement eliminates those failure modes.
- The scripts still batch the heavy lifting: statistics update live as soon as you validate a hole, and aggregate metrics are ready for plotting without copy/paste gymnastics.
- Compared to doing everything by hand in Gwyddion or similar software, this workflow has already cut our analysis time from weeks to **1-2 days** per growth campaign while delivering higher-confidence morphology readouts for quantum-dot design.

For a deeper dive into the rejected approaches (purely automatic clustering, watershed segmentation, etc.) and validation studies, see the internship report referenced above.

---

### Tips and customization

- **Scan calibration** - inside `2_automatic_hole_depth_&_size.py`, update `pixels_to_nano_meters = 5000 / dims_of_scans` if you are not working with 5 um x 5 um windows.
- **Saving figures** - the first script contains commented-out high-resolution export logic; enable it if you need publication-quality heatmaps.
- **Notebook analysis** - `data_analysis.ipynb` is intentionally lightweight. Adapt it to compare different MBE parameter sets, run ellipse eccentricity histograms, or fit trendlines against droplet density.
- **Code quality** - these scripts were written for rapid lab use, so there is room for refactoring and typing. For a taste of my production-ready code, check out https://github.com/AnsImran/productionReady-langgraph-fastapi-nextjs-template.

---

### Acknowledgements

- INSP Sorbonne Paris - Coupled Quantum Dots for Quantum Computation group.
- Everyone who helped test fully automatic approaches (documented in the report) before settling on this robust semi-automatic pipeline.

Feel free to reach out via ansimran18@gmail.com for questions, suggestions, or collaboration ideas.
