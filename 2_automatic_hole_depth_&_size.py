"""Interactive nanohole morphology measurement script with inline narration, typing, logging, and configuration support."""

import copy  # Merge user-supplied configuration safely with built-in defaults.
import json  # Persist click logs and measurement dictionaries.
import logging  # Provide structured diagnostics alongside console prints.
import os  # Resolve filesystem paths for scan inputs and outputs.
from argparse import ArgumentParser  # Offer a lightweight CLI without altering workflow.
from typing import Any, Dict, List, Optional, Tuple  # Clarify data structures through typing.

import matplotlib.pyplot as plt  # Render contour plots for rim selection.
import numpy as np  # Manipulate AFM height maps and perform numeric calculations.
from matplotlib.backend_bases import MouseEvent  # Annotate interactive callbacks with typing.
from matplotlib.patches import Ellipse  # Overlay fitted ellipses so users can validate geometry.
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Align colorbars with contour plots just like before.
from scipy.optimize import least_squares  # Fit ellipses using the original least-squares routine.

Coordinate = Tuple[int, int]
ContourMetrics = Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[List[float]],
    Optional[float],
]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

DEFAULT_OUTPUT_PREFIX = "dict_"
DEFAULT_CONFIG_PATH = "config.json"
MAX_EXPECTED_PIXELS = 4096 * 4096

CONFIG_FALLBACK: Dict[str, Any] = {
    "defaults": {
        "clicked_points_file": "clicked_points_20250630_124212.json",
        "files_directory": "./files",
        "scan_size_pixels": 2048,
        "scan_area_square_micrometers": 25,
        "nominal_scan_length_nanometers": 5000,
    },
    "presets": {
        "1024": {
            "square_size_for_lowest_point_finder": 20,
            "big_square_size_for_highest_point_finder": 41,
            "small_square_size_for_highest_point_finder": 5,
            "length_of_line_for_height_profile_extractors": 41,
            "outer_square_length_for_annular_square_region_extractor": 102,
            "inner_square_length_for_annular_square_region_extractor": 60,
        },
        "2048": {
            "square_size_for_lowest_point_finder": 40,
            "big_square_size_for_highest_point_finder": 81,
            "small_square_size_for_highest_point_finder": 11,
            "length_of_line_for_height_profile_extractors": 81,
            "outer_square_length_for_annular_square_region_extractor": 204,
            "inner_square_length_for_annular_square_region_extractor": 120,
        },
        "fallback": {
            "square_size_for_lowest_point_finder": 40,
            "big_square_size_for_highest_point_finder": 81,
            "small_square_size_for_highest_point_finder": 11,
            "length_of_line_for_height_profile_extractors": 81,
            "outer_square_length_for_annular_square_region_extractor": 204,
            "inner_square_length_for_annular_square_region_extractor": 120,
        },
    },
}

clicked_points = CONFIG_FALLBACK["defaults"]["clicked_points_file"]
FILES_DIRECTORY = CONFIG_FALLBACK["defaults"]["files_directory"]
path = os.path.join(FILES_DIRECTORY, clicked_points)
new_path = os.path.join(FILES_DIRECTORY, f"{DEFAULT_OUTPUT_PREFIX}{clicked_points}")


def ensure_directory_exists(target_directory: str) -> None:
    """Create the working directory when it is missing so downstream reads do not fail."""
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)


def warn_if_unexpected_shape(matrix: np.ndarray, scan_name: str) -> None:
    """Log warnings for empty, higher-dimensional, or extremely large scans without halting execution."""
    if matrix.size == 0:
        logger.warning("Scan %s is empty and may not produce meaningful measurements.", scan_name)
    if matrix.ndim != 2:
        logger.warning("Scan %s has %s dimensions; expected exactly 2.", scan_name, matrix.ndim)
    if matrix.size > MAX_EXPECTED_PIXELS:
        logger.warning(
            "Scan %s contains %s pixels, exceeding the recommended %s pixels; rendering might be slow.",
            scan_name,
            matrix.size,
            MAX_EXPECTED_PIXELS,
        )


def load_json(file_name: str) -> Dict[str, Any]:
    """Load JSON data from disk and return it as native Python objects."""
    with open(file_name, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration overrides into the provided base dictionary."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from disk, falling back to built-in defaults when missing."""
    config = copy.deepcopy(CONFIG_FALLBACK)
    if not os.path.exists(config_path):
        logger.info("Config file %s not found. Using built-in defaults.", config_path)
        return config
    try:
        with open(config_path, "r", encoding="utf-8") as file_handle:
            user_config = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read config %s: %s. Falling back to defaults.", config_path, exc)
        return config
    return deep_update(config, user_config)


def prompt_with_default(message: str, default: Optional[int]) -> int:
    """Ask the user for an integer value, accepting the provided default on empty input."""
    prompt = f"{message} [{default}]: " if default is not None else f"{message}: "
    while True:
        try:
            response = input(prompt)
        except EOFError:
            logger.info("EOF received while prompting. Using default value %s.", default)
            return int(default) if default is not None else 0
        cleaned = response.strip()
        if cleaned:
            try:
                return int(cleaned)
            except ValueError:
                print("Please enter a valid integer value.")
                continue
        if default is not None:
            return int(default)
        print("A value is required.")


def safe_std(values: List[float]) -> float:
    """Return the standard deviation for a list or NaN when insufficient data is available."""
    return float(np.std(values)) if values else float("nan")


def safe_average(values: List[float]) -> float:
    """Return the average for a list or NaN when insufficient data is available."""
    return float(np.average(values)) if values else float("nan")


def find_lowest_point(npy_file: str, x: int, y: int, square_size: int) -> Coordinate:
    """Locate the lowest point inside a square window centred at the provided coordinates."""
    data = np.load(npy_file)
    height, width = data.shape
    half_size = square_size // 2
    x_min, x_max = max(0, x - half_size), min(width, x + half_size + 1)
    y_min, y_max = max(0, y - half_size), min(height, y + half_size + 1)
    subarray = data[y_min:y_max, x_min:x_max]
    min_index = np.unravel_index(np.argmin(subarray), subarray.shape)
    min_x = x_min + min_index[1]
    min_y = y_min + min_index[0]
    return min_x, min_y


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Fit an ellipse to a series of contour points using least squares."""

    def ellipse_equation(params: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        xc, yc, a, b, theta = params
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_rot = cos_theta * (x_coords - xc) + sin_theta * (y_coords - yc)
        y_rot = -sin_theta * (x_coords - xc) + cos_theta * (y_coords - yc)
        return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1

    xc_guess = np.mean(x)
    yc_guess = np.mean(y)
    a_guess = (np.max(x) - np.min(x)) / 2
    b_guess = (np.max(y) - np.min(y)) / 2
    theta_guess = 0.0
    initial_params = np.array([xc_guess, yc_guess, a_guess, b_guess, theta_guess])
    result = least_squares(ellipse_equation, initial_params, args=(x, y))
    xc, yc, a, b, theta = result.x
    return xc, yc, a, b, theta


def plot_square_contour(filename: str, x_hole: int, y_hole: int, big_square_size: int) -> ContourMetrics:
    """Display contour plots around a detected hole and collect ellipse statistics."""
    result: List[Optional[Any]] = [None, None, None, None, None, None]
    data = np.load(filename)
    warn_if_unexpected_shape(data, filename)

    half_size = big_square_size // 2
    x_start = max(0, x_hole - half_size)
    x_end = min(data.shape[1], x_hole + half_size + 1)
    y_start = max(0, y_hole - half_size)
    y_end = min(data.shape[0], y_hole + half_size + 1)
    square_data = data[y_start:y_end, x_start:x_end]

    x_coords = np.arange(x_start, x_end)
    y_coords = np.arange(y_start, y_end)
    X, Y = np.meshgrid(x_coords, y_coords)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    contour = ax.contourf(X, Y, square_data, cmap="viridis", levels=30)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(contour, cax=cax, label="Height")

    all_levels_sorted = sorted(contour.levels)
    result[4] = [float(level) for level in all_levels_sorted]

    ax.set_title("Contour Map of the Hole")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    contour_lines = ax.contour(X, Y, square_data, levels=contour.levels, colors="white", alpha=0.5)

    def on_click(event: MouseEvent) -> None:
        if result[0] is not None or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x_click = event.xdata
        y_click = event.ydata
        min_dist = float("inf")
        selected_path = None
        selected_level = None

        for level_index, level in enumerate(contour_lines.levels):
            segments = contour_lines.allsegs[level_index]
            for segment in segments:
                for point in segment:
                    distance = np.sqrt((point[0] - x_click) ** 2 + (point[1] - y_click) ** 2)
                    if distance < min_dist and distance < 5:
                        min_dist = distance
                        selected_path = segment
                        selected_level = level

        result[5] = selected_level

        if selected_path is not None:
            x_contour, y_contour = selected_path.T
            xc, yc, a, b, theta = fit_ellipse(x_contour, y_contour)

            distance = np.sqrt((xc - x_hole) ** 2 + (yc - y_hole) ** 2)
            print(f"Distance between ellipse center ({xc}, {yc}) and hole center ({x_hole}, {y_hole}): {distance}")

            result[0] = distance
            result[1] = 2 * max(a, b)
            result[2] = 2 * min(a, b)
            result[3] = theta

            fig_ellipse, ax_ellipse = plt.subplots(figsize=(6, 6))
            ax_ellipse.set_aspect("equal")

            contour2 = ax_ellipse.contourf(X, Y, square_data, cmap="viridis", levels=30)
            ax_ellipse.contour(X, Y, square_data, levels=[selected_level], colors="white")

            ellipse = Ellipse((xc, yc), 2 * a, 2 * b, angle=np.degrees(theta), edgecolor="red", facecolor="none", linewidth=2)
            ax_ellipse.add_patch(ellipse)

            ax_ellipse.plot(xc, yc, "ro", label="Ellipse Center", markersize=8)
            ax_ellipse.plot(x_hole, y_hole, "bo", label="Hole Center", markersize=8)

            divider_secondary = make_axes_locatable(ax_ellipse)
            cax_secondary = divider_secondary.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(contour2, cax=cax_secondary, label="Height")

            ax_ellipse.set_title(f"Ellipse Fitted to Contour at height = {selected_level:.2e} m")
            ax_ellipse.set_xlabel("X")
            ax_ellipse.set_ylabel("Y")
            ax_ellipse.legend()

            plt.show()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return (
        result[0],
        result[1],
        result[2],
        np.degrees(result[3]) if result[3] is not None else None,
        result[4],
        result[5],
    )


def hole_depth_and_size_main(
    path_of_file_from_interactive_clicks: str,
    config: Dict[str, Any],
    files_directory: str,
    output_path: str,
) -> Dict[str, Any]:
    """Iterate through each clicked hole, collect shape statistics, and record aggregated metrics."""
    data = load_json(path_of_file_from_interactive_clicks)

    dict_of_scan_names_hole_quantities_holes_depths_and_average_depths: Dict[str, Any] = {}

    defaults = config.get("defaults", {})
    presets = config.get("presets", {})

    dims_default = defaults.get("scan_size_pixels")
    scan_area_default = defaults.get("scan_area_square_micrometers")
    scan_length_nm_default = defaults.get("nominal_scan_length_nanometers", 5000)

    dims_of_scans = prompt_with_default("Kindly enter the size of all scans in pixels", dims_default)
    scan_area = prompt_with_default("Kindly enter scan area in square micrometers", scan_area_default)

    preset_values = presets.get(str(dims_of_scans), presets.get("fallback", {}))
    square_size_for_lowest_point_finder = preset_values.get("square_size_for_lowest_point_finder", 40)
    big_square_size_for_highest_point_finder = preset_values.get("big_square_size_for_highest_point_finder", 81)
    # Retain unused parameters for clarity and future tooling hooks.
    small_square_size_for_highest_point_finder = preset_values.get("small_square_size_for_highest_point_finder", 11)
    length_of_line_for_height_profile_extractors = preset_values.get("length_of_line_for_height_profile_extractors", 81)
    outer_square_length_for_annular_square_region_extractor = preset_values.get(
        "outer_square_length_for_annular_square_region_extractor", 204
    )
    inner_square_length_for_annular_square_region_extractor = preset_values.get(
        "inner_square_length_for_annular_square_region_extractor", 120
    )

    scan_length_nanometers = preset_values.get("scan_length_nanometers", scan_length_nm_default)
    pixels_to_nano_meters = scan_length_nanometers / dims_of_scans if dims_of_scans else 0

    for scan_name in data.keys():
        print("\033[1;34mSCAN_NAME: {scan_name}\033[0m".format(scan_name=scan_name))
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name] = {}
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "total_number_of_clicked_points"
        ] = data[scan_name]["total_number_of_clicked_points"]

        npy_file_address = os.path.join(files_directory, scan_name)
        npy_file = np.load(npy_file_address)
        warn_if_unexpected_shape(npy_file, scan_name)

        hole_number = 0
        list_of_hole_depths: List[float] = []
        list_of_distances_bw_hole_center_vs_ellipse_center: List[float] = []
        list_of_major_axes_of_nanohole: List[float] = []
        list_of_minor_axes_of_nanohole: List[float] = []
        list_of_minor_over_major_axes: List[float] = []
        list_of_angles_of_nanohole: List[float] = []
        list_of_contour_heights_of_nanohole: List[List[float]] = []

        for clicked_point in data[scan_name]["clicked_points"]:
            print("clicked_point: ", clicked_point)
            hole_number += 1
            current_clicked_x, current_clicked_y = clicked_point[0], clicked_point[1]

            lowest_point = find_lowest_point(
                npy_file_address,
                current_clicked_x,
                current_clicked_y,
                square_size_for_lowest_point_finder,
            )
            current_hole_x, current_hole_y = int(lowest_point[0]), int(lowest_point[1])
            print(
                "Coords of hole_number {hole_number} are: ({current_hole_x}, {current_hole_y})".format(
                    hole_number=hole_number,
                    current_hole_x=current_hole_x,
                    current_hole_y=current_hole_y,
                )
            )

            repeat_prompt = "y"
            while repeat_prompt == "y":
                (
                    distance,
                    major_axis,
                    minor_axis,
                    theta,
                    all_levels,
                    selected_level,
                ) = plot_square_contour(
                    npy_file_address,
                    current_hole_x,
                    current_hole_y,
                    big_square_size_for_highest_point_finder,
                )
                repeat_prompt = str(input("Do you want to repeat current iteration? (y/n): "))

            hole_key = "hole_number_{hole_number}".format(hole_number=hole_number)
            hole_entry: Dict[str, Any] = {
                "selected_contour_height": selected_level,
                "all_contour_heights": all_levels,
                "useable": "true",
            }

            if any(value is None for value in (distance, major_axis, minor_axis, theta, selected_level)):
                logger.warning(
                    "Hole %s within scan %s did not yield complete contour data. Skipping measurement.",
                    hole_number,
                    scan_name,
                )
                hole_entry["useable"] = "false"
                dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][hole_key] = hole_entry
                continue

            major_axis_nm = major_axis * pixels_to_nano_meters
            minor_axis_nm = minor_axis * pixels_to_nano_meters
            distance_nm = distance * pixels_to_nano_meters

            if major_axis_nm == 0 or minor_axis_nm == 0:
                logger.warning(
                    "Hole %s within scan %s produced a zero-length axis. Skipping aspect ratio to avoid division by zero.",
                    hole_number,
                    scan_name,
                )
                if hole_entry["selected_contour_height"] is not None:
                    hole_entry["selected_contour_height"] = float(hole_entry["selected_contour_height"])
                hole_entry.update(
                    {
                        "distance_h_e": round(float(distance_nm), 4),
                        "major_axis": round(float(major_axis_nm), 4),
                        "minor_axis": round(float(minor_axis_nm), 4),
                        "theta": round(float(theta), 4),
                        "useable": "false",
                    }
                )
                dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][hole_key] = hole_entry
                continue

            print("Distance (nm):", distance_nm)
            print("Major axis (nm):", major_axis_nm)
            print("Minor axis (nm):", minor_axis_nm)
            print("Theta (deg):", theta)
            print("\n")
            print("All contour heights (nm) (sorted):", [level * pixels_to_nano_meters for level in all_levels])
            print("\n")
            print("Selected contour height (nm):", selected_level)

            if selected_level is not None:
                hole_entry["selected_contour_height"] = float(selected_level)
            hole_entry.update(
                {
                    "distance_h_e": round(float(distance_nm), 4),
                    "major_axis": round(float(major_axis_nm), 4),
                    "minor_axis": round(float(minor_axis_nm), 4),
                    "minor_over_major": round(float(minor_axis_nm / major_axis_nm), 4),
                    "theta": round(float(theta), 4),
                }
            )

            list_of_distances_bw_hole_center_vs_ellipse_center.append(float(distance_nm))
            list_of_major_axes_of_nanohole.append(float(major_axis_nm))
            list_of_minor_axes_of_nanohole.append(float(minor_axis_nm))
            list_of_minor_over_major_axes.append(float(minor_axis_nm / major_axis_nm))
            list_of_angles_of_nanohole.append(float(theta))
            list_of_contour_heights_of_nanohole.append(all_levels)

            depth_of_nanohole = selected_level - npy_file[current_hole_y, current_hole_x]
            print("depth_of_nanohole: ", depth_of_nanohole)
            hole_entry["depth"] = depth_of_nanohole
            list_of_hole_depths.append(float(depth_of_nanohole))
            hole_entry["useable"] = "true"

            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][hole_key] = hole_entry
            print("\n", scan_name, "\n")
            print("\n\n\n\n\n")

        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["depths_std"] = safe_std(list_of_hole_depths)
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["distances_std"] = safe_std(
            list_of_distances_bw_hole_center_vs_ellipse_center
        )
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["major_axes_std"] = safe_std(
            list_of_major_axes_of_nanohole
        )
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["minor_axes_std"] = safe_std(
            list_of_minor_axes_of_nanohole
        )
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["minor_over_major_std"] = safe_std(
            list_of_minor_over_major_axes
        )
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["angles_std"] = safe_std(
            list_of_angles_of_nanohole
        )

        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "average_hole_depth_for_{scan_name}".format(scan_name=scan_name)
        ] = safe_average(list_of_hole_depths)
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "density_of_holes_for_{scan_name}".format(scan_name=scan_name)
        ] = (
            dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name]["total_number_of_clicked_points"]
            / scan_area
            if scan_area
            else float("nan")
        )

        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "list_of_distances_bw_hole_center_vs_ellipse_center_for_{scan_name}".format(scan_name=scan_name)
        ] = list_of_distances_bw_hole_center_vs_ellipse_center
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "list_of_major_axes_of_nanohole_for_{scan_name}".format(scan_name=scan_name)
        ] = list_of_major_axes_of_nanohole
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "list_of_minor_axes_of_nanohole_for_{scan_name}".format(scan_name=scan_name)
        ] = list_of_minor_axes_of_nanohole
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "list_of_minor_over_major_axes_for_{scan_name}".format(scan_name=scan_name)
        ] = list_of_minor_over_major_axes
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "list_of_angles_of_nanohole_for_{scan_name}".format(scan_name=scan_name)
        ] = list_of_angles_of_nanohole
        dict_of_scan_names_hole_quantities_holes_depths_and_average_depths[scan_name][
            "list_of_contour_heights_of_nanohole_for_{scan_name}".format(scan_name=scan_name)
        ] = list_of_contour_heights_of_nanohole

        print("")

        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(dict_of_scan_names_hole_quantities_holes_depths_and_average_depths, file_handle)

        print(dict_of_scan_names_hole_quantities_holes_depths_and_average_depths)

    return dict_of_scan_names_hole_quantities_holes_depths_and_average_depths


def build_parser() -> ArgumentParser:
    """Create a CLI parser so advanced users can override default paths without touching source code."""
    parser = ArgumentParser(description="Compute nanohole depths and shapes from click annotations.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the configuration file (default: %(default)s).")
    parser.add_argument(
        "--clicked-points",
        default=None,
        help="Filename of the clicked-points JSON generated by the first script (defaults to config value).",
    )
    parser.add_argument(
        "--files-dir",
        default=None,
        help="Directory that stores .npy scans and JSON outputs (defaults to config value).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Adjust logging verbosity while retaining the original prints (default: %(default)s).",
    )
    return parser


def main() -> None:
    """Entry point wrapping the legacy behaviour with optional CLI overrides."""
    global clicked_points, FILES_DIRECTORY, path, new_path

    parser = build_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    config = load_config(args.config)
    defaults = config.get("defaults", {})

    files_directory = args.files_dir or defaults.get("files_directory", FILES_DIRECTORY)
    clicked_points_file = args.clicked_points or defaults.get("clicked_points_file", clicked_points)

    ensure_directory_exists(files_directory)

    clicked_points = clicked_points_file
    FILES_DIRECTORY = files_directory
    path = os.path.join(FILES_DIRECTORY, clicked_points)
    new_path = os.path.join(FILES_DIRECTORY, f"{DEFAULT_OUTPUT_PREFIX}{clicked_points}")

    hole_depth_and_size_main(path, config, FILES_DIRECTORY, new_path)


if __name__ == "__main__":
    main()
