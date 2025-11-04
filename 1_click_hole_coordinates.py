import json  # Import the JSON module to handle reading and writing click logs.
import logging  # Import logging to provide detailed runtime diagnostics for operators and tests.
import os  # Import os to work with filesystem paths and directories.
from argparse import ArgumentParser  # Import argparse to create a light CLI wrapper without altering logic.
from datetime import datetime  # Import datetime to timestamp the click-log files for traceability.
from typing import Dict, List, Tuple  # Import typing helpers so every collection is well annotated.

import matplotlib.pyplot as plt  # Import matplotlib for heatmap rendering and interactive clicks.
import numpy as np  # Import numpy to load AFM height maps stored as .npy matrices.
from matplotlib.backend_bases import MouseEvent  # Import MouseEvent to type annotate interactive callbacks.

Coordinate = Tuple[int, int]  # A single click is stored as an (x, y) integer pixel coordinate.
ClickSummary = Dict[str, Dict[str, object]]  # Aggregated click data keyed by scan filename and metadata.

logger = logging.getLogger(__name__)  # Build a module-level logger so everything funnels through one channel.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")  # Configure logging output once.

DEFAULT_DIRECTORY = "files"  # Keep the historical default directory but allow overrides via CLI.
MAX_ALLOWED_PIXELS = 4096 * 4096  # Define a soft sanity ceiling so oversized scans trigger a warning.


def ensure_directory_exists(target_directory: str) -> None:
    """Guarantee that the working directory for .npy files is present before the UI launches."""
    # Check whether the directory already exists to prevent unnecessary creations.
    if not os.path.exists(target_directory):
        # Create the directory so subsequent reads and writes do not fail.
        os.makedirs(target_directory)


def write_click_data(json_path: str, json_data: ClickSummary) -> None:
    """Persist the accumulated click metadata to disk using UTF-8 so tools can read it everywhere."""
    # Attempt to open the file path in write mode, respecting the JSON format used previously.
    try:
        with open(json_path, "w", encoding="utf-8") as file_handle:
            # Dump the in-memory dictionary with indentation for readability during reviews.
            json.dump(json_data, file_handle, indent=4)
    except OSError as exc:
        # Emit a logger error if the filesystem rejects the write so the operator can react.
        logger.error("Unable to write click log %s: %s", json_path, exc)


def validate_matrix_shape(matrix: np.ndarray, file_path: str) -> bool:
    """Inspect the loaded AFM matrix and confirm it looks like a 2D scan before plotting."""
    # If the matrix is empty, surface a warning and refuse to plot to avoid confusing the user.
    if matrix.size == 0:
        logger.warning("Scan %s is empty and will be skipped.", file_path)
        return False
    # If the matrix is not two-dimensional, warn because the heatmap expects 2D inputs.
    if matrix.ndim != 2:
        logger.warning("Scan %s has %s dimensions; expected exactly two.", file_path, matrix.ndim)
        return False
    # If the matrix is larger than our soft threshold, warn but still continue.
    if matrix.size > MAX_ALLOWED_PIXELS:
        logger.warning(
            "Scan %s contains %s pixels, which is larger than the recommended %s. Rendering may be slow.",
            file_path,
            matrix.size,
            MAX_ALLOWED_PIXELS,
        )
    # All checks passed, so signal that the plot can proceed.
    return True


def process_file(file_path: str, json_data: ClickSummary, json_path: str) -> List[Coordinate]:
    """
    Launch an interactive viewer for the provided AFM height map and persist clicked coordinates.

    Parameters
    ----------
    file_path:
        Absolute path to the `.npy` height map to display.
    json_data:
        In-memory store for all click sessions accumulated during the current run.
    json_path:
        Destination path where click data should be written after each selection.

    Returns
    -------
    list[tuple[int, int]]
        List of pixel coordinates corresponding to the user's clicks for the current file.
    """
    # Attempt to load the matrix from disk while keeping the original error handling intact.
    try:
        matrix = np.load(file_path)
    except (OSError, ValueError) as exc:
        # Log the failure and bail out so the main loop can continue with other files.
        logger.error("Failed to load height map %s: %s", file_path, exc)
        return []
    # Validate the shape to avoid passing unexpected arrays to the renderer.
    if not validate_matrix_shape(matrix, file_path):
        return []
    # Capture only the basename for display and for the JSON key.
    file_name = os.path.basename(file_path)
    # Start an empty list to collect every click the user performs during this session.
    clicked_points: List[Coordinate] = []

    def save_to_file() -> None:
        """Persist the current session's clicks to disk without mutating pipeline logic."""
        # Update the in-memory dictionary under the current scan name.
        json_data[file_name] = {
            "clicked_points": clicked_points,
            "total_number_of_clicked_points": len(clicked_points),
        }
        # Delegate to the shared helper so tests can exercise the write path directly.
        write_click_data(json_path, json_data)

    def onclick(event: MouseEvent) -> None:
        """Capture mouse clicks inside the plot axes."""
        # Guard against clicks outside the axes, which report None for xdata/ydata.
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        # Convert the floating-point coordinates to integers while guarding against bad values.
        try:
            x_coord = int(event.xdata)
            y_coord = int(event.ydata)
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring click with non-numeric coordinates: %s", exc)
            return
        # Append the new coordinate to the list, preserving the existing tuple structure.
        clicked_points.append((x_coord, y_coord))
        # Write the updated list immediately so progress is never lost.
        save_to_file()
        # Provide instant feedback in the console just like the original script.
        print("Clicked points:", clicked_points)
        print("Number of points:", len(clicked_points))

    # Build the figure and axes so the heatmap looks identical to the earlier version.
    fig, ax = plt.subplots()
    # Draw the matrix as a heatmap to help the user visually locate nanoholes.
    heatmap = ax.imshow(matrix, cmap="viridis")
    # Attach a colorbar so the user understands the height scale while inspecting the scan.
    plt.colorbar(heatmap)
    # Add the descriptive titles and axis labels to match the historical UI.
    ax.set_title(f"Interactive Height Map - {file_name} - Click to Select Points")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    # Connect the mouse click callback so every click triggers the logging workflow.
    fig.canvas.mpl_connect("button_press_event", onclick)

    # Display the plot and handle any rendering errors gracefully.
    try:
        plt.show()
    except RuntimeError as exc:
        logger.error("Matplotlib encountered an error while displaying %s: %s", file_name, exc)

    # Return the list of clicks so callers have access to the captured coordinates if needed.
    return clicked_points


def build_parser() -> ArgumentParser:
    """Construct a parser so the script can be launched with optional directory overrides."""
    # Instantiate the parser with a brief description to help future operators.
    parser = ArgumentParser(description="Log nanohole coordinates from AFM height maps.")
    # Allow callers to point at a different directory without editing the source file.
    parser.add_argument(
        "--directory",
        default=DEFAULT_DIRECTORY,
        help="Directory containing .npy height maps (default: %(default)s).",
    )
    # Allow callers to override the log level when debugging or running tests.
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set the logging level (default: %(default)s).",
    )
    # Return the configured parser so main can consume the CLI arguments.
    return parser


def main() -> None:
    """Entry point that mirrors the historic while-loop, now behind a CLI-friendly wrapper."""
    # Parse CLI arguments so callers can customize behavior without code edits.
    parser = build_parser()
    args = parser.parse_args()
    # Update the root logger level using the parsed string value.
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    # Determine which directory to scan for .npy files.
    target_directory = args.directory
    # Guarantee the directory exists before attempting to find .npy files.
    ensure_directory_exists(target_directory)
    # Compose the timestamped JSON filename exactly like the previous script.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"clicked_points_{timestamp}.json"
    json_path = os.path.join(target_directory, json_filename)
    # Prepare the dictionary that will accumulate clicks across scans during this session.
    json_data: ClickSummary = {}

    try:
        # Loop until the user decides to exit, preserving the original workflow.
        while True:
            # Gather available .npy files so the user can choose what to inspect.
            npy_files = [f for f in os.listdir(target_directory) if f.endswith(".npy")]
            print("Available .npy files in /files/:")
            for index, file_name in enumerate(npy_files, 1):
                print(f"{index}. {file_name}")

            # Keep asking for a valid filename until the user provides one.
            while True:
                try:
                    file_choice = input("Enter the name of the file you want to open (including .npy): ")
                except EOFError:
                    logger.info("EOF received. Exiting.")
                    raise SystemExit

                file_path = os.path.join(target_directory, file_choice)
                if os.path.exists(file_path):
                    break
                print("File not found. Please try again.")

            # Process the selected file and update the JSON log.
            process_file(file_path, json_data, json_path)

            # Prepare progress reports for the user so they know what has been scanned.
            scanned_files = list(json_data.keys())
            not_scanned_files = [f for f in npy_files if f not in scanned_files]

            print("\nAlready scanned files:")
            if scanned_files:
                for index, file_name in enumerate(scanned_files, 1):
                    print(f"{index}. {file_name}")
            else:
                print("None")

            print("\nFiles not yet scanned:")
            if not_scanned_files:
                for index, file_name in enumerate(not_scanned_files, 1):
                    print(f"{index}. {file_name}")
            else:
                print("None")

            # Prompt the user to continue or exit, preserving the original choices.
            while True:
                try:
                    choice = input("Do you want to (m)ove to another file or (e)xit? ").lower()
                except EOFError:
                    logger.info("EOF received. Exiting.")
                    raise SystemExit

                if choice in ["m", "e"]:
                    break
                print("Please enter 'm' or 'e'")

            # Exit the loop gracefully if the user selects 'e'.
            if choice == "e":
                print(f"Exiting program. Data saved to {json_filename}")
                break
    except KeyboardInterrupt:
        # Handle Ctrl+C the same way as before, reminding users where their data lives.
        logger.info("Interrupted by user. Latest clicks are available in %s.", json_filename)


if __name__ == "__main__":
    # Invoke the CLI entry point when the module is executed directly.
    main()
