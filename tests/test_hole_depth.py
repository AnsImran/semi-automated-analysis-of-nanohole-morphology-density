import importlib.util  # Dynamically load the production script whose filename starts with a numeral.
import json  # Inspect JSON payloads emitted by helper utilities.
import sys  # Register dynamically imported modules so patches resolve correctly.
from pathlib import Path  # Construct paths in a cross-platform manner.
from tempfile import TemporaryDirectory  # Provide isolated directories for filesystem interactions.
import unittest  # Lightweight harness requested by the user.
from unittest.mock import patch  # Replace interactive matplotlib calls during smoke tests.

import matplotlib  # Force a headless backend for safe automated execution.
import numpy as np  # Build synthetic AFM scans and annotation artefacts.

matplotlib.use("Agg")  # Ensure matplotlib never tries to open GUI windows during the tests.


def load_hole_module():
    """Import the second script dynamically so we can access its helpers."""
    module_path = Path(__file__).resolve().parents[1] / "2_automatic_hole_depth_&_size.py"
    spec = importlib.util.spec_from_file_location("hole_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hole_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HOLE_MODULE = load_hole_module()


class HoleDepthTests(unittest.TestCase):
    """Unit tests targeting helper functions from the second pipeline script."""

    def test_load_json_roundtrip(self) -> None:
        """Verify that JSON payloads are read exactly as they were written."""
        with TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "clicks.json"
            payload = {"scan.npy": {"clicked_points": [[1, 2], [3, 4]], "total_number_of_clicked_points": 2}}
            with open(json_path, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle)
            loaded = HOLE_MODULE.load_json(str(json_path))
            self.assertEqual(loaded, payload)

    def test_find_lowest_point_basic(self) -> None:
        """Ensure the lowest value inside a window is located correctly."""
        with TemporaryDirectory() as tmp_dir:
            array_path = Path(tmp_dir) / "scan.npy"
            matrix = np.array([[0, -5, 3], [2, -10, 4], [1, 6, 7]], dtype=float)
            np.save(array_path, matrix)
            lowest = HOLE_MODULE.find_lowest_point(str(array_path), 1, 1, 3)
            self.assertEqual(lowest, (1, 1))

    def test_fit_ellipse_returns_parameters(self) -> None:
        """Fit an ellipse to synthetic data and check the output tuple length."""
        angles = np.linspace(0, 2 * np.pi, 200)
        x = 5 * np.cos(angles) + 10
        y = 3 * np.sin(angles) - 4
        params = HOLE_MODULE.fit_ellipse(x, y)
        self.assertEqual(len(params), 5)

    @patch("matplotlib.pyplot.show")
    def test_plot_square_contour_smoke(self, mock_show) -> None:
        """Exercise the contour plotting helper with a tiny synthetic scan."""
        with TemporaryDirectory() as tmp_dir:
            array_path = Path(tmp_dir) / "synthetic.npy"
            matrix = np.zeros((10, 10), dtype=float)
            matrix[5, 5] = -1.0
            np.save(array_path, matrix)
            result = HOLE_MODULE.plot_square_contour(str(array_path), 5, 5, 5)
            self.assertEqual(len(result), 6)
            mock_show.assert_called()


if __name__ == "__main__":
    unittest.main()
