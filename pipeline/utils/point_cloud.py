"""Point cloud and RANSAC plane segmentation utilities with automatic fallback.

Provides a drop-in Open3D-compatible interface for PointCloud, segment_plane,
and statistical outlier filtering, ensuring pipeline robustness across all platforms
even when system C-runtimes or display drivers interfere with Open3D's C-bindings.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as _real_o3d
    # Verify open3d basic function actually works
    _p = _real_o3d.geometry.PointCloud()
    HAS_REAL_O3D = True
except Exception:
    _real_o3d = None
    HAS_REAL_O3D = False


class _NumpyVector3d:
    def __init__(self, data: Union[np.ndarray, List]):
        self._data = np.asarray(data, dtype=np.float64)

    def __len__(self) -> int:
        return len(self._data)

    def __array__(self) -> np.ndarray:
        return self._data


class NumpyPointCloud:
    """NumPy-based PointCloud mimicking Open3D PointCloud API."""

    def __init__(self, points: Optional[Union[np.ndarray, List, _NumpyVector3d]] = None):
        if points is None:
            self._points = np.empty((0, 3), dtype=np.float64)
        elif isinstance(points, _NumpyVector3d):
            self._points = points._data
        else:
            self._points = np.asarray(points, dtype=np.float64)

    @property
    def points(self) -> Union[np.ndarray, _NumpyVector3d]:
        return self._points

    @points.setter
    def points(self, pts: Union[np.ndarray, List, _NumpyVector3d]):
        if isinstance(pts, _NumpyVector3d):
            self._points = pts._data
        else:
            self._points = np.asarray(pts, dtype=np.float64)

    def segment_plane(
        self,
        distance_threshold: float = 0.08,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> Tuple[List[float], List[int]]:
        """RANSAC plane segmentation matching Open3D segment_plane."""
        n_pts = len(self._points)
        if n_pts < ransac_n:
            return [0.0, 1.0, 0.0, 0.0], []

        pts = self._points
        best_inliers: np.ndarray = np.empty(0, dtype=int)
        best_plane: List[float] = [0.0, 1.0, 0.0, 0.0]

        # Use fixed random seed for reproducibility
        rng = np.random.default_rng(42)

        for _ in range(num_iterations):
            idx = rng.choice(n_pts, ransac_n, replace=False)
            p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-7:
                continue
            normal /= norm
            d = -float(np.dot(normal, p0))
            dists = np.abs(np.dot(pts, normal) + d)
            inliers = np.where(dists < distance_threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = [float(normal[0]), float(normal[1]), float(normal[2]), d]
                # Early stop if majority inliers found
                if len(best_inliers) > 0.85 * n_pts:
                    break

        return best_plane, best_inliers.tolist()

    def select_by_index(
        self, indices: Union[List[int], np.ndarray], invert: bool = False
    ) -> "NumpyPointCloud":
        """Filter point cloud by integer indices matching Open3D select_by_index."""
        mask = np.zeros(len(self._points), dtype=bool)
        if len(indices) > 0:
            mask[indices] = True
        if invert:
            mask = ~mask
        return NumpyPointCloud(self._points[mask])

    def remove_statistical_outlier(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ) -> Tuple["NumpyPointCloud", List[int]]:
        """Statistical outlier removal matching Open3D remove_statistical_outlier."""
        n_pts = len(self._points)
        if n_pts <= nb_neighbors:
            return NumpyPointCloud(self._points.copy()), list(range(n_pts))

        k = min(nb_neighbors + 1, n_pts)
        tree = cKDTree(self._points)
        dists, _ = tree.query(self._points, k=k)
        # Exclude self distance (index 0)
        mean_dists = np.mean(dists[:, 1:], axis=1)
        mean_val = float(np.mean(mean_dists))
        std_val = float(np.std(mean_dists))
        thresh = mean_val + std_ratio * std_val

        inliers = np.where(mean_dists <= thresh)[0]
        return NumpyPointCloud(self._points[inliers]), inliers.tolist()


class _O3DCompat:
    """Wrapper that exposes open3d or Numpy fallback."""
    def __init__(self):
        if HAS_REAL_O3D:
            self.geometry = _real_o3d.geometry
            self.utility = _real_o3d.utility
            self.io = _real_o3d.io
        else:
            class _Geometry:
                PointCloud = NumpyPointCloud
            class _Utility:
                @staticmethod
                def Vector3dVector(data):
                    return _NumpyVector3d(data)
            self.geometry = _Geometry()
            self.utility = _Utility()
            self.io = None


o3d = _O3DCompat()
