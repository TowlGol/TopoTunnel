"""
Generate Gaussian iso-surface meshes for binding site visualization.

Takes atom coordinates from binding site probe molecules and produces
smooth semi-transparent surfaces using a Gaussian density field +
marching cubes iso-surface extraction via PyVista.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    pv = None  # type: ignore[assignment]
    HAS_PYVISTA = False


_VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "MG": 1.73,
    "ZN": 1.39,
    "FE": 1.56,
    "CA": 1.94,
}

_TARGET_MESH_VOLUME_SCALE = 1.0 / 3.0
_TARGET_MESH_LINEAR_SCALE = _TARGET_MESH_VOLUME_SCALE ** (1.0 / 3.0)


def _vdw_radius_for_element(element: str) -> float:
    key = str(element or "").strip().upper()
    if not key:
        return 1.70
    return _VDW_RADII.get(key, _VDW_RADII.get(key[:1], 1.70))


def _prepare_exclusion_atoms(
    protein_atoms: Optional[Sequence[dict]],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not protein_atoms:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    centers = []
    radii = []
    for atom in protein_atoms:
        element = str(atom.get("element") or atom.get("elem") or "").strip().upper()
        if element.startswith("H"):
            continue
        try:
            pos = np.asarray(atom["position"], dtype=np.float64).reshape(3)
        except Exception:
            continue
        radius = _vdw_radius_for_element(element)
        cutoff = radius + float(margin)
        if np.any(pos < (bbox_min - cutoff)) or np.any(pos > (bbox_max + cutoff)):
            continue
        centers.append(pos)
        radii.append(radius)

    if not centers:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    return (
        np.vstack(centers).astype(np.float64),
        np.asarray(radii, dtype=np.float64),
    )


def _classify_points_outside_atoms(
    points: np.ndarray,
    atom_centers: np.ndarray,
    atom_radii: np.ndarray,
    margin: float,
    chunk_size: int = 512,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0 or atom_centers.size == 0 or atom_radii.size == 0:
        return np.ones(len(pts), dtype=bool)

    cutoff_sq = np.square(np.asarray(atom_radii, dtype=np.float64) + float(margin))
    keep = np.ones(len(pts), dtype=bool)
    for start in range(0, len(pts), chunk_size):
        stop = min(len(pts), start + chunk_size)
        chunk = pts[start:stop]
        diff = chunk[:, np.newaxis, :] - atom_centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        inside = np.any(dist_sq <= cutoff_sq[np.newaxis, :], axis=1)
        keep[start:stop] = ~inside
    return keep


def _choose_surface_shrink_factor(
    points: np.ndarray,
    atom_centers: np.ndarray,
    atom_radii: np.ndarray,
    margin: float,
    candidate_factors: Sequence[float] = (1.0, 0.97, 0.94, 0.91, 0.88, 0.85),
) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0 or atom_centers.size == 0 or atom_radii.size == 0:
        return 1.0

    center = np.mean(pts, axis=0)
    best_factor = 1.0
    best_overlap = len(pts) + 1

    for factor in candidate_factors:
        scaled = center + (pts - center) * float(factor)
        keep = _classify_points_outside_atoms(
            scaled,
            atom_centers,
            atom_radii,
            margin=margin,
        )
        overlap_count = int((~keep).sum())
        if overlap_count < best_overlap:
            best_overlap = overlap_count
            best_factor = float(factor)
        if overlap_count == 0:
            return float(factor)

    return best_factor


def _shrink_surface_uniformly(
    surface: "pv.PolyData",
    center: np.ndarray,
    factor: float,
) -> Optional["pv.PolyData"]:
    if surface is None or surface.n_points < 3:
        return surface
    factor = float(factor)
    if factor >= 0.999:
        return surface
    try:
        shrunk = surface.copy(deep=True)
        pts = np.asarray(shrunk.points, dtype=np.float64)
        ctr = np.asarray(center, dtype=np.float64).reshape(3)
        shrunk.points = ctr + (pts - ctr) * factor
        return shrunk
    except Exception:
        return surface


def _apply_target_mesh_scale(base_factor: float) -> float:
    factor = float(base_factor) * _TARGET_MESH_LINEAR_SCALE
    return max(0.05, min(1.0, factor))


def _reduce_overlap_by_uniform_shrink(
    surface: "pv.PolyData",
    ligand_coords: np.ndarray,
    atom_centers: np.ndarray,
    atom_radii: np.ndarray,
    margin: float,
) -> Optional["pv.PolyData"]:
    if surface is None or surface.n_points < 3:
        return surface

    factor = 1.0
    if atom_centers.size != 0 and atom_radii.size != 0:
        factor = _choose_surface_shrink_factor(
            np.asarray(surface.points, dtype=np.float64),
            atom_centers,
            atom_radii,
            margin=margin,
        )
    factor = _apply_target_mesh_scale(factor)
    center = np.mean(np.asarray(ligand_coords, dtype=np.float64), axis=0)
    return _shrink_surface_uniformly(surface, center=center, factor=factor)


def generate_binding_site_surface(
    coords: np.ndarray,
    sigma: float = 1.45,
    iso_value: float = 0.58,
    grid_spacing: float = 0.9,
    padding: float = 3.2,
    protein_atoms: Optional[Sequence[dict]] = None,
    exclusion_margin: float = 0.35,
) -> Optional["pv.PolyData"]:
    """Generate a smooth Gaussian iso-surface mesh from atom positions.

    Parameters
    ----------
    coords : (N, 3) array
        Atom positions in Angstroms.
    sigma : float
        Gaussian width in Angstroms. Larger = smoother/bigger surface.
    iso_value : float
        Density threshold for the iso-surface. Lower = larger surface.
    grid_spacing : float
        Resolution of the 3D grid. Smaller = finer mesh, more memory.
    padding : float
        Extra space around atom bounding box in Angstroms.
    protein_atoms : sequence of dict, optional
        Aligned protein atoms used to estimate a safe global shrink amount
        for the closed mesh.
    exclusion_margin : float
        Extra clearance beyond protein atom van der Waals radii.

    Returns
    -------
    pv.PolyData or None
        Triangle mesh of the iso-surface, or None on failure.
    """
    if not HAS_PYVISTA:
        return None
    if coords is None or len(coords) < 3:
        return None

    coords = np.asarray(coords, dtype=np.float64)

    # Bounding box with padding
    bbox_min = coords.min(axis=0) - padding
    bbox_max = coords.max(axis=0) + padding
    exclusion_centers, exclusion_radii = _prepare_exclusion_atoms(
        protein_atoms,
        bbox_min,
        bbox_max,
        margin=max(2.0, padding),
    )

    # Grid dimensions
    dims = np.ceil((bbox_max - bbox_min) / grid_spacing).astype(int) + 1
    # Clamp to reasonable size to avoid memory issues
    dims = np.clip(dims, 3, 150)

    grid = pv.ImageData(
        dimensions=tuple(dims),
        spacing=(grid_spacing, grid_spacing, grid_spacing),
        origin=tuple(bbox_min),
    )

    # Compute Gaussian density at each grid point
    grid_points = np.asarray(grid.points, dtype=np.float64)  # (M, 3)
    density = np.zeros(len(grid_points), dtype=np.float64)

    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    # Process atoms in chunks to limit memory
    chunk_size = 32
    for i in range(0, len(coords), chunk_size):
        chunk = coords[i : i + chunk_size]  # (C, 3)
        # Broadcast: (M, 1, 3) - (1, C, 3) => (M, C, 3)
        diff = grid_points[:, np.newaxis, :] - chunk[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)  # (M, C)
        density += np.sum(np.exp(-dist_sq * inv_2sigma2), axis=1)

    grid["density"] = density

    # Extract iso-surface
    try:
        surface = grid.contour(
            isosurfaces=[iso_value],
            scalars="density",
            method="marching_cubes",
        )
    except Exception:
        return None

    if surface is None or surface.n_points < 3:
        return None

    # Keep a clean triangle mesh so wireframe rendering looks regular.
    try:
        surface = surface.triangulate()
        surface = surface.clean()
    except Exception:
        pass

    # PyMOL-like mesh benefits from a coarser, less over-smoothed triangle net.
    try:
        if surface.n_cells > 1200:
            surface = surface.decimate_pro(0.5, preserve_topology=True)
            surface = surface.clean()
    except Exception:
        pass

    # Smooth the surface for a nicer look
    try:
        surface = surface.smooth_taubin(
            n_iter=6,
            pass_band=0.18,
        )
    except Exception:
        pass  # smoothing is optional

    # Compute normals for proper lighting
    try:
        surface.compute_normals(inplace=True, consistent_normals=True)
    except Exception:
        pass

    surface = _reduce_overlap_by_uniform_shrink(
        surface,
        coords,
        exclusion_centers,
        exclusion_radii,
        margin=exclusion_margin,
    )
    if surface is None or surface.n_points < 3 or surface.n_cells < 1:
        return None

    return surface
