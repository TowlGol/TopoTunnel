# -*- coding: utf-8 -*-
"""
RDP (Ramer-Douglas-Peucker) Path Simplification Utilities

This module provides the Douglas-Peucker algorithm for 3D polyline simplification.
It reduces point count while preserving spatial shape fidelity based on a 
maximum deviation threshold (epsilon).

Usage:
    from path_simplification import rdp_simplify_path
    
    xs_simplified, ys_simplified, zs_simplified = rdp_simplify_path(
        xs, ys, zs, epsilon=0.15, min_points=3
    )
"""

import numpy as np


def rdp_3d(points, epsilon):
    """
    Ramer-Douglas-Peucker algorithm for 3D polylines.
    
    Recursively simplifies a polyline by removing points that are within
    epsilon distance from the line connecting endpoints.
    
    Parameters:
        points: np.ndarray of shape (N, 3) - 3D coordinates
        epsilon: float - maximum perpendicular distance threshold (in same units as points)
    
    Returns:
        np.ndarray of shape (M, 3) - simplified points where M <= N
    """
    n = len(points)
    
    if n <= 2:
        return points
    
    # Line from first to last point
    start = points[0]
    end = points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    # Handle degenerate case where start == end
    if line_len < 1e-10:
        # Find the point furthest from start
        distances = np.linalg.norm(points - start, axis=1)
        max_idx = np.argmax(distances)
        if distances[max_idx] > epsilon:
            return np.vstack([start, points[max_idx], end])
        else:
            return np.array([start, end])
    
    line_unit = line_vec / line_len
    
    # Calculate perpendicular distances for all intermediate points
    # Using vectorized operations for efficiency
    vectors_from_start = points[1:-1] - start
    projections = np.dot(vectors_from_start, line_unit)
    
    # Clamp projections to line segment
    projections = np.clip(projections, 0, line_len)
    
    # Closest points on line
    closest_points = start + np.outer(projections, line_unit)
    
    # Perpendicular distances
    distances = np.linalg.norm(points[1:-1] - closest_points, axis=1)
    
    # Find point with maximum distance
    max_dist_idx = np.argmax(distances)
    max_dist = distances[max_dist_idx]
    
    # Adjust index to account for skipped first point
    actual_idx = max_dist_idx + 1
    
    if max_dist > epsilon:
        # Recursively simplify both segments
        left = rdp_3d(points[:actual_idx + 1], epsilon)
        right = rdp_3d(points[actual_idx:], epsilon)
        
        # Combine results, avoiding duplicate point at split
        return np.vstack([left[:-1], right])
    else:
        # All points are close enough to line, keep only endpoints
        return np.array([start, end])


def rdp_simplify_path(xs, ys, zs, epsilon=0.15, min_points=3):
    """
    Apply RDP simplification to coordinate arrays.
    
    Parameters:
        xs, ys, zs: array-like - x, y, z coordinates
        epsilon: float - maximum deviation threshold (default 0.15 Ångströms)
        min_points: int - minimum number of points to keep (default 3)
    
    Returns:
        tuple of (xs_simplified, ys_simplified, zs_simplified) as np.ndarray
    """
    # Convert to numpy arrays
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)
    
    n = len(xs)
    
    # Don't simplify if already at or below minimum
    if n <= min_points:
        return xs, ys, zs
    
    # Stack into (N, 3) array
    points = np.column_stack([xs, ys, zs])
    
    # Apply RDP
    simplified = rdp_3d(points, epsilon)
    
    # Ensure we keep at least min_points
    if len(simplified) < min_points and n >= min_points:
        # Keep evenly spaced points including endpoints
        indices = np.linspace(0, n - 1, min_points, dtype=int)
        simplified = points[indices]
    
    return (
        simplified[:, 0].astype(np.float32),
        simplified[:, 1].astype(np.float32),
        simplified[:, 2].astype(np.float32)
    )


def rdp_simplify_with_attributes(xs, ys, zs, attributes, epsilon=0.15, min_points=3):
    """
    Apply RDP simplification while preserving associated attribute arrays.
    
    Useful when you need to keep bfactor, residue IDs, etc. in sync with coordinates.
    
    Parameters:
        xs, ys, zs: array-like - x, y, z coordinates
        attributes: dict of {name: array} - associated attribute arrays
        epsilon: float - maximum deviation threshold
        min_points: int - minimum number of points to keep
    
    Returns:
        tuple of (xs, ys, zs, simplified_attributes)
        where simplified_attributes is a dict with the same keys
    """
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)
    
    n = len(xs)
    
    if n <= min_points:
        return xs, ys, zs, attributes
    
    points = np.column_stack([xs, ys, zs])
    
    # Build index mask for RDP
    # We need to track which original indices are kept
    mask = _rdp_mask(points, epsilon)
    
    # Ensure minimum points
    kept_count = np.sum(mask)
    if kept_count < min_points and n >= min_points:
        # Add more points evenly spaced
        indices_needed = min_points - kept_count
        remaining_indices = np.where(~mask)[0]
        if len(remaining_indices) > 0:
            step = max(1, len(remaining_indices) // indices_needed)
            extra_indices = remaining_indices[::step][:indices_needed]
            mask[extra_indices] = True
    
    # Apply mask
    xs_simplified = xs[mask]
    ys_simplified = ys[mask]
    zs_simplified = zs[mask]
    
    simplified_attrs = {}
    for name, arr in attributes.items():
        arr = np.asarray(arr)
        if len(arr) == n:
            simplified_attrs[name] = arr[mask]
        else:
            simplified_attrs[name] = arr  # Leave unchanged if size mismatch
    
    return xs_simplified, ys_simplified, zs_simplified, simplified_attrs


def _rdp_mask(points, epsilon):
    """
    Internal function to build a boolean mask of which points to keep.
    
    Returns:
        np.ndarray of bool, shape (N,)
    """
    n = len(points)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True  # Always keep first
    mask[-1] = True  # Always keep last
    
    _rdp_mask_recursive(points, 0, n - 1, epsilon, mask)
    
    return mask


def _rdp_mask_recursive(points, start_idx, end_idx, epsilon, mask):
    """
    Recursive helper for building RDP mask.
    """
    if end_idx <= start_idx + 1:
        return
    
    start = points[start_idx]
    end = points[end_idx]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        # Find furthest point
        max_dist = 0
        max_idx = start_idx
        for i in range(start_idx + 1, end_idx):
            d = np.linalg.norm(points[i] - start)
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_dist > epsilon:
            mask[max_idx] = True
            _rdp_mask_recursive(points, start_idx, max_idx, epsilon, mask)
            _rdp_mask_recursive(points, max_idx, end_idx, epsilon, mask)
        return
    
    line_unit = line_vec / line_len
    
    # Find point with maximum distance
    max_dist = 0
    max_idx = start_idx
    
    for i in range(start_idx + 1, end_idx):
        vec = points[i] - start
        proj = np.dot(vec, line_unit)
        proj = max(0, min(line_len, proj))
        closest = start + proj * line_unit
        d = np.linalg.norm(points[i] - closest)
        if d > max_dist:
            max_dist = d
            max_idx = i
    
    if max_dist > epsilon:
        mask[max_idx] = True
        _rdp_mask_recursive(points, start_idx, max_idx, epsilon, mask)
        _rdp_mask_recursive(points, max_idx, end_idx, epsilon, mask)


if __name__ == "__main__":
    # Simple test
    print("Testing RDP simplification...")
    
    # Generate a noisy sine wave in 3D
    t = np.linspace(0, 4 * np.pi, 200)
    xs = t
    ys = np.sin(t) + np.random.normal(0, 0.01, len(t))
    zs = np.cos(t) + np.random.normal(0, 0.01, len(t))
    
    print(f"Original points: {len(xs)}")
    
    for eps in [0.05, 0.1, 0.15, 0.3]:
        xs_s, ys_s, zs_s = rdp_simplify_path(xs, ys, zs, epsilon=eps)
        reduction = (1 - len(xs_s) / len(xs)) * 100
        print(f"  epsilon={eps}: {len(xs_s)} points ({reduction:.1f}% reduction)")
    
    print("Done!")
