#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target overlap ("tube mode"): for targets that are not compact blobs, e.g. a white-matter tract, where aiming at the
target's centroid is wrong (the centroid of a curved tube can lie outside it) and what matters is how much of the
ultrasound focus actually lies in the target.

For every skin vertex this finds the aim point *inside the target* that gives the best overlap between PlanTUS' focal
zone and the target, and scores that overlap:

* Focal zone: PlanTUS' own model, an ellipsoid whose axial length is the calibrated focal length (FLHM, at that
  focal distance) and whose lateral width is 5 mm (``focus_width_mm``). It is centred on the aim point and its long axis
  lies along the beam (skin vertex -> aim point).
* Score = the OVERLAP COEFFICIENT  |focus & target| / min(|focus|, |target|). If the target is bigger than the focus
  this is the fraction of the focus inside the target (a focus neatly inside a large tract scores 1); if the target is
  smaller than the focus it is the fraction of the target covered (a small target fully inside the focus scores 1).
  The two fractions are also returned on their own (``precision`` = focus inside target, ``coverage`` = target covered).
* Limits, as in PlanTUS: the focal distance (skin to aim point + additional_offset) must lie in
  [min_distance, max_distance], and the beam may not tilt more than ``max_angle`` from the skin normal.
* Positions whose best overlap is below ``target_overlap_min`` are removed from the ranking.

Off by default; the pipeline and the viewer are unchanged unless ``target_overlap`` is switched on.
Planning aid only: skull aberration and side lobes are not modelled.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import nibabel as nib
import numpy as np

DEFAULTS = {"target_overlap_min": 0.5, "focus_width_mm": 5.0, "target_overlap_max_aims": 80,
            "target_overlap_focus_points": 512, "target_overlap_tract_points": 800}


def settings(cfg: dict) -> dict:
    """Effective settings: the config's values over the defaults."""
    s = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    if not (0.0 <= float(s["target_overlap_min"]) <= 1.0):
        raise ValueError("target_overlap_min must be between 0 and 1")
    if float(s["focus_width_mm"]) <= 0:
        raise ValueError("focus_width_mm must be > 0")
    return s


def enabled(cfg: dict) -> bool:
    v = cfg.get("target_overlap", False)
    return str(v).lower() in ("true", "tube", "on", "1", "yes")


def fingerprint(cfg: dict, tract_path: str) -> Optional[dict]:
    """What affects the result, for the run-parameter fingerprint (None when switched off, so old runs are unchanged)."""
    if not enabled(cfg):
        return None
    md5 = hashlib.md5(open(tract_path, "rb").read()).hexdigest() if tract_path else None
    return {**{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in settings(cfg).items()}, "target_md5": md5}


class Tract:
    """The target as a binary voxel mask in world (scanner) mm."""

    def __init__(self, path: str):
        img = nib.load(path)
        data = np.asanyarray(img.dataobj)
        data = data[..., 0] if data.ndim == 4 else data
        self.binary = data > 0
        if not self.binary.any():
            raise ValueError("target mask %s is empty" % path)
        self.affine = img.affine
        self._inv = np.linalg.inv(self.affine)
        self.voxel_volume = float(abs(np.linalg.det(self.affine[:3, :3])))
        self.ijk = np.argwhere(self.binary)
        self.points = nib.affines.apply_affine(self.affine, self.ijk)  # voxel centres, mm
        self.volume = len(self.ijk) * self.voxel_volume
        self.centroid = self.points.mean(0)

    def contains(self, xyz: np.ndarray) -> np.ndarray:
        """Is each world point (any shape (..., 3)) inside the target (nearest voxel)?"""
        xyz = np.asarray(xyz)
        flat = xyz.reshape(-1, 3)
        ijk = np.rint(nib.affines.apply_affine(self._inv, flat)).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.binary.shape)), axis=1)
        out = np.zeros(len(flat), dtype=bool)
        out[ok] = self.binary[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return out.reshape(xyz.shape[:-1])

    def subsample(self, n: int) -> np.ndarray:
        """At most n voxel centres, evenly spread through the target (deterministic)."""
        if len(self.points) <= n:
            return self.points
        return self.points[np.linspace(0, len(self.points) - 1, n).astype(int)]

    def aim_candidates(self, max_aims: int) -> np.ndarray:
        """Aim points inside the target: its voxel centres thinned on a grid that is coarsened until at most max_aims
        remain (each kept point is the voxel centre nearest the middle of its cell)."""
        pts = self.points
        step = float(np.min(np.linalg.norm(self.affine[:3, :3], axis=0)))
        while True:
            cells = np.floor(pts / step).astype(np.int64)
            _, inverse = np.unique(cells, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            n_cells = inverse.max() + 1
            if n_cells <= max_aims or step > 200:
                break
            step *= 1.25
        keep = []
        for c in range(n_cells):
            members = np.flatnonzero(inverse == c)
            mid = pts[members].mean(0)
            keep.append(members[np.argmin(np.linalg.norm(pts[members] - mid, axis=1))])
        return pts[np.array(keep)]

    def elongation(self) -> float:
        """Length / width of the target (ratio of the two largest standard deviations along its principal axes)."""
        if len(self.points) < 4:
            return 1.0
        w = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(self.points.T)), 0))[::-1]
        return float(w[0] / max(w[1], 1e-9))


def unit_ball_points(n: int) -> np.ndarray:
    """n quasi-random (Halton) points spread evenly through the unit ball."""
    from scipy.stats import qmc
    pts = 2.0 * qmc.Halton(d=3, scramble=False).random(int(n * 2.4) + 64) - 1.0
    return pts[np.linalg.norm(pts, axis=1) <= 1.0][:n]


def _frame(d: np.ndarray):
    """Two unit vectors perpendicular to each beam direction (rows of d)."""
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    u = np.cross(d, a)
    u /= np.linalg.norm(u, axis=1)[:, None]
    return u, np.cross(d, u)


def overlap_for_aim(V: np.ndarray, aim: np.ndarray, a: np.ndarray, b: float, tract: Tract, E: np.ndarray, Q: np.ndarray):
    """Overlap of the focal ellipsoid (centre ``aim``, axial semi-axis ``a`` per vertex, lateral radius ``b``, long axis along
    vertex -> aim) with the target, for each vertex in V. Returns (coefficient, precision, coverage)."""
    d = aim - V
    d /= np.linalg.norm(d, axis=1)[:, None]
    F = 4.0 / 3.0 * np.pi * a * b * b
    T = tract.volume
    P = np.zeros(len(V))
    R = np.zeros(len(V))
    big = F <= T  # target at least as large as the focus: measure the focus (sampling the smaller object is the accurate way)
    if big.any():
        u, w = _frame(d[big])
        X = (aim + E[None, :, 2:3] * a[big][:, None, None] * d[big][:, None, :] + E[None, :, 0:1] * b * u[:, None, :]
             + E[None, :, 1:2] * b * w[:, None, :])
        P[big] = tract.contains(X).mean(axis=1)
    if (~big).any():
        rel = Q[None, :, :] - aim
        z = np.einsum("vqk,vk->vq", rel, d[~big])
        r2 = np.einsum("vqk,vqk->vq", rel, rel) - z * z
        R[~big] = (((z / a[~big][:, None]) ** 2 + r2 / (b * b)) <= 1.0).mean(axis=1)
    inter = np.where(big, P * F, R * T)  # volume of focus & target
    precision = np.clip(inter / F, 0.0, 1.0)
    coverage = np.clip(inter / T, 0.0, 1.0)
    return np.where(big, precision, coverage), precision, coverage


def compute(skin_xyz: np.ndarray, normals: np.ndarray, tract: Tract, cfg: dict, plantus, candidates: np.ndarray,
            log=print) -> dict:
    """The best aim point and overlap for every candidate skin vertex.

    Returns per-vertex arrays ``coefficient`` (the score), ``precision``, ``coverage``, ``aim`` (N, 3), ``tilt`` (degrees
    from the skin normal), ``depth`` (mm, skin to aim) and ``feasible`` (some aim satisfied the depth and tilt limits),
    plus a ``summary`` dict. Vertices outside ``candidates`` or with no feasible aim get coefficient 0 and aim = the
    target's centroid."""
    from TargetConflict import flhm_function

    s = settings(cfg)
    V_all = np.asarray(skin_xyz, float)
    N_all = np.asarray(normals, float)
    N_all = N_all / np.maximum(np.linalg.norm(N_all, axis=1)[:, None], 1e-12)
    n = len(V_all)
    lo, hi = float(cfg["min_distance"]), float(cfg["max_distance"])
    add, max_tilt = float(cfg["additional_offset"]), float(cfg["max_angle"])
    b = float(s["focus_width_mm"]) / 2.0
    flhm = flhm_function(cfg, plantus)
    E = unit_ball_points(int(s["target_overlap_focus_points"]))
    Q = tract.subsample(int(s["target_overlap_tract_points"]))
    aims = tract.aim_candidates(int(s["target_overlap_max_aims"]))

    best = {"key": np.full(n, -np.inf), "coefficient": np.zeros(n), "precision": np.zeros(n), "coverage": np.zeros(n),
            "aim": np.tile(tract.centroid, (n, 1)), "tilt": np.zeros(n), "depth": np.zeros(n), "feasible": np.zeros(n, bool)}
    cand_idx = np.flatnonzero(candidates)
    for aim in aims:
        d = aim - V_all[cand_idx]
        dist = np.linalg.norm(d, axis=1)
        f = dist + add
        d = d / np.maximum(dist, 1e-9)[:, None]
        tilt = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d, -N_all[cand_idx]), -1.0, 1.0)))
        ok = (f >= lo) & (f <= hi) & (tilt <= max_tilt)
        if not ok.any():
            continue
        idx = cand_idx[ok]
        a = flhm(f[ok]) / 2.0
        for c0 in range(0, len(idx), 400):  # chunks keep the point arrays small
            sl = slice(c0, c0 + 400)
            oc, p, r = overlap_for_aim(V_all[idx[sl]], aim, a[sl], b, tract, E, Q)
            key = oc + 0.05 * 0.5 * (p + r) - 0.0005 * tilt[ok][sl]  # ties: more of both fractions, then less tilt
            better = key > best["key"][idx[sl]]
            tgt = idx[sl][better]
            best["key"][tgt] = key[better]
            best["coefficient"][tgt], best["precision"][tgt], best["coverage"][tgt] = oc[better], p[better], r[better]
            best["aim"][tgt] = aim
            best["tilt"][tgt] = tilt[ok][sl][better]
            best["depth"][tgt] = dist[ok][sl][better]
            best["feasible"][tgt] = True
    ncand = int(cand_idx.size)
    nfeas = int(best["feasible"].sum())
    summary = {"candidates": ncand, "with_a_feasible_aim": nfeas, "aims_tried": int(len(aims)),
               "target_volume_mm3": round(tract.volume, 1), "elongation": round(tract.elongation(), 2),
               "settings": {k: v for k, v in s.items()}}
    for th in (0.25, 0.5, 0.75, 0.9):
        summary["at_least_%d%%" % round(th * 100)] = int(np.count_nonzero(best["coefficient"] >= th))
    log("Target overlap: %d candidate positions, %d can aim into the target within the depth and tilt limits; overlap >= 50%%: %d, "
        ">= 90%%: %d (%d aim points tried)." % (ncand, nfeas, summary["at_least_50%"], summary["at_least_90%"], len(aims)))
    best.pop("key")
    best["summary"] = summary
    return best


def apply_minimum(coefficient: np.ndarray, minimum: float) -> np.ndarray:
    """1 where the overlap is at least ``minimum`` (positions below it are removed), else 0."""
    return (np.asarray(coefficient) >= float(minimum) - 1e-12).astype(float)
