#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target conflict: a transducer placement is penalised when its beam passes through (or close to) a region that should
not be sonicated, given as one or more binary masks in the subject's T1 space.

It is the reverse of PlanTUS' target-intersection metric and is built from the same pieces:

* each conflict mask becomes a smoothed STL with ``PlanTUS.stl_from_nii`` (as the target ROI does),
* one ray per skin vertex is intersected with it by ``PlanTUS.compute_vector_mesh_intersections``,
* the length of beam inside the region is read from the hits (all hits are paired, not only the first four).

Differences from the target-intersection metric, on purpose:

* The ray follows the real beam axis: the line from the skin vertex to the target centre, which is what
  ``prepare_acoustic_simulation`` aims the transducer along. (The target-intersection metric casts along the skin normal.)
* The ray continues past the target by half the calibrated focal length (FLHM / 2 at that vertex's focal distance, the
  far edge of PlanTUS' own focal ellipsoid), or by a fixed ``target_conflict_beyond`` mm.
* ``margin``: the mask is dilated by this many mm first, so "passes through" means "passes within margin of".
* The result is an absolute utility (1 = no conflict), never min-max scaled, so it does not depend on how many vertices
  happen to avoid the region.

Approximations to know about: the beam ends at the target centre + FLHM/2, whereas ``prepare_acoustic_simulation`` may
place the focus a few mm nearer or farther (at the middle of where the skin-normal ray crosses the ROI); and for
``margin`` > 0 the distance is measured from voxel centres, so a margin of 3 mm acts like about 3.4 mm.

Per mask: ``hard`` (any conflict -> utility 0, the position can never be chosen) or ``soft`` (utility falls linearly
with the length of beam inside the region, reaching 0 at ``target_conflict_full_mm``). The composite is multiplied by
``prod(utility_k ** (weight_target_conflict * weight_k))``; with no masks nothing changes.

This is a geometric planning aid. It does not model skull aberration or side lobes; BabelBrain remains the check.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

MODES = ("hard", "soft")
CHUNK_RAYS = 2000  # rays handled at a time when building beam stretches
HARD_EPS_MM = 1e-6  # any beam length inside the region above this vetoes a hard-mode vertex


# -----------------------------------------------------------------------------
# Specs: which masks, and how each one is treated
# -----------------------------------------------------------------------------
def md5_of_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(path: str) -> str:
    stem = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
    return re.sub(r"[^A-Za-z0-9]+", "", stem) or "mask"


def parse_spec(spec, cfg: dict, base_dir: Optional[str] = None) -> dict:
    """One mask spec -> {path, name, margin, mode, weight}.

    ``spec`` is a path, ``"path,margin=3,mode=soft,weight=2"`` (command line) or a dict with the same keys (config).
    Anything not given comes from the config defaults (``target_conflict_margin`` / ``target_conflict_mode``) or 1.0.
    A relative path is taken relative to ``base_dir`` (the config file's folder) when given, else the current folder.
    """
    opts = {}
    if isinstance(spec, dict):
        opts = dict(spec)
        path = opts.pop("path")
    else:
        parts = [p.strip() for p in str(spec).split(",")]
        path = parts[0]
        for p in parts[1:]:
            if "=" not in p:
                raise ValueError("conflict mask option '%s' should look like key=value (margin, mode, weight)" % p)
            k, v = p.split("=", 1)
            opts[k.strip()] = v.strip()
    unknown = set(opts) - {"margin", "mode", "weight"}
    if unknown:
        raise ValueError("unknown conflict mask option(s): %s (use margin, mode, weight)" % ", ".join(sorted(unknown)))
    path = os.path.expanduser(str(path))
    if base_dir and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    path = os.path.abspath(path)
    margin = float(opts.get("margin", cfg.get("target_conflict_margin", 3.0)))
    mode = str(opts.get("mode", cfg.get("target_conflict_mode", "hard"))).lower()
    weight = float(opts.get("weight", 1.0))
    if mode not in MODES:
        raise ValueError("conflict mask mode must be 'hard' or 'soft', not '%s'" % mode)
    if margin < 0 or weight < 0:
        raise ValueError("conflict mask margin and weight must be >= 0")
    return {"path": path, "name": _safe_name(path), "margin": margin, "mode": mode, "weight": weight}


def check_settings(cfg: dict) -> None:
    """Reject settings that would silently give nonsense: a zero/negative full_mm (NaN or no penalty in soft mode), a
    negative weight (0 ** negative = inf would rank vetoed positions first), a negative margin or beyond."""
    if float(cfg.get("weight_target_conflict", 1.0)) < 0:
        raise ValueError("weight_target_conflict must be >= 0 (0 switches target conflict off)")
    if float(cfg.get("target_conflict_full_mm", 20.0)) <= 0:
        raise ValueError("target_conflict_full_mm must be > 0")
    if float(cfg.get("target_conflict_margin", 3.0)) < 0:
        raise ValueError("target_conflict_margin must be >= 0")
    if cfg.get("target_conflict_beyond") is not None and float(cfg["target_conflict_beyond"]) < 0:
        raise ValueError("target_conflict_beyond must be >= 0 (or null for half the focal length)")


def _drop_repeats(specs: List[dict]) -> List[dict]:
    """If the same file is listed more than once, keep the last entry (so an explicit --conflict-mask with options
    replaces the same mask found in a folder or in the config instead of counting it twice)."""
    last = {os.path.realpath(s["path"]): i for i, s in enumerate(specs)}
    return [s for i, s in enumerate(specs) if last[os.path.realpath(s["path"])] == i]


def _unique_names(specs: List[dict]) -> List[dict]:
    seen = {}
    for s in specs:  # compared case-insensitively: LV and lv would be the same file on macOS
        key = s["name"].lower()
        n = seen.get(key, 0)
        seen[key] = n + 1
        if n:
            s["name"] += str(n + 1)
    return specs


def collect_specs(cfg: dict, cli_specs: Sequence[str] = (), base_dir: Optional[str] = None) -> List[dict]:
    """All conflict masks for a run: the config list (relative paths are relative to ``base_dir``, the config's folder)
    plus any given on the command line (already absolute). A file listed twice keeps its last entry; names are made
    unique."""
    check_settings(cfg)
    specs = []
    for r in list(cfg.get("target_conflict_masks") or []) + list(cli_specs or []):
        s = parse_spec(r, cfg, base_dir)
        if not os.path.isfile(s["path"]):
            raise FileNotFoundError("conflict mask not found: %s" % s["path"])
        specs.append(s)
    return _unique_names(_drop_repeats(specs))


def fingerprint(specs: Sequence[dict], cfg: dict) -> Optional[dict]:
    """What affects the result, for PlanTUS' 'is the cached output still valid' check. None when there are no masks,
    so runs without conflict masks keep exactly the fingerprint they always had."""
    if not specs:
        return None
    return {
        "masks": [{"name": s["name"], "md5": md5_of_file(s["path"]), "margin": s["margin"], "mode": s["mode"],
                   "weight": s["weight"]} for s in specs],
        "weight": float(cfg.get("weight_target_conflict", 1.0)),
        "full_mm": float(cfg.get("target_conflict_full_mm", 20.0)),
        "beyond": cfg.get("target_conflict_beyond"),
    }


# -----------------------------------------------------------------------------
# The mask, with its margin
# -----------------------------------------------------------------------------
class ConflictMask:
    """A binary mask (any voxel > 0) with cheap dilation by a margin in mm."""

    def __init__(self, path: str):
        img = nib.load(path)
        data = np.asanyarray(img.dataobj)
        data = data[..., 0] if data.ndim == 4 else data
        self.path = path
        self.binary = data > 0
        if not self.binary.any():
            raise ValueError("conflict mask %s is empty" % path)
        self.affine = img.affine
        self._inv = np.linalg.inv(self.affine)
        self.voxel_mm = np.linalg.norm(self.affine[:3, :3], axis=0)
        self._edt = None

    def region(self, margin_mm: float) -> np.ndarray:
        """Voxels within margin_mm of the mask (the mask itself when margin is 0)."""
        if margin_mm <= 0:
            return self.binary
        if self._edt is None:
            from scipy import ndimage
            self._edt = ndimage.distance_transform_edt(~self.binary, sampling=self.voxel_mm).astype(np.float32)
        return self.binary | (self._edt <= margin_mm)

    def contains(self, region: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Is each world point (N, 3) inside ``region`` (nearest voxel)?"""
        ijk = np.rint(nib.affines.apply_affine(self._inv, xyz)).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(region.shape)), axis=1)
        out = np.zeros(len(xyz), dtype=bool)
        out[ok] = region[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return out

    def write(self, region: np.ndarray, out_path: str) -> None:
        nib.save(nib.Nifti1Image(region.astype(np.uint8), self.affine), out_path)


# -----------------------------------------------------------------------------
# Beam geometry
# -----------------------------------------------------------------------------
def flhm_function(cfg: dict, plantus):
    """Calibrated focal length (FLHM, mm) as a function of focal distance: PlanTUS' own cubic fit, evaluated once on a
    0.5 mm grid and interpolated (compute_FLHM_for_focal_distance refits on every call)."""
    lo, hi = float(cfg["min_distance"]), float(cfg["max_distance"])
    if len(cfg["flhm_list"]) <= 1:
        const = float(plantus.compute_FLHM_for_focal_distance(lo, cfg["focal_distance_list"], cfg["flhm_list"]))
        return lambda f: np.full(np.shape(f), const)
    grid = np.arange(lo, hi + 0.5, 0.5)
    table = np.array([plantus.compute_FLHM_for_focal_distance(f, cfg["focal_distance_list"], cfg["flhm_list"]) for f in grid])
    return lambda f: np.interp(f, grid, table)


def beyond_mm(cfg: dict, distances: np.ndarray, plantus) -> np.ndarray:
    """How far past the target each beam is followed: ``target_conflict_beyond`` mm if set, else half the calibrated
    focal length (FLHM) at that vertex's focal distance - the far edge of PlanTUS' own focal ellipsoid."""
    fixed = cfg.get("target_conflict_beyond")
    if fixed is not None:
        return np.full(len(distances), float(fixed))
    lo, hi = float(cfg["min_distance"]), float(cfg["max_distance"])
    focal = np.clip(np.asarray(distances) + float(cfg["additional_offset"]), lo, hi)  # as prepare_acoustic_simulation does
    return flhm_function(cfg, plantus)(focal) / 2.0


def beam_lengths(origins: np.ndarray, target: np.ndarray, beyond: np.ndarray, mask: ConflictMask, region: np.ndarray,
                 stl_path: Optional[str], plantus, valid: Optional[np.ndarray] = None,
                 fallback_step_mm: float = 0.25) -> np.ndarray:
    """Millimetres of each skin vertex's beam (skin vertex -> target centre -> ``beyond`` mm past it) that lie inside
    ``region``. The mesh hits are PlanTUS' own; every pair of hits bounds a stretch of beam, and each stretch is
    counted when its midpoint is inside the region, so any number of hits (and a beam that starts or ends inside the
    region) is handled. Vertices outside ``valid`` get 0."""
    n = len(origins)
    out = np.zeros(n)
    idx = np.flatnonzero(valid) if valid is not None else np.arange(n)
    if idx.size == 0:
        return out
    o = np.asarray(origins, float)[idx]
    target = np.asarray(target, float)
    vec = (target[idx] if target.ndim == 2 else target) - o
    dist = np.linalg.norm(vec, axis=1)
    unit = vec / dist[:, None]
    total = dist + np.asarray(beyond, float)[idx]  # beam length to follow, per vertex

    hits = None
    if stl_path and os.path.isfile(stl_path) and os.path.getsize(stl_path) > 0:
        # PlanTUS' convention: a ray runs from p to p - L*v, so v is the outward direction
        hits = plantus.compute_vector_mesh_intersections(o, -unit, stl_path, float(total.max()))

    for a in range(0, len(idx), CHUNK_RAYS):  # in chunks of rays, so the step-walking fallback cannot exhaust memory
        ray, t0, t1 = [], [], []
        for i in range(a, min(a + CHUNK_RAYS, len(idx))):
            if hits is not None:
                ts = [0.0] + sorted(d for d in (float(np.linalg.norm(np.asarray(h) - o[i])) for h in hits[i]) if 0.0 < d < total[i])
                ts.append(float(total[i]))
            else:  # no usable mesh (e.g. a mask too small to tessellate): walk the beam in small steps instead
                ts = list(np.arange(0.0, total[i], fallback_step_mm)) + [float(total[i])]
            ts = np.asarray(ts)
            ray.append(np.full(len(ts) - 1, i))
            t0.append(ts[:-1])
            t1.append(ts[1:])
        ray, t0, t1 = np.concatenate(ray), np.concatenate(t0), np.concatenate(t1)
        mid = o[ray] + unit[ray] * (0.5 * (t0 + t1))[:, None]
        inside = mask.contains(region, mid)
        out[idx[a:a + CHUNK_RAYS]] += np.bincount(ray - a, weights=(t1 - t0) * inside, minlength=min(CHUNK_RAYS, len(idx) - a))
    return out


def utility(length_mm: np.ndarray, mode: str, full_mm: float) -> np.ndarray:
    """1 = no conflict. hard: 0 for any beam inside the region; soft: falls linearly to 0 at ``full_mm`` of beam."""
    if mode == "hard":
        return np.where(length_mm > HARD_EPS_MM, 0.0, 1.0)
    return np.clip(1.0 - length_mm / float(full_mm), 0.0, 1.0)


def combine(utilities: Sequence[np.ndarray], specs: Sequence[dict], global_weight: float) -> np.ndarray:
    """prod(u_k ** (global_weight * weight_k)). A zero exponent switches a mask off (0 ** 0 == 1)."""
    f = np.ones(len(utilities[0])) if utilities else np.ones(0)
    for u, s in zip(utilities, specs):
        f = f * np.power(u, float(global_weight) * s["weight"])
    return f


# -----------------------------------------------------------------------------
# The whole metric
# -----------------------------------------------------------------------------
def compute_target_conflict(skin_xyz: np.ndarray, target: np.ndarray, specs: Sequence[dict], cfg: dict, valid: np.ndarray,
                            workdir: str, plantus, log=print, mask_cache: Optional[dict] = None) -> dict:
    """Run the metric for every conflict mask. Writes each mask's margin-dilated NIfTI and its smoothed STL/GIFTI into
    ``workdir`` (``target_conflict-<name>.nii.gz`` ...). Returns
    ``{"factor": per-vertex multiplier for the composite, "per_mask": {name: {"length", "utility", "spec"}},
    "summary": {...}}``. ``mask_cache`` (a dict, optional) keeps each mask's distance map between calls, so repeating the
    computation with a different margin (as the viewer's slider does) only redoes the cheap part."""
    os.makedirs(workdir, exist_ok=True)
    full_mm = float(cfg.get("target_conflict_full_mm", 20.0))
    wt = float(cfg.get("weight_target_conflict", 1.0))
    skin_xyz = np.asarray(skin_xyz, float)
    target = np.asarray(target, float)  # one point for everyone, or one aim point per skin vertex (target overlap)
    dist = np.linalg.norm(skin_xyz - target, axis=1)
    beyond = beyond_mm(cfg, dist, plantus)

    per_mask, utils = {}, []
    for s in specs:
        if mask_cache is not None and s["path"] in mask_cache:
            m = mask_cache[s["path"]]
        else:
            m = ConflictMask(s["path"])
            if mask_cache is not None:
                mask_cache[s["path"]] = m
        region = m.region(s["margin"])
        nii = os.path.join(workdir, "target_conflict-%s.nii.gz" % s["name"])
        m.write(region, nii)
        stl = None
        try:
            plantus.stl_from_nii(nii, 0.25)  # same call as for the target ROI -> target_conflict-<name>_3Dmodel.stl
            stl = os.path.join(workdir, "target_conflict-%s_3Dmodel.stl" % s["name"])
        except Exception as e:  # a region too small to tessellate: fall back to walking the beam
            log("Note: could not build a mesh for conflict mask '%s' (%s); using voxel sampling instead." % (s["name"], e))
        length = beam_lengths(skin_xyz, target, beyond, m, region, stl, plantus, valid=valid)
        u = utility(length, s["mode"], full_mm)
        per_mask[s["name"]] = {"length": length, "utility": u, "spec": s}
        utils.append(u)
        nv = int(np.count_nonzero(valid))
        log("Target conflict '%s' (margin %g mm, %s, weight %g): %d of %d candidate positions have beam inside it."
            % (s["name"], s["margin"], s["mode"], s["weight"], int(np.count_nonzero(length[valid] > HARD_EPS_MM)), nv))

    factor = combine(utils, specs, wt)
    vetoed = int(np.count_nonzero(valid & (factor <= 0)))
    nv = int(np.count_nonzero(valid))
    summary = {"candidates": nv, "vetoed": vetoed, "weight": wt, "full_mm": full_mm,
               "beyond": cfg.get("target_conflict_beyond"),
               "masks": [{"name": s["name"], "file": os.path.basename(s["path"]), "margin": s["margin"], "mode": s["mode"],
                          "weight": s["weight"]} for s in specs]}
    log("Target conflict: %d of %d candidate positions removed (%.1f%%)." % (vetoed, nv, 100.0 * vetoed / max(1, nv)))
    if nv and vetoed == nv:
        raise RuntimeError(
            "Every candidate transducer position conflicts with the conflict mask(s), so none is left. Check that the "
            "mask does not overlap the target, reduce target_conflict_margin, or use mode=soft for that mask.")
    return {"factor": factor, "per_mask": per_mask, "summary": summary}


def write_sidecar(path: str, summary: dict) -> None:
    """Small JSON next to the maps, so the viewer can show (and re-run) what this run used."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# -----------------------------------------------------------------------------
# Staging for the pipeline scripts (02_planTUS.sh / Petra2Target.sh):  python TargetConflict.py prepare ...
# -----------------------------------------------------------------------------
SETTING_KEYS = ("weight_target_conflict", "target_conflict_margin", "target_conflict_mode", "target_conflict_full_mm",
                "target_conflict_beyond")


def prepare(t1_path: str, outdir: str, cfg: dict, cli_specs: Sequence[str] = (), folder: Optional[str] = None,
            base_dir: Optional[str] = None, overlap: bool = False, min_overlap: Optional[float] = None,
            allow_misaligned: bool = False) -> dict:
    """Check and stage the conflict masks for one run.

    Every mask must be on the T1w grid (same shape and affine), unless ``allow_misaligned`` is set: then a mask that is not
    is resampled onto the T1w grid (nearest neighbour) with a warning, for quick tests. A binarised copy, keeping the file's stem so the mask
    keeps its name, is written to ``outdir``. Returns ``{"specs", "settings", "signature"}``: ``settings`` is what the
    scripts merge into the PlanTUS config, and ``signature`` identifies the conflict set-up so a finished run can be
    recognised as out of date when a mask or setting changes ("none" when there are no masks).
    """
    raw = []
    if folder and os.path.isdir(folder):
        raw += sorted(os.path.join(folder, f) for f in os.listdir(folder)
                      if (f.endswith(".nii") or f.endswith(".nii.gz")) and not f.startswith("."))
    raw += list(cli_specs or [])
    check_settings(cfg)
    # Target overlap (tube mode) is staged here too, so one signature covers every setting that makes a finished run out of date.
    ov_settings = {}
    if overlap or str(cfg.get("target_overlap", False)).lower() in ("true", "tube", "on", "1", "yes"):
        ov = float(min_overlap if min_overlap is not None else cfg.get("target_overlap_min", 0.5))
        if not 0.0 <= ov <= 1.0:
            raise ValueError("the minimum overlap must be between 0 and 1")
        ov_settings = {"target_overlap": True, "target_overlap_min": ov}
        for k in ("focus_width_mm", "target_overlap_max_aims", "target_overlap_focus_points", "target_overlap_tract_points"):
            if k in cfg:
                ov_settings[k] = cfg[k]
    if not raw:
        if not ov_settings:
            return {"specs": [], "settings": {}, "signature": "none"}
        sig = hashlib.md5(json.dumps({"overlap": ov_settings}, sort_keys=True).encode()).hexdigest()
        return {"specs": [], "settings": ov_settings, "signature": sig}
    parsed = []
    for r in raw:
        s = parse_spec(r, cfg, base_dir)
        if not os.path.isfile(s["path"]):
            raise FileNotFoundError("conflict mask not found: %s" % s["path"])
        parsed.append(s)
    parsed = _drop_repeats(parsed)  # e.g. a mask in the folder that is also given with options: the explicit one wins
    t1 = nib.load(t1_path)
    os.makedirs(outdir, exist_ok=True)
    specs, used = [], set()
    for s in parsed:
        img = nib.load(s["path"])
        data = np.asanyarray(img.dataobj)
        data = data[..., 0] if data.ndim == 4 else data
        if img.shape[:3] != t1.shape[:3] or not np.allclose(img.affine, t1.affine, atol=1e-3):
            if not allow_misaligned:
                raise ValueError("conflict mask %s is not on the T1w grid (shape/affine differ); resample it to the T1w first"
                                 % os.path.basename(s["path"]))
            from nibabel.processing import resample_from_to
            print("Warning: conflict mask %s is not on the T1w grid; resampled onto it (nearest neighbour) because of --allow-misaligned"
                  % os.path.basename(s["path"]), file=sys.stderr)
            data = np.asanyarray(resample_from_to(nib.Nifti1Image((data > 0).astype(np.uint8), img.affine), (t1.shape[:3], t1.affine), order=0).dataobj)
        data = data > 0
        if not data.any():
            raise ValueError("conflict mask %s is empty" % os.path.basename(s["path"]))
        stem = os.path.basename(s["path"]).replace(".nii.gz", "").replace(".nii", "")
        unique, n = stem, 1
        while unique in used:  # two different files with the same name must not overwrite each other when staged
            n += 1
            unique = "%s_%d" % (stem, n)
        used.add(unique)
        staged = os.path.join(outdir, unique + ".nii.gz")
        out = nib.Nifti1Image(data.astype(np.uint8), t1.affine)
        out.set_sform(t1.affine, 1)
        out.set_qform(t1.affine, 1)
        nib.save(out, staged)
        s["path"], s["name"] = staged, _safe_name(staged)
        specs.append(s)
    specs = _unique_names(specs)
    settings = {k: cfg[k] for k in SETTING_KEYS if k in cfg}
    settings["target_conflict_masks"] = [{"path": s["path"], "margin": s["margin"], "mode": s["mode"], "weight": s["weight"]}
                                         for s in specs]
    payload = fingerprint(specs, {**cfg, **settings})
    if ov_settings:
        payload = {"conflict": payload, "overlap": ov_settings}  # (the format is unchanged when overlap is off)
    sig = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"specs": specs, "settings": {**settings, **ov_settings}, "signature": sig}


def _main(argv=None) -> int:
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="Target conflict helpers for the pipeline scripts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="check and stage conflict masks; print the run's conflict signature")
    p.add_argument("--t1", required=True)
    p.add_argument("--outdir", required=True, help="where the binarised copies are written")
    p.add_argument("--config", help="the PlanTUS yaml (for the defaults)")
    p.add_argument("--dir", help="a folder of conflict masks (every .nii/.nii.gz in it)")
    p.add_argument("--mask", action="append", default=[], metavar="FILE[,margin=MM][,mode=hard|soft][,weight=W]")
    p.add_argument("--weight", type=float, help="weight_target_conflict")
    p.add_argument("--margin", type=float, help="default margin in mm")
    p.add_argument("--mode", choices=MODES, help="default mode")
    p.add_argument("--target-overlap", action="store_true", help="target overlap (tube mode) is on")
    p.add_argument("--min-overlap", type=float, help="minimum overlap fraction (0-1) for target overlap")
    p.add_argument("--allow-misaligned", action="store_true", help="resample masks that are not on the T1w grid instead of stopping")
    p.add_argument("--settings-json", help="write the settings to merge into the PlanTUS config here")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config)) if a.config else {}
    for key, val in (("weight_target_conflict", a.weight), ("target_conflict_margin", a.margin), ("target_conflict_mode", a.mode)):
        if val is not None:
            cfg[key] = val
    try:
        res = prepare(a.t1, a.outdir, cfg, a.mask, a.dir, os.path.dirname(os.path.abspath(a.config)) if a.config else None,
                      overlap=a.target_overlap, min_overlap=a.min_overlap, allow_misaligned=a.allow_misaligned)
    except (ValueError, FileNotFoundError) as e:
        sys.exit("Error: %s" % e)
    if a.settings_json:
        with open(a.settings_json, "w") as f:
            json.dump(res["settings"], f, indent=2)
    print(res["signature"])
    for s in res["specs"]:
        print("conflict mask %s: margin %g mm, %s, weight %g" % (s["name"], s["margin"], s["mode"], s["weight"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
