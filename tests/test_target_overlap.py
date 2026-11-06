#!/usr/bin/env python3
"""
Checks of the target-overlap metric against exact geometry (cylinders/spheres/ellipsoids on a synthetic 'head').
Run with the SimNIBS python:   simnibs_python tests/test_target_overlap.py
"""
import os
import sys
import tempfile

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE)
os.chdir(CODE)
import PlanTUS  # noqa: E402
import TargetOverlap as TO  # noqa: E402

CFG = {
    "min_distance": 50, "max_distance": 120, "optimal_distance": 82.7, "additional_offset": 10.0, "max_angle": 15.0,
    "focal_distance_list": [50, 54.6, 59.2, 63.9, 68.7, 73.6, 78, 82.7, 87.1, 92, 96.8, 101.4, 106.1, 110.7, 115.8, 121.2],
    "flhm_list": [18.1, 18.4, 22.1, 26.6, 31.1, 34.7, 36.9, 41.3, 44, 47.8, 51.9, 54.7, 59.2, 62.8, 66.7, 73.2],
    "target_overlap": True,
}
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s %s" % ("PASS" if ok else "FAIL", name, detail))


def make_mask(path, shapes, size=160):
    """Binary NIfTI on a 1 mm grid centred on the origin. shapes: ('cyl', p0, p1, r) or ('sphere', c, r)."""
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = -size / 2.0
    ijk = np.stack(np.meshgrid(*[np.arange(size)] * 3, indexing="ij"), -1)
    xyz = nib.affines.apply_affine(aff, ijk)
    vol = np.zeros((size,) * 3, np.uint8)
    for sh in shapes:
        if sh[0] == "sphere":
            vol |= (np.linalg.norm(xyz - np.asarray(sh[1]), axis=-1) <= sh[2]).astype(np.uint8)
        else:
            p0, p1, r = np.asarray(sh[1], float), np.asarray(sh[2], float), sh[3]
            ax = p1 - p0
            t = np.clip(((xyz - p0) @ ax) / (ax @ ax), 0, 1)
            vol |= (np.linalg.norm(xyz - (p0 + t[..., None] * ax), axis=-1) <= r).astype(np.uint8)
    nib.save(nib.Nifti1Image(vol, aff), path)
    return path


def head(n=0):
    """Skin vertices on a sphere of radius 70 mm (normals radial) - a stand-in for the scalp."""
    rng = np.random.default_rng(3)
    d = rng.normal(size=(600, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d * 70.0, d


def run(tract, V, N, cfg=CFG, cand=None):
    cand = np.ones(len(V), bool) if cand is None else cand
    return TO.compute(V, N, tract, cfg, PlanTUS, cand, log=lambda *a: None)


def main():
    tmp = tempfile.mkdtemp(prefix="to_test_")
    V, N = head()

    # 1/2/4. a long tube, bigger than the focus
    tube = TO.Tract(make_mask(os.path.join(tmp, "tube.nii.gz"), [("cyl", (-35, 0, 0), (35, 0, 0), 4.0)]))
    res = run(tube, V, N)
    along = np.argmax(V[:, 0] - 3 * np.abs(V[:, 1]) - 3 * np.abs(V[:, 2]))      # vertex at the end of the tube's axis
    across = np.argmax(V[:, 2] - 3 * np.abs(V[:, 0]) - 3 * np.abs(V[:, 1]))     # vertex above the tube
    check("a beam along the tube puts the whole focus inside it (coefficient ~1)", res["coefficient"][along] > 0.95,
          "(%.3f, tilt %.1f deg)" % (res["coefficient"][along], res["tilt"][along]))
    h = 0.5 * 2 * 4.0 / (res["depth"][across] * 0 + 20.0)  # slab half-width / axial semi-axis (about 20 mm at this depth)
    f = res["depth"][across] + 10.0
    a = PlanTUS.compute_FLHM_for_focal_distance(f, CFG["focal_distance_list"], CFG["flhm_list"]) / 2
    hh = 4.0 / a
    exact = (3 * hh - hh ** 3) / 2  # fraction of an ellipsoid within a slab of half-width 4 mm across its long axis
    check("a beam across the tube: the focus fraction inside matches the exact slab volume", abs(res["precision"][across] - exact) < 0.06,
          "(%.3f vs exact %.3f)" % (res["precision"][across], exact))
    F = 4 / 3 * np.pi * (PlanTUS.compute_FLHM_for_focal_distance(res["depth"][along] + 10, CFG["focal_distance_list"], CFG["flhm_list"]) / 2) * 2.5 ** 2
    check("coverage = |focus| / |tube| when the focus lies inside", abs(res["coverage"][along] - F / tube.volume) < 0.04,
          "(%.3f vs %.3f)" % (res["coverage"][along], F / tube.volume))

    # 3. a small target, smaller than the focus: fully covered -> 1 although the focus is mostly outside it
    small = TO.Tract(make_mask(os.path.join(tmp, "small.nii.gz"), [("sphere", (0, 0, 0), 2.2)]))  # narrower than the focus (radius 2.5)
    rs = run(small, V, N)
    feas = rs["feasible"]
    check("a target smaller than the focus: fully covered scores ~1 from every feasible position", feas.sum() > 100 and rs["coefficient"][feas].min() > 0.9,
          "(%d positions, min %.3f)" % (feas.sum(), rs["coefficient"][feas].min()))
    check("...while the focus fraction inside is low (the two fractions are reported separately)", np.median(rs["precision"][feas]) < 0.35,
          "(median %.2f)" % np.median(rs["precision"][feas]))

    # 5/6. a bent tract whose centroid is outside it: the aim point must be chosen inside, and beat the centroid
    bent = TO.Tract(make_mask(os.path.join(tmp, "bent.nii.gz"), [("cyl", (0, 0, 0), (36, 0, 0), 3.5), ("cyl", (0, 0, 0), (0, 36, 0), 3.5)]))
    check("(test set-up) the bent tract's centroid is not inside it", not bent.contains(bent.centroid[None])[0])
    rb = run(bent, V, N)
    fb = rb["feasible"]
    check("every chosen aim point lies inside the tract", bent.contains(rb["aim"][fb]).all(), "(%d positions)" % fb.sum())
    a_c = np.array([PlanTUS.compute_FLHM_for_focal_distance(np.linalg.norm(bent.centroid - v) + 10, CFG["focal_distance_list"], CFG["flhm_list"]) / 2 for v in V[fb]])
    cen_oc = TO.overlap_for_aim(V[fb], bent.centroid, a_c, 2.5, bent, TO.unit_ball_points(512), bent.subsample(800))[0]
    check("aiming at the best point inside beats aiming at the centroid", np.all(rb["coefficient"][fb] >= cen_oc - 0.03) and rb["coefficient"][fb].mean() - cen_oc.mean() > 0.1,
          "(mean %.2f vs %.2f at the centroid)" % (rb["coefficient"][fb].mean(), cen_oc.mean()))

    # 7. depth limit
    cfg2 = dict(CFG)
    cfg2["max_distance"] = 60
    r7 = run(tube, V, N, cfg2)
    base_feasible = int(res["feasible"].sum())
    check("every chosen aim respects max_distance (fewer positions can reach)", (r7["depth"][r7["feasible"]] + 10 <= 60 + 1e-6).all()
          and r7["feasible"].sum() < base_feasible, "(%d feasible vs %d with max_distance 120)" % (r7["feasible"].sum(), base_feasible))
    wide = TO.Tract(make_mask(os.path.join(tmp, "wide.nii.gz"), [("sphere", (0, 0, 0), 3.0)]))  # wider than the focus: cannot be fully covered
    rw = run(wide, V, N)
    check("a target wider than the focus is never reported as more covered than geometry allows", rw["coefficient"][rw["feasible"]].max() < 0.9,
          "(max %.2f; a 5 mm-wide focus covers about 80%% of a 6 mm sphere)" % rw["coefficient"][rw["feasible"]].max())

    # 8. tilt limit
    Nt = N.copy()
    rot = np.cross(N, [0.3, 0.5, 0.8])
    rot /= np.linalg.norm(rot, axis=1)[:, None]
    Nt = N * np.cos(np.radians(40)) + rot * np.sin(np.radians(40))  # normals tipped 40 deg away from the radial direction
    r8 = run(small, V, Nt)
    check("a skin normal tilted beyond max_angle from the beam leaves no feasible aim", (r8["tilt"][r8["feasible"]] <= 15.0).all())
    # 9. ties: with a huge tract many aims score 1; the least tilted should win
    big = TO.Tract(make_mask(os.path.join(tmp, "big.nii.gz"), [("cyl", (-60, 0, 0), (60, 0, 0), 12.0)]))
    r9 = run(big, V, N)
    f9 = r9["feasible"] & (r9["coefficient"] > 0.99)
    check("among aims that tie on overlap the least tilted is chosen", f9.sum() > 20 and np.median(r9["tilt"][f9]) < 6.0,
          "(%d positions, median tilt %.1f deg)" % (f9.sum(), np.median(r9["tilt"][f9])))
    # 10. determinism
    r10 = run(bent, V, N)
    check("deterministic", np.array_equal(r10["coefficient"], rb["coefficient"]) and np.array_equal(r10["aim"], rb["aim"]))
    # 11. aim thinning
    aims = bent.aim_candidates(10)
    check("aim candidates are thinned to the limit, inside the tract and spread along it", len(aims) <= 10 and bent.contains(aims).all() and np.ptp(aims[:, 0]) > 20,
          "(%d aims, x-range %.0f mm)" % (len(aims), np.ptp(aims[:, 0])))
    # 12. elongation hint
    check("elongation tells a tube from a blob", tube.elongation() > 3 and small.elongation() < 1.5, "(tube %.1f, sphere %.1f)" % (tube.elongation(), small.elongation()))
    # 13/14. threshold and settings
    check("minimum overlap removes positions below it", list(TO.apply_minimum(np.array([0.2, 0.5, 0.51, 1.0]), 0.5)) == [0.0, 1.0, 1.0, 1.0])
    off = dict(CFG)
    off["target_overlap"] = False
    check("switched off, the fingerprint is None (old runs stay valid)", TO.fingerprint(off, os.path.join(tmp, "tube.nii.gz")) is None)
    c3 = dict(CFG)
    c3["target_overlap_min"] = 0.7
    check("the fingerprint changes with the minimum overlap", TO.fingerprint(CFG, os.path.join(tmp, "tube.nii.gz")) != TO.fingerprint(c3, os.path.join(tmp, "tube.nii.gz")))
    for bad in ({"target_overlap_min": 1.5}, {"focus_width_mm": 0}):
        try:
            TO.settings({**CFG, **bad})
            check("rejects %s" % bad, False)
        except ValueError:
            check("rejects %s" % bad, True)

    n_fail = RESULTS.count(False)
    print("\n%d checks, %d failed" % (len(RESULTS), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
