#!/usr/bin/env python3
"""
Checks of the target-conflict metric against exact geometry (the chord a straight line cuts through a sphere).
Run with the SimNIBS python:   simnibs_python tests/test_target_conflict.py
Re-run after merging a new upstream PlanTUS: every line should say PASS.
"""
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")
sys.path.insert(0, CODE)
os.chdir(CODE)
import PlanTUS  # noqa: E402
import TargetConflict as TC  # noqa: E402

CFG = {  # a transducer like the lab's DPX-500 (only the keys the metric reads)
    "min_distance": 50, "max_distance": 120, "optimal_distance": 82.7, "additional_offset": 10.0,
    "focal_distance_list": [50, 54.6, 59.2, 63.9, 68.7, 73.6, 78, 82.7, 87.1, 92, 96.8, 101.4, 106.1, 110.7, 115.8, 121.2],
    "flhm_list": [18.1, 18.4, 22.1, 26.6, 31.1, 34.7, 36.9, 41.3, 44, 47.8, 51.9, 54.7, 59.2, 62.8, 66.7, 73.2],
    "target_conflict_beyond": 22.0, "target_conflict_full_mm": 20.0,
}
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print("%s  %s %s" % ("PASS" if ok else "FAIL", name, detail))


def sphere_mask(path, centres_radii, size=140, mm=1.0):
    """A binary NIfTI on a 1 mm grid centred on the world origin, with spheres at the given (centre, radius)."""
    aff = np.diag([mm, mm, mm, 1.0])
    aff[:3, 3] = -np.array([size, size, size]) / 2.0 * mm
    ijk = np.stack(np.meshgrid(*[np.arange(size)] * 3, indexing="ij"), -1)
    xyz = nib.affines.apply_affine(aff, ijk)
    vol = np.zeros((size,) * 3, np.uint8)
    for c, r in centres_radii:
        vol |= (np.linalg.norm(xyz - np.asarray(c), axis=-1) <= r).astype(np.uint8)
    nib.save(nib.Nifti1Image(vol, aff), path)


def chord(origin, target, total, centre, radius):
    """Exact length of the segment origin -> origin + unit*total inside a sphere."""
    o, t, c = map(np.asarray, (origin, target, centre))
    u = (t - o) / np.linalg.norm(t - o)
    b = u @ (o - c)
    disc = b * b - ((o - c) @ (o - c) - radius ** 2)
    if disc <= 0:
        return 0.0
    s0, s1 = -b - np.sqrt(disc), -b + np.sqrt(disc)
    return max(0.0, min(s1, total) - max(s0, 0.0))


def run(tmp, mask_spheres, origins, target, margin, beyond=22.0, valid=None):
    p = os.path.join(tmp, "m.nii.gz")
    sphere_mask(p, mask_spheres)
    m = TC.ConflictMask(p)
    region = m.region(margin)
    nii = os.path.join(tmp, "dilated.nii.gz")
    m.write(region, nii)
    PlanTUS.stl_from_nii(nii, 0.25)
    stl = os.path.join(tmp, "dilated_3Dmodel.stl")
    dist = np.linalg.norm(origins - target, axis=1)
    got = TC.beam_lengths(origins, np.asarray(target, float), np.full(len(origins), beyond), m, region, stl, PlanTUS, valid=valid)
    return got, dist, m, region


def main():
    tmp = tempfile.mkdtemp(prefix="tc_test_")
    target = np.array([0.0, 0.0, 0.0])
    rng = np.random.default_rng(1)
    # origins on a cap 70-100 mm from the target, like skin positions
    dirs = rng.normal(size=(60, 3))
    dirs[:, 2] = np.abs(dirs[:, 2]) + 0.5
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    origins = dirs * rng.uniform(70, 100, size=(60, 1))

    # 1. a sphere on the axis beyond the target (inside the 22 mm 'beyond' range)
    c, r = (0.0, 0.0, -12.0), 6.0
    o = np.array([[0.0, 0.0, 100.0]])
    got, dist, *_ = run(tmp, [(c, r)], o, target, margin=0.0)
    exp = chord(o[0], target, dist[0] + 22.0, c, r)
    check("on-axis sphere past the target", abs(got[0] - exp) < 1.0, "(beam %.2f mm, exact %.2f mm)" % (got[0], exp))

    # 2. margin: the same sphere with a 3 mm margin behaves like a sphere 3 mm larger
    got, dist, *_ = run(tmp, [(c, r)], o, target, margin=3.0)
    exp = chord(o[0], target, dist[0] + 22.0, c, r + 3.0)
    check("margin 3 mm = a 3 mm larger sphere", abs(got[0] - exp) < 1.0, "(beam %.2f mm, exact %.2f mm)" % (got[0], exp))

    # 3. many off-axis beams against an off-centre sphere: compare each with the exact chord
    c2, r2 = (8.0, -6.0, -4.0), 9.0
    got, dist, *_ = run(tmp, [(c2, r2)], origins, target, margin=0.0)
    exp = np.array([chord(origins[i], target, dist[i] + 22.0, c2, r2) for i in range(len(origins))])
    u = (target - origins) / np.linalg.norm(target - origins, axis=1)[:, None]
    off = np.linalg.norm(np.cross(origins - np.array(c2), u), axis=1)  # distance of each beam from the sphere's centre
    robust = np.abs(r2 - off) > 1.0  # the voxelised surface is only good to about half a voxel, so ignore grazes < 1 mm
    err = np.abs(got - exp)
    check("beams clear of the sphere's edge match the exact chord", err[robust].max() < 1.5 and err[robust].mean() < 0.5,
          "(%d beams, max error %.2f mm, mean %.2f mm; %d of them cross the sphere)" % (robust.sum(), err[robust].max(), err[robust].mean(), int((exp[robust] > 0).sum())))
    check("hit / miss agrees for every beam more than 1 mm from the edge", np.all((got[robust] > 1e-6) == (exp[robust] > 0)))
    check("a beam that misses the sphere never registers a conflict", np.all(got[exp == 0] < 1e-6), "(largest false length %.2f mm)" % got[exp == 0].max())

    # 4. a sphere in front of the target (between skin and target)
    c3, r3 = (0.0, 0.0, 40.0), 7.0
    got, dist, *_ = run(tmp, [(c3, r3)], o, target, margin=0.0)
    exp = chord(o[0], target, dist[0] + 22.0, c3, r3)
    check("sphere between skin and target", abs(got[0] - exp) < 1.0, "(beam %.2f mm, exact %.2f mm)" % (got[0], exp))

    # 5. a sphere beyond the 'beyond' distance is ignored; the same sphere inside it is not
    c4, r4 = (0.0, 0.0, -40.0), 6.0
    got_far, *_ = run(tmp, [(c4, r4)], o, target, margin=0.0, beyond=22.0)
    got_near, *_ = run(tmp, [(c4, r4)], o, target, margin=0.0, beyond=40.0)
    check("a region past the 'beyond' range is ignored", got_far[0] < 1e-6, "(%.2f mm)" % got_far[0])
    check("...and counts once the range reaches it", got_near[0] > 2.0, "(%.2f mm)" % got_near[0])

    # 6. two separate regions on one beam: every hit is paired (4 hits), both chords are summed
    a, b = ((0.0, 0.0, 30.0), 6.0), ((0.0, 0.0, -10.0), 5.0)
    got, dist, *_ = run(tmp, [a, b], o, target, margin=0.0)
    exp = chord(o[0], target, dist[0] + 22.0, *a) + chord(o[0], target, dist[0] + 22.0, *b)
    check("two regions on one beam (4 hits)", abs(got[0] - exp) < 1.5, "(beam %.2f mm, exact %.2f mm)" % (got[0], exp))

    # 7. a beam that starts inside the region
    o_in = np.array([[0.0, 0.0, 60.0]])  # inside the test volume (+-70 mm), and inside the sphere
    got, dist, *_ = run(tmp, [((0.0, 0.0, 58.0), 8.0)], o_in, target, margin=0.0)
    exp = chord(o_in[0], target, dist[0] + 22.0, (0.0, 0.0, 58.0), 8.0)
    check("beam starting inside the region", abs(got[0] - exp) < 1.5, "(beam %.2f mm, exact %.2f mm)" % (got[0], exp))

    # 8. vertices outside 'valid' are skipped (and cost nothing)
    valid = np.zeros(len(origins), bool)
    valid[::2] = True
    got_all, *_ = run(tmp, [(c2, r2)], origins, target, margin=0.0)
    got_v, *_ = run(tmp, [(c2, r2)], origins, target, margin=0.0, valid=valid)
    check("vertices that are not candidates stay 0", np.all(got_v[~valid] == 0) and np.allclose(got_v[valid], got_all[valid], atol=1e-6))

    # 9. utilities and weights
    L = np.array([0.0, 0.5, 10.0, 25.0])
    check("hard utility: any beam inside vetoes", list(TC.utility(L, "hard", 20.0)) == [1.0, 0.0, 0.0, 0.0])
    check("soft utility: linear to 0 at full_mm", np.allclose(TC.utility(L, "soft", 20.0), [1.0, 0.975, 0.5, 0.0]))
    u1, u2 = np.array([1.0, 0.0, 0.5]), np.array([0.5, 1.0, 0.5])
    s = [{"weight": 1.0}, {"weight": 2.0}]
    check("combine: prod(u ** (w_global * w_k))", np.allclose(TC.combine([u1, u2], s, 1.0), [0.25, 0.0, 0.5 * 0.25]))
    check("weight 0 switches the metric off", np.allclose(TC.combine([u1, u2], s, 0.0), [1.0, 1.0, 1.0]))

    # 10. half-FLHM 'beyond' from the lab's calibration curve
    cfg = dict(CFG)
    cfg["target_conflict_beyond"] = None
    b = TC.beyond_mm(cfg, np.array([60.0, 72.7, 110.0]), PlanTUS)  # focal distance = + 10 mm offset -> 70, 82.7, 120
    exp = np.array([PlanTUS.compute_FLHM_for_focal_distance(f, cfg["focal_distance_list"], cfg["flhm_list"]) / 2 for f in (70, 82.7, 120)])
    check("'beyond' is half the calibrated focal length", np.allclose(b, exp, atol=0.05), "(%s mm; at the optimum depth %.1f mm)" % (np.round(b, 1), b[1]))

    # 11. spec parsing
    sp = TC.parse_spec("/x/Left DLPFC.nii.gz,margin=5,mode=soft,weight=2", {"target_conflict_margin": 3.0})
    check("command-line spec parses", (sp["name"], sp["margin"], sp["mode"], sp["weight"]) == ("LeftDLPFC", 5.0, "soft", 2.0))
    try:
        TC.parse_spec("/x/m.nii.gz,mode=medium", {})
        check("a bad mode is rejected", False)
    except ValueError:
        check("a bad mode is rejected", True)

    # 12. staging for the pipeline scripts
    t1 = os.path.join(tmp, "t1.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((140,) * 3, np.float32), nib.load(os.path.join(tmp, "m.nii.gz")).affine), t1)
    mk = os.path.join(tmp, "Left DLPFC.nii.gz")
    sphere_mask(mk, [((0.0, 0.0, -12.0), 6.0)])
    res = TC.prepare(t1, os.path.join(tmp, "staged"), {"target_conflict_margin": 3.0}, [mk + ",margin=5,mode=soft"])
    check("prepare stages a binarised copy and keeps the mask's name", res["specs"][0]["name"] == "LeftDLPFC" and os.path.isfile(res["specs"][0]["path"]))
    sig_a = res["signature"]
    sig_b = TC.prepare(t1, os.path.join(tmp, "staged"), {"target_conflict_margin": 3.0}, [mk + ",margin=6,mode=soft"])["signature"]
    sig_c = TC.prepare(t1, os.path.join(tmp, "staged"), {"target_conflict_margin": 3.0}, [mk + ",margin=5,mode=soft"])["signature"]
    check("the signature changes with the margin and is otherwise stable", sig_a != sig_b and sig_a == sig_c)
    check("no masks: signature 'none', nothing to merge", TC.prepare(t1, os.path.join(tmp, "s2"), {}, [])["signature"] == "none")
    off_grid = os.path.join(tmp, "off.nii.gz")
    sphere_mask(off_grid, [((0.0, 0.0, 0.0), 5.0)], size=100)
    try:
        TC.prepare(t1, os.path.join(tmp, "s3"), {}, [off_grid])
        check("a mask off the T1 grid is rejected", False)
    except ValueError as e:
        check("a mask off the T1 grid is rejected", "T1w grid" in str(e))
    rs = TC.prepare(t1, os.path.join(tmp, "s3b"), {}, [off_grid], allow_misaligned=True)  # a 100^3 sphere (radius 5) on a coarser/smaller grid
    st = nib.load(rs["specs"][0]["path"])
    vox = int(np.asanyarray(st.dataobj).sum())
    check("allow_misaligned resamples an off-grid mask onto the T1 grid, keeping its size", st.shape == nib.load(t1).shape
          and np.allclose(st.affine, nib.load(t1).affine) and abs(vox - 523) < 60, "(%d voxels, expected about 523)" % vox)
    # a changed file changes the signature
    sphere_mask(mk, [((0.0, 0.0, -12.0), 7.0)])
    check("editing the mask changes the signature", TC.prepare(t1, os.path.join(tmp, "staged"), {"target_conflict_margin": 3.0}, [mk + ",margin=5,mode=soft"])["signature"] != sig_a)

    # 13. the viewer's live margin: a cached mask gives the same answer as a fresh one, and a larger margin only adds conflicts
    pm = os.path.join(tmp, "live.nii.gz")
    sphere_mask(pm, [((8.0, -6.0, -4.0), 4.0)])  # small enough that a 4 mm margin removes some positions, not all
    spec = [{"path": pm, "name": "live", "margin": 0.0, "mode": "hard", "weight": 1.0}]
    skin = origins.copy()
    valid_all = np.ones(len(skin), bool)
    cache = {}
    cfg = dict(CFG)
    a0 = TC.compute_target_conflict(skin, target, spec, cfg, valid_all, os.path.join(tmp, "w0"), PlanTUS, log=lambda *a: None, mask_cache=cache)
    spec[0]["margin"] = 4.0
    a4 = TC.compute_target_conflict(skin, target, spec, cfg, valid_all, os.path.join(tmp, "w4"), PlanTUS, log=lambda *a: None, mask_cache=cache)
    spec[0]["margin"] = 4.0
    fresh = TC.compute_target_conflict(skin, target, spec, cfg, valid_all, os.path.join(tmp, "w4b"), PlanTUS, log=lambda *a: None)
    check("a cached mask reproduces a fresh computation", np.allclose(a4["factor"], fresh["factor"]) and len(cache) == 1)
    check("a larger margin never frees a vetoed position", np.all(a4["factor"] <= a0["factor"] + 1e-12) and a4["summary"]["vetoed"] >= a0["summary"]["vetoed"],
          "(vetoed %d at 0 mm, %d at 4 mm)" % (a0["summary"]["vetoed"], a4["summary"]["vetoed"]))

    # 14. when the conflict mask removes every candidate, say so instead of returning an empty ranking
    big = os.path.join(tmp, "everywhere.nii.gz")
    sphere_mask(big, [((0.0, 0.0, 0.0), 30.0)])  # a region around the target itself
    try:
        TC.compute_target_conflict(skin, target, [{"path": big, "name": "big", "margin": 3.0, "mode": "hard", "weight": 1.0}],
                                   cfg, valid_all, os.path.join(tmp, "wb"), PlanTUS, log=lambda *a: None)
        check("a mask that removes every position is reported", False)
    except RuntimeError as e:
        check("a mask that removes every position is reported", "none is left" in str(e))

    # 15. the same file given twice: the explicit entry wins; two different files with one name stay separate
    d1, d2 = os.path.join(tmp, "dirA"), os.path.join(tmp, "dirB")
    os.makedirs(d1), os.makedirs(d2)
    sphere_mask(os.path.join(d1, "roi.nii.gz"), [((0.0, 0.0, -12.0), 6.0)])
    sphere_mask(os.path.join(d2, "roi.nii.gz"), [((20.0, 0.0, -12.0), 3.0)])
    one = TC.prepare(t1, os.path.join(tmp, "dd1"), {}, [os.path.join(d1, "roi.nii.gz") + ",margin=7"], folder=d1)
    check("the same mask in the folder and on the command line counts once, with the explicit options",
          len(one["specs"]) == 1 and one["specs"][0]["margin"] == 7.0)
    two = TC.prepare(t1, os.path.join(tmp, "dd2"), {}, [os.path.join(d2, "roi.nii.gz")], folder=d1)
    a = nib.load(two["specs"][0]["path"]).get_fdata().sum()
    b = nib.load(two["specs"][1]["path"]).get_fdata().sum()
    check("two different files with the same name are both kept", len(two["specs"]) == 2 and a != b and
          two["specs"][0]["path"] != two["specs"][1]["path"] and two["specs"][0]["name"] != two["specs"][1]["name"],
          "(%s, voxels %d and %d)" % ([x["name"] for x in two["specs"]], a, b))

    # 16. config robustness
    sp = TC.collect_specs({"target_conflict_masks": [{"path": "rel/roi.nii.gz"}]}, [], base_dir=os.path.join(tmp, "cfgdir")) if False else None
    os.makedirs(os.path.join(tmp, "cfgdir", "rel"))
    sphere_mask(os.path.join(tmp, "cfgdir", "rel", "roi.nii.gz"), [((0.0, 0.0, 0.0), 5.0)])
    os.chdir(os.path.join(tmp, "dirA"))  # somewhere else, as the wrapper's chdir does
    sp = TC.collect_specs({"target_conflict_masks": [{"path": "rel/roi.nii.gz"}]}, [], base_dir=os.path.join(tmp, "cfgdir"))
    os.chdir(CODE)
    check("a relative path in the config is relative to the config's folder", sp[0]["path"] == os.path.join(tmp, "cfgdir", "rel", "roi.nii.gz"))
    for bad, why in (({"target_conflict_full_mm": 0}, "full_mm 0"), ({"target_conflict_full_mm": -5}, "negative full_mm"),
                     ({"weight_target_conflict": -1}, "negative weight"), ({"target_conflict_margin": -1}, "negative margin"),
                     ({"target_conflict_beyond": -2}, "negative beyond")):
        try:
            TC.check_settings(bad)
            check("rejects " + why, False)
        except ValueError:
            check("rejects " + why, True)
    sphere_mask(os.path.join(tmp, "LV.nii.gz"), [((0.0, 0.0, 0.0), 5.0)])
    sphere_mask(os.path.join(tmp, "lv.nii.gz"), [((9.0, 0.0, 0.0), 5.0)])
    both = TC.collect_specs({}, [os.path.join(tmp, "LV.nii.gz"), os.path.join(tmp, "lv.nii.gz")])
    check("names that differ only by case are made distinct", len({s["name"].lower() for s in both}) == 2, "(%s)" % [s["name"] for s in both])
    # the no-mesh fallback agrees with the mesh method
    pm = os.path.join(tmp, "fb.nii.gz")
    sphere_mask(pm, [((8.0, -6.0, -4.0), 9.0)])
    mk = TC.ConflictMask(pm)
    rg = mk.region(0.0)
    with_mesh = TC.beam_lengths(origins, target, np.full(len(origins), 22.0), mk, rg, os.path.join(tmp, "dilated_3Dmodel.stl"), PlanTUS)
    walked = TC.beam_lengths(origins, target, np.full(len(origins), 22.0), mk, rg, None, PlanTUS)
    check("the no-mesh fallback (walking the beam) agrees with the mesh method", np.abs(with_mesh - walked).mean() < 0.5,
          "(mean difference %.2f mm over %d beams)" % (np.abs(with_mesh - walked).mean(), len(origins)))

    # 17. target overlap is part of the staging signature (a change redoes finished runs); conflict-only signatures are unchanged
    ovm = os.path.join(tmp, "ovmask.nii.gz")
    sphere_mask(ovm, [((0.0, 0.0, -12.0), 6.0)])
    base_sig = TC.prepare(t1, os.path.join(tmp, "o1"), {}, [ovm + ",margin=5"])["signature"]
    ov_sig = TC.prepare(t1, os.path.join(tmp, "o2"), {}, [ovm + ",margin=5"], overlap=True)["signature"]
    ov_sig2 = TC.prepare(t1, os.path.join(tmp, "o3"), {}, [ovm + ",margin=5"], overlap=True, min_overlap=0.7)["signature"]
    only_ov = TC.prepare(t1, os.path.join(tmp, "o4"), {}, [], overlap=True)
    check("target overlap changes the signature, and so does its minimum", len({base_sig, ov_sig, ov_sig2}) == 3)
    check("overlap alone (no conflict masks) still gives a signature and settings", only_ov["signature"] != "none" and only_ov["settings"]["target_overlap"] is True and only_ov["settings"]["target_overlap_min"] == 0.5)

    n_fail = RESULTS.count(False)
    print("\n%d checks, %d failed" % (len(RESULTS), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
