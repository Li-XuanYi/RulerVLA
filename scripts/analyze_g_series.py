"""Episode-clustered analysis, seven figure pairs and explicit failure boundaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rulervla.evaluation.g_series import GROUPS, file_hash


def read_rows(path):
    with path.open() as f:
        result = []
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = json.loads(value) if value else None
                except (ValueError, TypeError):
                    parsed[key] = value
            result.append(parsed)
    return result


def metric(rows, key, design, *, accepted_only=False):
    eligible = [r for r in rows if r.get("eligible", True) and r.get(key) is not None
                and (not accepted_only or r["accepted"])]
    if not eligible:
        return dict(mean=None, ci95=[None, None], n_rows=0, n_episodes=0)
    episodes = sorted({r["episode_id"] for r in eligible})
    means = np.array([np.mean([float(r[key]) for r in eligible if r["episode_id"] == ep]) for ep in episodes])
    rng = np.random.default_rng(design["bootstrap_seed"])
    indices = rng.integers(0, len(means), size=(design["bootstrap_repetitions"], len(means)))
    boot = means[indices].mean(axis=1)
    return dict(mean=float(means.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                n_rows=len(eligible), n_episodes=len(episodes))


def summarize(rows, design):
    keys = ["accepted", "correction_accepted", "bias_error_mm", "estimated_bias_norm_mm",
            "scene_rms_before_mm", "scene_rms_after_mm", "scene_p95_before_mm", "scene_p95_after_mm",
            "near_rms_after_mm", "middle_rms_after_mm", "far_rms_after_mm", "target_rms_before_mm", "target_rms_after_mm",
            "anchor_residual_median_mm", "anchor_residual_p95_mm", "reliability_score", "scene_valid_coverage",
            "mean_patch_weight", "num_anchor_used", "anchor_spatial_spread_px", "measured_depth_missing_fraction",
            "silent_wrong_correction", "scene_harmed", "mean_bias_error_mm", "mean_scene_rms_after_mm"]
    out = dict(rows=len(rows), eligible_rows=sum(r.get("eligible", True) for r in rows),
               n_episodes=len({r["episode_id"] for r in rows}), metrics={k: metric(rows, k, design) for k in keys})
    out["accepted_bias_error_mm"] = metric(rows, "bias_error_mm", design, accepted_only=True)
    out["failure_reasons"] = {reason: sum(r["failure_reason"] == reason for r in rows)
                              for reason in sorted({r["failure_reason"] for r in rows if r["failure_reason"]})}
    out["conditions"] = {name: dict(rows=sum(r["condition"] == name for r in rows),
        metrics={k: metric([r for r in rows if r["condition"] == name], k, design) for k in keys},
        accepted_bias_error_mm=metric([r for r in rows if r["condition"] == name], "bias_error_mm", design, accepted_only=True))
        for name in dict.fromkeys(r["condition"] for r in rows)}
    return out


def fmt(value, digits=3):
    return "null" if value is None else f"{value:.{digits}f}"


def estimate(summary, key):
    m = summary["metrics"][key]
    return f"{fmt(m['mean'])} [{fmt(m['ci95'][0])}, {fmt(m['ci95'][1])}]"


def plot_line(ax, rows, xs, key, design, label, **kw):
    estimates = [metric([r for r in rows if r["severity"] == x], key, design) for x in xs]
    y = np.array([np.nan if m["mean"] is None else m["mean"] for m in estimates])
    low = np.array([np.nan if m["ci95"][0] is None else m["ci95"][0] for m in estimates])
    high = np.array([np.nan if m["ci95"][1] is None else m["ci95"][1] for m in estimates])
    line, = ax.plot(xs, y, "o-", label=label, **kw)
    ax.fill_between(xs, low, high, alpha=.15, color=line.get_color())


def figures(group, rows, summary, design, path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    path.mkdir(parents=True, exist_ok=True)
    if group == "G01":
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [0]+design["translation_mm"]
        plot_line(ax, rows, xs, "scene_rms_before_mm", design, "Before")
        plot_line(ax, rows, xs, "scene_rms_after_mm", design, "Translation correction")
        ax.set(xlabel="Shared translation magnitude (mm)", ylabel="Nonrobot scene RMS error (mm)")
        ax.legend()
    elif group == "G02":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        xs = sorted(design["anchor_counts"])
        plot_line(axes[0], rows, xs, "bias_error_mm", design, "All estimates; rejected = zero bias")
        accepted_rows = [r for r in rows if r["accepted"]]
        plot_line(axes[0], accepted_rows, xs, "bias_error_mm", design, "Accepted only")
        plot_line(axes[1], rows, xs, "accepted", design, "Effective acceptance")
        axes[0].set(xlabel="Retained distinct anchors", ylabel="Bias error (mm)")
        axes[1].set(xlabel="Retained distinct anchors", ylabel="Acceptance fraction", ylim=(-.05, 1.05))
        for ax in axes: ax.legend(fontsize=8)
    elif group == "G03":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        xs = design["outlier_counts"]
        for ax, robust, mean, label in zip(axes, ("bias_error_mm", "scene_rms_after_mm"),
                ("mean_bias_error_mm", "mean_scene_rms_after_mm"), ("Bias error (mm)", "Corrected scene RMS (mm)")):
            plot_line(ax, rows, xs, robust, design, "Huber")
            plot_line(ax, rows, xs, mean, design, "Plain mean")
            ax.set(xlabel="Wrong correspondences / 8 anchors", ylabel=label)
            ax.legend()
    elif group == "G04":
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, axis in zip(axes, design["rotation_axes"]):
            selected = [r for r in rows if r["condition"].startswith(axis+"_")]
            for band in ("near", "middle", "far"):
                plot_line(ax, selected, design["rotation_degrees"], band+"_rms_after_mm", design, band)
            ax.set(xlabel="Extrinsic rotation (degrees)", title="Optical camera "+axis)
            ax.legend()
        axes[0].set_ylabel("RMS after translation-only correction (mm)")
    elif group == "G05":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        names = list(summary["conditions"])
        x = np.arange(len(names))
        for ax, key, label in zip(axes, ("scene_rms_after_mm", "accepted", "scene_valid_coverage"),
                ("Scene RMS on retained support (mm)", "Acceptance fraction", "Nonrobot scene valid coverage")):
            values = [summary["conditions"][n]["metrics"][key]["mean"] for n in names]
            ax.bar(x, values)
            ax.set_xticks(x, [n.replace("missing_", "").replace("_", "\n") for n in names], fontsize=8)
            ax.set_ylabel(label)
        fig.suptitle("Missing depth is coverage loss, not accurate zero geometry")
    else:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        selected_key = "scene" if group == "G06" else "target"
        xs = design["local_depth_bias_mm"]
        for stage in ("before", "after"):
            plot_line(axes[0], rows, xs, selected_key+"_rms_"+stage+"_mm", design, stage)
        axes[0].set(xlabel="Local optical depth bias (mm)", ylabel=selected_key+" RMS error (mm)")
        axes[0].legend()
        with np.load(ROOT/"outputs/g_series"/(group+"_representative.npz"), allow_pickle=False) as z:
            axes[1].imshow(z["rgb"])
            roi = z["anchor_roi"] if group == "G06" else z["target"]
            axes[1].imshow(np.ma.masked_where(~roi, roi), cmap="autumn", alpha=.7, vmin=0, vmax=1)
            axes[1].set_title("First frozen frame: intervention ROI")
            mask = z["scene"] if group == "G06" else z["target"]
            maximum = max(1, float(np.max(z["after_error_mm"][mask])))
            for ax, key, title in zip(axes[2:], ("before_error_mm", "after_error_mm"), ("Before error", "After error")):
                im = ax.imshow(np.ma.masked_where(~mask, z[key]), vmin=0, vmax=maximum, cmap="magma")
                ax.set_title(title)
            fig.colorbar(im, ax=axes[2:], label="Metric XYZ error (mm)", shrink=.8)
        for ax in axes[1:]: ax.set_xticks([]); ax.set_yticks([])
    if group not in {"G06", "G07"}:
        fig.suptitle(group+": frozen 50 frames; CI resamples 10 episodes")
        fig.tight_layout()
    else:
        fig.subplots_adjust(top=.84, bottom=.2, wspace=.4)
        fig.suptitle(group+": anchor agreement is not full-scene correctness")
    for ext in ("png", "pdf"):
        fig.savefig(path/(group+"_boundary."+ext), dpi=160, bbox_inches="tight")
    plt.close(fig)


def conclusion(group, rows, summary, design):
    metrics = summary["metrics"]
    result = [f"- Effective acceptance: {metrics['accepted']['mean']*100:.1f}%.",
              f"- Scene RMS before/after (mm; episode cluster CI): {estimate(summary, 'scene_rms_before_mm')} / {estimate(summary, 'scene_rms_after_mm')}.",
              f"- Anchor post-fit median residual (mm): {estimate(summary, 'anchor_residual_median_mm')}.",
              f"- Reliability fit score: {estimate(summary, 'reliability_score')}."]
    if group == "G01":
        changed = [r for r in rows if r["severity"] > 0]
        accepted = [r for r in changed if r["accepted"]]
        maximum = max(r["bias_error_mm"] for r in accepted) if accepted else None
        result.append(f"- Maximum accepted bias recovery error over 2-20 mm: {fmt(maximum)} mm.")
        interpretation = "Shared translation is supported on this fixed GT-anchor set if accepted recovery remains within 2 mm; it does not test arbitrary calibration errors."
        boundary = "Clean-control fitting has a raster/mesh residual floor. The largest tested translation is 20 mm; no claim beyond it."
        decision = "continue: retain as the positive mechanism control"
    elif group == "G02":
        small = [r for r in rows if r["severity"] <= 2]
        wrong = sum(r["silent_wrong_correction"] for r in rows)
        accepted_count = sum(r["accepted"] for r in rows)
        result.append(f"- Accepted with <=2 anchors: {sum(r['accepted'] for r in small)}/{len(small)}; accepted recovery errors >2 mm: {wrong}/{accepted_count} accepted records ({wrong}/{len(rows)} total).")
        interpretation = "Insufficient-anchor rejection is supported, but accepted 3/4-anchor recovery can exceed the predeclared 2 mm accuracy limit despite a small fit residual. Consistency does not guarantee accuracy. These cases still improve the original 10 mm error; they are not necessarily scene-harming corrections."
        boundary = "Subset A/B/C are deterministic spatial selections, not an exhaustive occlusion distribution. Rejected bias error uses zero correction and must not be described as accepted recovery."
        decision = "modify hypothesis: retain the accepted low-count accuracy boundary" if wrong else "continue within tested scope"
    elif group == "G03":
        comparison = [dict(r, robust_minus_mean_mm=r["bias_error_mm"]-r["mean_bias_error_mm"]) for r in rows if r["severity"] > 0]
        effect = metric(comparison, "robust_minus_mean_mm", design)
        result.append(f"- Paired Huber minus mean bias error for 1-3 outliers: {fmt(effect['mean'])} mm, CI {effect['ci95']}.")
        interpretation = "A negative paired difference supports robust-location resistance to these fixed correspondence errors. Plain mean remains an analytical baseline without acceptance gating."
        boundary = "Only up to 3/8 outliers and the preselected +x/-y/+z 20 mm direction set were tested; coordinated adversarial errors are not covered."
        decision = "continue within the tested outlier regime"
    elif group == "G04":
        result.append(f"- Far scene RMS after: {estimate(summary, 'far_rms_after_mm')} mm; near: {estimate(summary, 'near_rms_after_mm')} mm.")
        interpretation = "A locally fitted translation cannot eliminate rotation-induced position-dependent error. Low robot-cluster residual can coexist with large distant-scene error."
        boundary = "Rotation is not repaired. Acceptance of rotation conditions is a reliability scope limitation, not proof of successful correction."
        decision = "modify hypothesis: reliability describes local reference consistency"
    elif group == "G05":
        result.append(f"- Retained nonrobot scene coverage: {estimate(summary, 'scene_valid_coverage')}; mean patch weight: {estimate(summary, 'mean_patch_weight')}.")
        interpretation = "Report noise-induced error, removed coverage and rejection separately. Missing depth can make retained-point RMS small while usable geometry decreases."
        boundary = "No denoising or depth inpainting is implemented. Global fit_score may stay high under random missing while patch weights decrease."
        decision = "continue only with measured validity and coverage recorded"
    elif group == "G06":
        result.append(f"- Accepted scene-harming corrections: {sum(r['accepted'] and r['scene_harmed'] for r in rows)}/{len(rows)}.")
        interpretation = "Consistent robot-local depth bias is confounded with shared reference translation. Its agreement can cause a nonzero global correction that moves a previously clean nonrobot scene."
        boundary = "Identifiability limitation: anchor-only evidence cannot establish that the measured bias is shared by the whole scene. Do not patch it using GT corruption labels."
        decision = "modify hypothesis: require a shared-error assumption or independent scene evidence"
    else:
        result.append(f"- Target ROI RMS before/after: {estimate(summary, 'target_rms_before_mm')} / {estimate(summary, 'target_rms_after_mm')} mm; eligible records {summary['eligible_rows']}/{len(rows)}.")
        interpretation = "Correct robot anchors cannot diagnose a target-local depth bias. Unchanged anchor fit is evidence about the reference only, not the target's metric correctness."
        boundary = "Target-only depth error remains undetected by this reliability mechanism; invisible-target rows have null target metrics and are not substituted."
        decision = "modify hypothesis: distinguish reference consistency from full-scene correctness"
    setup = "Use the locked conditions in config.yaml and the same 50 base frames. K/T, EE pose, RGB, wrist, instruction and production thresholds remain fixed except the specified intervention."
    return (f"# {group} Conclusion\n\n## Question\n\n{GROUPS[group][1]}\n\n## Setup\n\n{setup}\n\n## Result\n\n"+
            "\n".join(result)+f"\n\n## Interpretation\n\n{interpretation}\n\n## Failure Boundary\n\n{boundary}\n\n## Cannot Conclude\n\n"
            "No policy success-rate improvement, learned anchor-detector performance, full SE(3) recovery, unseen tasks or real-world calibration guarantee follows from this geometry-only study.\n\n"
            f"## Decision\n\n{decision}.\n")


def main():
    design = yaml.safe_load((ROOT/"configs/g_series.yaml").read_text())
    lock = json.loads((ROOT/"experiments/g_series_protocol_lock.json").read_text())
    manifest = json.loads((ROOT/"experiments/base_frame_manifest.json").read_text())
    if file_hash(ROOT/"experiments/base_frame_manifest.json") != lock["base_manifest_sha256"]:
        raise AssertionError("Base manifest changed")
    from g_series_study import verify_lock
    verify_lock()
    all_rows, summaries = {}, {}
    expected_pairs = {(r["episode_id"], r["frame_id"]) for r in manifest["frames"]}
    for group, (directory, _) in GROUPS.items():
        path = ROOT/"experiments"/directory
        rows = read_rows(path/"raw/results.csv")
        group_conditions = [x["condition"] for x in lock["conditions"] if x["experiment_id"] == group]
        if len(rows) != len(group_conditions)*50:
            raise AssertionError("Incomplete rows: "+group)
        if len({(r["episode_id"], r["frame_id"], r["condition"]) for r in rows}) != len(rows):
            raise AssertionError("Duplicate condition rows: "+group)
        for name in group_conditions:
            if {(r["episode_id"], r["frame_id"]) for r in rows if r["condition"] == name} != expected_pairs:
                raise AssertionError("Unpaired base set")
        if not all(r["base_manifest_sha256"] == lock["base_manifest_sha256"] for r in rows):
            raise AssertionError("Mixed manifests")
        summary = summarize(rows, design)
        summary.update(base_manifest_sha256=lock["base_manifest_sha256"], protocol_sha256=file_hash(ROOT/"experiments/g_series_protocol_lock.json"),
                       csv_sha256=file_hash(path/"raw/results.csv"), analysis="episode-cluster bootstrap; 2000 draws; deterministic seed; descriptive CI")
        (path/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        (path/"conclusion.md").write_text(conclusion(group, rows, summary, design))
        figures(group, rows, summary, design, path/"figures")
        all_rows[group], summaries[group] = rows, summary
    # Paired critical scope checks relative to the clean control.
    clean = {(r["episode_id"], r["frame_id"]): r for r in all_rows["G01"] if r["severity"] == 0}
    g7_score_change = max(abs(r["reliability_score"]-clean[(r["episode_id"], r["frame_id"])]["reliability_score"]) for r in all_rows["G07"])
    g1 = [r for r in all_rows["G01"] if r["severity"] > 0]
    g2small = [r for r in all_rows["G02"] if r["severity"] <= 2]
    g3 = [dict(r, paired_error_mm=r["bias_error_mm"]-r["mean_bias_error_mm"]) for r in all_rows["G03"] if r["severity"] > 0]
    effect = metric(g3, "paired_error_mm", design)
    gates = dict(g1_recovery=all(r["accepted"] and r["bias_error_mm"] <= design["g2_bias_error_limit_mm"] for r in g1),
                 g2_insufficiency=not any(r["accepted"] for r in g2small),
                 g2_silent_wrong=sum(r["silent_wrong_correction"] for r in all_rows["G02"]),
                 g3_reasonable=(effect["mean"] is not None and effect["mean"] < 0),
                 complete_frozen_matrix=sum(len(rows) for rows in all_rows.values()) == lock["expected_rows"],
                 g4_g6_g7_boundaries_recorded=True)
    decision = "CONDITIONAL GO" if gates["g1_recovery"] and gates["g2_insufficiency"] and gates["g2_silent_wrong"] == 0 and gates["g3_reasonable"] else "NO-GO"
    combined = dict(schema="rulervla_g_series_summary_v1", rows_total=sum(len(rows) for rows in all_rows.values()),
                    base_manifest_sha256=lock["base_manifest_sha256"], protocol_sha256=file_hash(ROOT/"experiments/g_series_protocol_lock.json"),
                    groups=summaries, paired_g3_robust_minus_mean_bias_error_mm=effect,
                    g7_max_reliability_score_change_vs_clean=g7_score_change, scientific_gates=gates,
                    decision=decision, software_and_git_gates="Verified separately in reports/G_SERIES_VALIDATION.json", policy_training_started=False)
    reports = ROOT/"reports"
    reports.mkdir(exist_ok=True)
    (reports/"G_SERIES_SUMMARY.json").write_text(json.dumps(combined, indent=2, allow_nan=False)+"\n")
    g6 = summaries["G06"]["metrics"]
    g7 = summaries["G07"]["metrics"]
    g4 = summaries["G04"]["metrics"]
    report = f"""# G-Series Geometry Boundary Study

## Q1 — Which errors can V0.5 reliably repair?

Shared world translation at 2/5/10/20 mm along +x/+y/+z and the fixed mixed direction:
{sum(r['accepted'] for r in g1)}/{len(g1)} accepted, maximum accepted recovery error
{fmt(max(r['bias_error_mm'] for r in g1 if r['accepted']))} mm. All conditions use the
same frozen ten episodes / fifty frames. Across 1-3 of eight 20 mm correspondence
outliers, paired Huber-minus-mean bias error is {fmt(effect['mean'])} mm, episode
cluster CI [{fmt(effect['ci95'][0])}, {fmt(effect['ci95'][1])}].

## Q2 — Which unrepaired errors can reliability detect?

With one or two retained anchors, {sum(r['accepted'] for r in g2small)}/{len(g2small)}
conditions accept; insufficient evidence falls back to zero correction.
However, G2 contains {gates['g2_silent_wrong']} accepted 3/4-anchor recoveries above
the fixed 2 mm accuracy limit. Their small fit residual and high score do not
certify recovery accuracy. They improve the original 10 mm error but fail the
predeclared accuracy gate, rather than necessarily worsening the scene.
Depth-noise and missing-depth acceptance/coverage are reported condition by
condition in G05. Anchor-centred block missing can remove reference evidence;
random missing can reduce patch weights without changing the global fit score.
This is validity/consistency detection, not depth reconstruction. Missing pixels
are excluded from retained-support RMS and their lost coverage is reported.

## Q3 — Which errors cannot be repaired or detected?

- Extrinsic rotation: acceptance {g4['accepted']['mean']*100:.1f}%; after-error
  near {fmt(g4['near_rms_after_mm']['mean'])} mm / far {fmt(g4['far_rms_after_mm']['mean'])} mm.
  A robot-local translation fit leaves position-dependent error elsewhere.
- Anchor-only depth bias: scene RMS before {fmt(g6['scene_rms_before_mm']['mean'])} mm /
  after {fmt(g6['scene_rms_after_mm']['mean'])} mm, acceptance {g6['accepted']['mean']*100:.1f}%.
  Consistent local bias can wrongly move a clean scene. This is an identifiability
  limit under the shared-error assumption, not a reason to add a GT-specific patch.
- Target-only depth bias: target RMS before {fmt(g7['target_rms_before_mm']['mean'])} mm /
  after {fmt(g7['target_rms_after_mm']['mean'])} mm; acceptance {g7['accepted']['mean']*100:.1f}%.
  Maximum paired reliability-score change versus clean is {g7_score_change:.12g}.
  Reference consistency does not certify full-scene metric correctness.

## Q4 — Is the mechanism ready for a policy-level pilot?

**{decision}**. This is a scope-limited research decision, not training authorization.
The analysis code committed before outcomes uses a strict zero-exceedance G2
gate at 2 mm; it remains unchanged. The observed G2 exceedances are not erased
by changing thresholds after reviewing outcomes. NO-GO blocks automatic pilot
progress under this protocol; it does not imply shared-translation correction
is ineffective.

1. Require the G1 recovery, G2 insufficiency and G3 robustness gates in the JSON summary.
2. Restrict any later pilot to a declared shared-translation regime with GT anchors;
   reliability must not be presented as a general geometry validator.
3. Preserve the G4/G6/G7 failures and add independent evidence before claiming
   recovery from rotation or local depth error.
4. Require complete frozen manifests, regression tests, committed results and a clean tree.
5. The user must separately approve training. No policy inference, rollout, detector
   training or policy training was performed in this study.

These confidence intervals resample ten episodes, preserving frames and subsets;
3,350 interventions and image pixels are not independent trajectories. The small
deterministic task-2 base set cannot establish policy SR, learned-detector quality,
general tasks, negative rotation angles or adversarial outlier distributions.

Base SHA-256: `{lock['base_manifest_sha256']}`.
Protocol SHA-256: `{file_hash(ROOT/'experiments/g_series_protocol_lock.json')}`.
"""
    (reports/"G_SERIES_SUMMARY.md").write_text(report)
    with (ROOT/"experiments/registry.csv").open("w", newline="") as f:
        fields = ["ID", "Question", "IndependentVariable", "Status", "MainMetric", "MainResult", "Conclusion", "Decision", "Commit"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, (_, question) in GROUPS.items():
            s = summaries[group]
            writer.writerow(dict(ID=group, Question=question, IndependentVariable="; ".join(s["conditions"]), Status="done",
                MainMetric="scene_rms_after_mm", MainResult=estimate(s, "scene_rms_after_mm"),
                Conclusion=("partial support; accepted low-count accuracy boundary" if group == "G02" and gates["g2_silent_wrong"]
                            else "supported within tested scope" if group in {"G01", "G02", "G03"} else "failure boundary documented"),
                Decision="continue" if group in {"G01", "G03", "G05"} else "modify hypothesis", Commit=lock["runner_commit"]))
    print(json.dumps({"analysis_complete": True, "rows": combined["rows_total"], "decision": decision,
                      "gates": gates, "g3_paired_effect": effect, "g7_reliability_change": g7_score_change}), flush=True)


if __name__ == "__main__":
    main()
