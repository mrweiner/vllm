#!/usr/bin/env python3
"""repro_verdict.py — mechanical verdict computation for the repro A/B
matrix. Reads the frozen predictions (proxy/repro_predictions.json, the
freeze commit BEFORE the first leg), the orchestrator summary
(<run-dir>/ab-summary.json) and the per-leg verdict JSONs
(<run-dir>/legs/*.json), and emits <run-dir>/verdicts.json plus a stdout
report. No judgment happens here: every rule (signed variance, gate,
not-exercised, noise floor) is pinned in the predictions file; this
script implements exactly those rules and errors loudly if the
predictions file contains an item this script does not implement (and
vice versa) so the two cannot drift apart silently.

Gate enforcement: the gate needs the ORCHESTRATOR'S recorded verdict AND
the independently computed one, plus a calibration leg to exist at all —
a recorded true never overrides a failed computed gate (reviewer round
5, item 7). A failed gate issues NO item verdicts: every item reads
"gate-failed" and the script exits 2.

Round-5 registration: #901 and #96 are PRE-REGISTERED AT ZERO EFFECT for
the repro signature (their patches cannot touch the observables their
old credit rules predicted — see the predictions file for the mechanism
argument). "validated" is impossible for them by construction; an
observable that moves beyond variance is recorded honestly as
"unexpected-difference".

Missing data is never treated as zero: a required metric absent from any
leg aborts scoring with the leg and metric named.

Run: python3 proxy/repro_verdict.py --run-dir <dir>
Offline tests: timeout 60 python3 proxy/test_repro_verdict.py
"""
import argparse
import difflib
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS = os.path.join(HERE, "repro_predictions.json")

IMPLEMENTED_ITEMS = {"901", "902", "95", "96", "97",
                     "pin-gap", "queueing-check", "correctness"}

# Reviewer round 4: "One differing token is not an alarm; exact match is
# not required." A fixed-arm answer is a correctness violation only when
# it falls below BOTH the per-prompt reference floor AND this absolute
# agreement slack.
CORRECT_SLACK = 0.95

# Missing data must fail loudly, never read as zero (reviewer round 5,
# item 9): every leg in every state must carry these fields.
REQUIRED_METRICS = (
    "engine.refusals", "engine.restore_failures", "engine.recomputes",
    "engine.directory_misses", "engine.slow_restores", "engine.slow_copies",
    "metrics.preemptions_max", "summary.ttft_p50", "blocks.free_min",
    "blocks.unattributed_max", "blocks.pin_gap_max",
)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def _sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def variance_ok(base_vals, fixed_vals, direction=None):
    """The pinned rule (predictions.variance_rule), with the
    pre-registered direction applied: the median difference must be
    significant AND in the declared direction (down = fixed below base,
    up = fixed above base). |median diff| > 2 * pooled sd (ddof=1); if
    pooled sd == 0, the ranges must be disjoint in the declared
    direction: down means max(fixed) < min(base), up means
    min(fixed) > max(base) (reviewer round 5, item 6 — the branches
    were inverted and are regression-tested both ways). Returns
    (ok, evidence) or (None, reason) when n < 2 makes the rule
    undecidable. direction=None keeps the legacy two-sided rule."""
    if len(base_vals) < 2 or len(fixed_vals) < 2:
        return None, "n<2 in at least one arm; variance rule undecidable"
    mb, mf = _median(base_vals), _median(fixed_vals)
    pooled = (_sd(base_vals) ** 2 * (len(base_vals) - 1)
              + _sd(fixed_vals) ** 2 * (len(fixed_vals) - 1)) / (
        len(base_vals) + len(fixed_vals) - 2)
    sd = pooled ** 0.5
    ev = {"median_base": mb, "median_fixed": mf, "pooled_sd": sd,
          "direction": direction}
    if sd > 0:
        effect = mf - mb
        if direction == "down":
            ok = effect < 0 and -effect > 2 * sd
        elif direction == "up":
            ok = effect > 0 and effect > 2 * sd
        else:
            ok = abs(effect) > 2 * sd
        ev["effect"] = effect
        ev["threshold"] = 2 * sd
        return ok, ev
    ev["range_base"] = [min(base_vals), max(base_vals)]
    ev["range_fixed"] = [min(fixed_vals), max(fixed_vals)]
    if direction == "down":
        ok = max(fixed_vals) < min(base_vals)
    elif direction == "up":
        ok = min(fixed_vals) > max(base_vals)
    else:
        ok = (max(base_vals) < min(fixed_vals)
              or max(fixed_vals) < min(base_vals))
    return ok, ev


def _legs_by_state(summary):
    """Group leg verdicts by state, loading each leg's full JSON. The
    profile tag is mandatory: a leg without one cannot be assigned to an
    arm and would silently drop out of every comparison. The calibration
    flag rides along; the calibration leg is excluded from every arm and
    used for the gate only (reviewer round 5, item 8)."""
    groups = {}
    for leg in summary["legs"]:
        profile = leg.get("profile")
        if not profile:
            raise RuntimeError(
                f"leg '{leg.get('leg')}' has no profile field; refusing "
                "to score — a profile-less leg would silently drop out "
                "of every arm")
        path = leg.get("leg_path") or os.path.join(
            os.path.dirname(summary.get("_path", "")),
            "legs", f"{leg['leg']}.json")
        with open(path) as f:
            v = json.load(f)
        g = groups.setdefault(leg["state"], [])
        g.append({"leg": leg["leg"], "verdict": v, "profile": profile,
                  "calibration": bool(leg.get("calibration"))})
    return groups


def _get(v, path):
    x = v
    for key in path.split("."):
        x = (x or {}).get(key) if isinstance(x, dict) else None
    return x


def _vals(legs, path):
    out = []
    for leg in legs:
        x = _get(leg["verdict"], path)
        if x is not None:
            out.append(x)
    return out


def _require_metrics(groups):
    """Every leg in every state must carry every required metric; a
    missing one aborts scoring with the leg and metric named (a missing
    field must never read as zero — reviewer round 5, item 9). The
    block minima may legitimately be None (a leg with no accounting
    samples) but the keys themselves must exist."""
    optional_none = {"blocks.free_min", "blocks.unattributed_max"}
    missing = []
    for state, legs in groups.items():
        for leg in legs:
            for path in REQUIRED_METRICS:
                leaf = path.split(".")[-1]
                container = _get(leg["verdict"], path.rsplit(".", 1)[0])
                if not isinstance(container, dict) or leaf not in container:
                    missing.append(f"{leg['leg']}.{path}")
                elif (container[leaf] is None
                        and path not in optional_none):
                    missing.append(f"{leg['leg']}.{path}")
    if missing:
        raise RuntimeError(
            "required metrics missing from legs: " + "; ".join(missing)
            + " — absent data must abort scoring, never read as zero")


def _counter_fired(legs, counter):
    return sum((leg["verdict"].get("engine", {}).get("counters", {})
                .get(counter, 0) for leg in legs), 0) > 0


def _preemptions(legs):
    return _vals(legs, "metrics.preemptions_max")


def _zero_prediction_verdict(bvals, fvals, counter_fired):
    """Mechanical rule for a PRE-REGISTERED ZERO-EFFECT item. Priority:
    the fix's own path never running is not-exercised REGARDLESS of the
    observables (any difference then cannot be the patch's doing; the
    values are still recorded in the evidence). With the path exercised:
    validated is impossible; an observable moving beyond variance
    (either direction) is recorded honestly as unexpected-difference;
    no movement is no-measurable-effect; an undecidable variance rule is
    unable-to-discriminate."""
    ev = variance_ok(bvals, fvals)[1] if len(bvals) >= 2 and len(fvals) \
        >= 2 else {"reason": "n<2 in at least one arm"}
    if not counter_fired:
        return "not-exercised", ev
    ok, ev2 = variance_ok(bvals, fvals)
    if ok is None:
        return "unable-to-discriminate", ev2
    if ok:
        return "unexpected-difference", ev2
    return "no-measurable-effect", ev2


def score(summary, preds):
    groups = _legs_by_state(summary)
    _require_metrics(groups)
    cal = next((g for g in groups.get("base", []) if g["calibration"]),
               None)
    gate = preds["signature_gate"]
    out = {"_meta": {"states": {s: [l["leg"] for l in legs]
                                for s, legs in groups.items()}}}

    # ---- calibration gate: recorded AND computed, needs a cal leg ----
    cal_ref = _vals([cal], "engine.refusals") if cal else []
    cal_ttft = _vals([cal], "summary.ttft_p50") if cal else []
    computed = bool(cal_ref and cal_ref[0] >= gate["require_refusals_min"]
                    and cal_ttft
                    and cal_ttft[0] >= gate["require_ttft_p50_s"])
    recorded = (summary.get("gate") or {}).get("reproduced")
    gate_ok = bool(cal) and computed and recorded is True
    out["_gate"] = {"reproduced": gate_ok, "recorded": recorded,
                    "computed": computed,
                    "calibration_leg": cal["leg"] if cal else None,
                    "calibration": {"refusals": cal_ref[0] if cal_ref else 0,
                                    "ttft_p50": cal_ttft[0]
                                    if cal_ttft else None}}
    if not gate_ok:
        for key in IMPLEMENTED_ITEMS:
            out[key] = ("gate-failed",
                        "the calibration gate did not reproduce the "
                        "collapse; no item may be scored")
        return out

    def arm(state):
        # the calibration leg is gate-only: excluded from every arm
        return [g for g in groups.get(state, [])
                if g["profile"] == "matched" and not g["calibration"]]

    refs = [g for g in groups.get("off", [])
            if g["profile"] == "reference"]
    base, fixed, f901, off = (arm("base"), arm("fixed"),
                              arm("fixed901"), arm("off"))
    out["_meta"]["refs"] = [g["leg"] for g in refs]
    out["_meta"]["excluded_calibration"] = cal["leg"] if cal else None

    # ---- not-exercised gate, split by item --------------------------
    nx = {}
    if cal and _vals([cal], "engine.restore_failures")[0] == 0:
        nx["96"] = ("not-exercised",
                    "calibration base leg had zero instant restore "
                    "failures; the walk-down signature did not reproduce")
    slow_b = _vals(base, "engine.slow_restores")
    slow_f = _vals(fixed, "engine.slow_restores")
    if not any(slow_b + slow_f):
        nx["902"] = ("not-exercised",
                     "no blocking restores (>10 s) appeared in either arm")
    sc_b = _vals(base, "engine.slow_copies")
    sc_f = _vals(fixed, "engine.slow_copies")
    if not any(sc_b + sc_f):
        nx["97"] = ("not-exercised",
                    "no blocking copies appeared in either arm")
    pb, pf = _preemptions(base), _preemptions(fixed)
    if not any(pb + pf):
        nx["95"] = ("not-exercised",
                    "zero preemptions in base and fixed; the f5 flush "
                    "mechanism engages only on the preemption path "
                    "(pre-registered zero-effect prediction)")

    # ---- 901: admission guards, PRE-REGISTERED AT ZERO EFFECT -------
    # (round 5: the guards act on vLLM's allocation-failure break after
    # a 300 s age gate; the refusals are LMCache-side and happen before
    # that path — "refusals drop >=80%" has no mechanism. The old credit
    # rule is gone; the guards' observable effects (f12 peer
    # preemptions, f8 delayed-free reap) are recorded honestly.)
    fired901 = _counter_fired(fixed, "admission_blocks_fixed")
    z901 = _zero_prediction_verdict(_vals(base, "engine.refusals"),
                                    _vals(fixed, "engine.refusals"),
                                    fired901)
    out["901"] = (z901[0], {
        "refusals": z901[1],
        "preemptions": {"base": pb, "fixed": pf},
        "free_min": {"base": _vals(base, "blocks.free_min"),
                     "fixed": _vals(fixed, "blocks.free_min")},
        "admission_counter_fired": fired901,
        "registration": "zero-effect (round 5)",
    })

    # ---- 902 / 97: zero-effect predictions --------------------------
    out["902"] = nx.get("902") or _zero_prediction_verdict(
        slow_b, slow_f, _counter_fired(fixed, "copy_wait_deadline_hits_fixed"))
    out["97"] = nx.get("97") or _zero_prediction_verdict(
        sc_b, sc_f, _counter_fired(fixed, "layout_deadline_hits_fixed"))

    # ---- 95: preemption-path flush ----------------------------------
    if "95" in nx:
        out["95"] = nx["95"]
    else:
        rv95 = variance_ok(_vals(f901, "engine.refusals"),
                           _vals(fixed, "engine.refusals"), "down")
        fired95 = _counter_fired(fixed, "preempt_flush_fired_fixed")
        down95 = (rv95[0] is True and rv95[1]["median_fixed"]
                  <= 0.2 * rv95[1]["median_base"])
        if fired95 and down95:
            out["95"] = ("validated", rv95[1])
        elif not fired95:
            out["95"] = ("not-exercised", "f5 counter never fired")
        elif rv95[0] is False:
            out["95"] = ("no-measurable-effect", rv95[1])
        else:
            out["95"] = ("unable-to-discriminate", rv95[1])

    # ---- 96: PRE-REGISTERED AT ZERO EFFECT (round 5) ----------------
    # (round 5, item A1: PR #96's diff is the checkpoint-task reaper
    # plus a teardown-only close drain/force-exit; it cannot touch the
    # instant retrieve-lease-miss restore failures — the PR body itself
    # flags that variant as not addressed. The serving-time fire counter
    # is the reaper counter; the f6 close-path counter fires only at
    # engine shutdown and is kept as verbatim evidence only.)
    brf = _vals(base, "engine.restore_failures")
    frf = _vals(fixed, "engine.restore_failures")
    fired96 = (_counter_fired(fixed, "checkpoint_reaper_task_fired")
               or _counter_fired(fixed, "checkpoint_reaper_lookup_fired"))
    z96 = _zero_prediction_verdict(brf, frf, fired96)
    out["96"] = (z96[0], {
        "restore_failures": z96[1],
        "recomputes": {"base": _vals(base, "engine.recomputes"),
                       "fixed": _vals(fixed, "engine.recomputes")},
        "ack_reasons": _vals(fixed, "engine.ack_reasons"),
        "reaper_counter_fired": fired96,
        "registration": "zero-effect (round 5)",
    })

    # ---- pin-gap (mechanism check; thresholds pinned in preds) ------
    # Round-6 measure (reviewer round 6, item 1): the unattributed held
    # blocks are split by source per 5 s sample; the pre-registered pin
    # gap is restore reservations + store pins + the true remainder
    # (restoreres + storepin + other). Policy-retained bundles (the
    # request-boundary policy keeps published bundles on the GPU as a
    # matter of course) are EXCLUDED — counting them would fake a pin
    # gap through ordinary retention. The usage gauge is recorded but
    # NEVER a gap condition: at the fork base get_usage() =
    # 1 - free/(num_gpu_blocks-1), so the old "free <= 33 AND gauge <=
    # 85%" pair was unsatisfiable by construction and the measured
    # 45-83% vs 0-32 free gap is a sampling-time artifact (60 s metric
    # snapshots vs instant refusal lines), not hidden pins.
    thr = preds["pin_gap_thresholds"]

    def gap_samples(legs):
        gaps = 0
        for g in legs:
            for s in (g["verdict"].get("blocks", {}).get("samples")
                      or []):
                pg = s.get("pin_gap")
                if pg is not None and pg >= thr["gap_blocks_min"]:
                    gaps += 1
        return gaps

    bfree = _vals(base, "blocks.free_min")
    ffree = _vals(fixed, "blocks.free_min")
    bgap = _vals(base, "blocks.pin_gap_max")
    fgap = _vals(fixed, "blocks.pin_gap_max")
    bunattr = _vals(base, "blocks.unattributed_max")
    funattr = _vals(fixed, "blocks.unattributed_max")
    gap_b, gap_f = gap_samples(base), gap_samples(fixed)
    gap_present = gap_b > 0
    pv = variance_ok(bgap, fgap, "down") if bgap and fgap \
        else (None, {})
    out["pin-gap"] = {
        "thresholds": thr,
        "gap_samples_base": gap_b, "gap_samples_fixed": gap_f,
        "gap_present_in_base": gap_present,
        "pin_gap_max": {"base": bgap, "fixed": fgap},
        "unattributed_max": {"base": bunattr, "fixed": funattr},
        "base_free_min": bfree, "fixed_free_min": ffree,
        "fixed_pin_gap_dropping_beyond_variance": pv[0],
        "verdict": ("falsified" if not gap_present else
                    "mechanism-supported" if pv[0] is True else
                    "gap-present-but-fix-path-undecided")}

    # ---- queueing-check: off vs base + fixed-vs-off rule ------------
    qv = variance_ok(_vals(base, "summary.ttft_p50"),
                     _vals(off, "summary.ttft_p50"), "down")
    fvo = variance_ok(_vals(off, "summary.ttft_p50"),
                      _vals(fixed, "summary.ttft_p50"), "down")
    out["queueing-check"] = {
        "off_ttft_p50": _vals(off, "summary.ttft_p50"),
        "base_ttft_p50": _vals(base, "summary.ttft_p50"),
        "fixed_ttft_p50": _vals(fixed, "summary.ttft_p50"),
        "off_refusals": _vals(off, "engine.refusals"),
        "off_below_variance_of_base": qv[0],
        "fixed_better_than_off_beyond_variance": fvo[0],
        # pre-registered: fixed NOT beating off means the fixes add
        # complexity without improving on disabling the feature
        "fixed_not_better_than_off": fvo[0] is not True,
        "verdict": ("checkpoint-path-confirmed" if qv[0] is True else
                    "unable-to-discriminate" if qv[0] is None else
                    "off-not-clearly-better")}

    rule = preds["correctness_rule"]
    out["correctness"] = _correctness(refs, fixed, base, rule)
    return out


def _agree(a, b):
    """Divergence-robust agreement: SequenceMatcher ratio (position of
    the divergence does not drive it to zero, unlike a common-prefix
    measure)."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _correctness_rates(arm_legs, ref_ans, floors):
    """Rate-based scoring for one arm (reviewer round 5, item 10):
    violations = answers whose agreement is below the per-prompt floor
    vs EVERY reference (max of the per-reference agreements, not min —
    min flags about half of prompts under pure noise) AND below the
    absolute slack; split by whether the request had restore events."""
    restored_v = restored_n = nonrestored_v = nonrestored_n = 0
    hits, no_restore = [], []
    for leg in arm_legs:
        for req in leg["verdict"].get("requests", []):
            key = (req.get("session"), req.get("turn"))
            answers = ref_ans.get(key) or []
            floor = floors.get(f"{key[0]},{key[1]}")
            if floor is None:
                continue
            agr = max((_agree(req.get("answer") or "", a)
                       for a in answers), default=1.0)
            entry = {"leg": leg["leg"], "session": key[0], "turn": key[1],
                     "agreement": round(agr, 3), "floor": floor,
                     "events": req.get("restore_events", [])}
            if agr < floor and agr < CORRECT_SLACK:
                if req.get("restore_events"):
                    restored_v += 1
                    hits.append(entry)
                else:
                    nonrestored_v += 1
                    no_restore.append(entry)
            if req.get("restore_events"):
                restored_n += 1
            else:
                nonrestored_n += 1
    return {
        "restored_violations": restored_v, "restored_requests": restored_n,
        "restored_rate": round(restored_v / restored_n, 3)
        if restored_n else 0.0,
        "nonrestored_violations": nonrestored_v,
        "nonrestored_requests": nonrestored_n,
        "nonrestored_rate": round(nonrestored_v / nonrestored_n, 3)
        if nonrestored_n else 0.0,
        "violations": hits,
        "divergence_without_restore_events": no_restore,
    }


def _correctness(refs, fixed_legs, base_legs, rule):
    if len(refs) < 2:
        return {"verdict": "unable-to-discriminate",
                "reason": "fewer than 2 reference runs"}
    ref_ans = {}
    for r in refs:
        for req in r["verdict"].get("requests", []):
            key = (req.get("session"), req.get("turn"))
            ref_ans.setdefault(key, []).append(req.get("answer") or "")
    floors, uncalibrated = {}, []
    for key, answers in ref_ans.items():
        if len(answers) < 2:
            uncalibrated.append(key)
            continue
        floor = 1.0
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                floor = min(floor, _agree(answers[i], answers[j]))
        floors[f"{key[0]},{key[1]}"] = round(floor, 3)
    frates = _correctness_rates(fixed_legs, ref_ans, floors)
    brates = _correctness_rates(base_legs, ref_ans, floors)
    control = max(brates["restored_rate"], frates["nonrestored_rate"],
                  brates["nonrestored_rate"])
    if frates["restored_requests"] < rule["min_restored_requests"]:
        verdict = "unable-to-discriminate"
    elif (frates["restored_violations"] >= rule["min_restored_violations"]
            and frates["restored_rate"] >= rule["min_restored_rate"]
            and frates["restored_rate"] >= rule["rate_ratio"] * control):
        verdict = "correctness-defect"
    else:
        verdict = "clean"
    # Round 6, item 3: a separate BASE-arm verdict, restored vs
    # non-restored. The shared control rate answers "did the fixes
    # INTRODUCE a regression?" — a defect present in both arms matches
    # the control and reads clean. This verdict answers the different
    # question "are restores correct at all?" against the BASE arm's
    # own non-restored rate, so a defect present in both arms cannot
    # hide behind the control.
    if brates["restored_requests"] < rule["min_restored_requests"]:
        base_verdict = "unable-to-discriminate"
    elif (brates["restored_violations"] >= rule["min_restored_violations"]
            and brates["restored_rate"] >= rule["min_restored_rate"]
            and brates["restored_rate"]
            >= rule["rate_ratio"] * brates["nonrestored_rate"]):
        base_verdict = "restores-corrupt-answers-in-base"
    else:
        base_verdict = "no-base-restored-defect-detected"
    return {
        "metric": "difflib.SequenceMatcher ratio (divergence-robust)",
        "aggregation": "max agreement vs references (below-floor vs "
                       "every reference required)",
        "slack_below_floor_also_required": CORRECT_SLACK,
        "rule": rule,
        "control_rate": control,
        "floors": floors,
        "uncalibrated_prompts": [f"{k[0]},{k[1]}"
                                 for k in uncalibrated],
        "fixed": frates,
        # the BASE arm scored the same way: the control the rate rule
        # is judged against (reviewer round 5, item 10)
        "control": brates,
        "base_verdict": base_verdict,
        "verdict": verdict,
    }


def check(predictions):
    """Loud drift check: every predictions item must be implemented and
    vice versa."""
    preds_items = set(predictions["items"])
    if preds_items != IMPLEMENTED_ITEMS:
        raise RuntimeError(
            "predictions/script drift: predictions has "
            f"{sorted(preds_items)}, script implements "
            f"{sorted(IMPLEMENTED_ITEMS)}")
    return True


def check_predictions_hash(predictions_path, run_dir, sha=None):
    """The run's predictions.sha256 sidecar (written by repro_ab at
    startup, from the file hash recorded in the freeze commit) must
    match the predictions file being scored. Missing sidecar is a loud
    warning; a mismatch aborts — a different table must never be scored
    against this run's artifacts."""
    if sha is None:
        path = os.path.join(run_dir, "predictions.sha256")
        if not os.path.exists(path):
            print("WARNING: no predictions.sha256 sidecar in the run "
                  "dir; the scored table cannot be tied to the freeze "
                  "commit")
            return
        with open(path) as f:
            sha = f.read().strip()
    with open(predictions_path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != sha:
        raise RuntimeError(
            f"predictions hash mismatch: sidecar {sha} != file {got}; "
            "the predictions file changed after the freeze — refuse to "
            "score")
    print("predictions hash verified:", got)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--predictions", default=PREDICTIONS)
    args = p.parse_args(argv)
    with open(args.predictions) as f:
        preds = json.load(f)
    check(preds)
    try:
        check_predictions_hash(args.predictions, args.run_dir)
    except RuntimeError as e:
        print("ABORT:", e)
        return 3
    spath = os.path.join(args.run_dir, "ab-summary.json")
    with open(spath) as f:
        summary = json.load(f)
    summary["_path"] = spath
    try:
        out = score(summary, preds)
    except RuntimeError as e:
        print("ABORT:", e)
        return 3
    opath = os.path.join(args.run_dir, "verdicts.json")
    with open(opath, "w") as f:
        json.dump(out, f, indent=1)
    gate_ok = out["_gate"]["reproduced"]
    if not gate_ok:
        print("GATE FAILED: the calibration leg did not reproduce the "
              "collapse; no item verdicts issued")
    for key in sorted(IMPLEMENTED_ITEMS):
        v = out.get(key)
        print(key, "->", v if isinstance(v, str) else
              (v or {}).get("verdict", v))
    print("written:", opath)
    return 2 if not gate_ok else 0


if __name__ == "__main__":
    sys.exit(main())
