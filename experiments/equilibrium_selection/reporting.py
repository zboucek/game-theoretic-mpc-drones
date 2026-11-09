# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Independent re-evaluation of controlled-study records (article V-A).

audit checks dynamics, objectives, KKT acceptance and unilateral witnesses.
It is called on a disposable data copy by tools/results.py."""
import json
import numpy as np
from .archive import digest, write_json
from .model import NumpyModel, step

def audit(archive):
    model = NumpyModel(archive.config)
    failures, checked = [], 0
    for record in archive.records():
        if record["kind"] == "deviation":
            if not np.isfinite(np.asarray(record["own_controls"], dtype=float)).all():
                if record["feasible"] or record["refutes_gne"]:
                    failures.append({"record_id": record["record_id"], "error": "Nonfinite deviation marked feasible"})
                checked += 1
                continue
            parent = archive.get(record["candidate_id"])
            controls = np.asarray(parent["controls"]).reshape(3, -1).copy()
            controls[record["player"]] = record["own_controls"]
            data = model.evaluate(controls.ravel(), parent["states"], parent["targets"])
            feasible = bool(np.isfinite(controls).all() and np.isfinite(data["g"]).all()
                and max(data["g"]) <= archive.config.feasibility_tolerance
                and np.max(np.abs(controls)) <= archive.config.bound+archive.config.feasibility_tolerance)
            baseline = model.evaluate(parent["controls"], parent["states"], parent["targets"])["costs"][record["player"]]
            improvement = baseline-data["costs"][record["player"]]
            refutes = feasible and improvement > max(1e-5, 1e-6*abs(baseline))
            if bool(record["refutes_gne"]) != bool(refutes) or record["feasible"] != feasible:
                failures.append({"record_id": record["record_id"], "error": "deviation feasibility/improvement mismatch"})
            checked += 1
            continue
        try:
            finite = np.isfinite(np.asarray(record["controls"], dtype=float)).all()
            if record["multipliers"] is not None:
                finite &= np.isfinite(np.asarray(record["multipliers"], dtype=float)).all()
            if not finite:
                if record["validation"]["accepted"]:
                    raise ValueError("Nonfinite solver output marked accepted")
                checked += 1
                continue
            quality, data, _ = model.validate(record["controls"], record["states"], record["targets"],
                record["multipliers"], variational=record["method"] != "gne")
            accepted = quality["accepted"]
            if accepted != record["validation"]["accepted"]:
                raise ValueError("Independent acceptance mismatch")
            for name, field in (("player_costs", "costs"), ("prediction", "prediction"),
                                ("potential", "potential"), ("shared_inequalities", "g"),
                                ("stage_costs", "stage_costs"), ("player_effort", "effort")):
                np.testing.assert_allclose(record[name], data[field], atol=1e-9, rtol=1e-10)
            if record["solver_reported_kkt_norm"] is not None:
                np.testing.assert_allclose(record["solver_reported_kkt_norm"], quality["kkt_fb_norm"], atol=1e-10, rtol=1e-7)
            if any(n != 1 for n in record.get("compiled_cache_entries", {}).values()):
                raise ValueError("Unexpected graph recompilation during measurement")
        except Exception as error:
            # Non-finite or unvalidatable rejected solver outputs remain valid failure records.
            if record["validation"]["accepted"] or not record.get("validation_error"):
                failures.append({"record_id": record["record_id"], "error": str(error)})
        checked += 1
    for relative, sha in archive.manifest.get("before_revision_sha256", {}).items():
        if digest(archive.directory / relative) != sha:
            failures.append({"file": relative, "error": "Pre-revision artifact changed"})
    for phase in ("pilot", "confirm"):
        for path in (archive.directory / "episodes" / phase).glob("*.json"):
            episode = json.loads(path.read_text())
            applied = np.asarray(episode["applied_states"])
            records = [archive.get(key) for key in episode["attempt_ids"]]
            accepted = [r for r in records if r["validation"]["accepted"]]
            try:
                if len(applied) != len(accepted)+1:
                    raise ValueError("Applied-node count mismatch")
                for k, record in enumerate(accepted):
                    np.testing.assert_allclose(record["states"], applied[k], atol=1e-12, rtol=0)
                    controls = np.asarray(record["controls"]).reshape(3, -1, 3)
                    np.testing.assert_allclose(step(applied[k], controls[:, 0]), applied[k+1], atol=1e-12, rtol=0)
            except Exception as error:
                failures.append({"file": str(path.relative_to(archive.directory)), "error": str(error)})
    result = {"passed": not failures, "checked_attempts": checked, "failures": failures}
    write_json(archive.directory / "independent_validation.json", result)
    return result
