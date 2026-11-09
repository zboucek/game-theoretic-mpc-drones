# Changes

2026-11-09: exclude the four nominal pursuit–evasion weight-comparison profiles from the article audit dataset and compact inputs. Remove their comparison code and audit steps from the article software. Retain fixed objective weights, the flight evidence and the damping comparison.

2026-11-09: align the paper citation and dataset title with the final manuscript title, “A Game-Theoretic MPC Planner for Cooperative and Adversarial Drone Coordination”. Numerical inputs and study scope are unchanged.

1.0.0 article edition: five studies, explicit compact input selection, independent verification, portable numeric figure regeneration, complete TOML configurations, callable experiments, editable installation, provenance, examples and cooperative trajectory media. The full release remains a separate profile.

2026-11-06: remove the deferred automatic weight-calibration study from the article software, configurations, figures, compact inputs and separate audit dataset. Research archives and the full release profile retain that study.

Compact distribution: keep figure/table inputs in the software ZIP; move complete audit evidence into a separately checksummed Dataset archive prepared for Zenodo. `verify --audit-data` preserves the full audit, while rendering no longer requires thousands of solver records.

2026-11-06: include the matched pursuit–evasion damping ablation, its raw flight pair, independent verification and configurable `model.response_fractions`.
