# docs/

Reference documents for the EWSHM 2026 competition.

## Files

| Document | What it covers |
|---|---|
| [Challenge Rules](20260127_rules_challenge_1_OSMOS.pdf) | Competition objectives, evaluation criteria, timeline, and contact information from OSMOS Group |
| [AQUINAS Dataset Handbook](Aquinas-Dataset-Handbook.pdf) | Full technical specification of the dataset: bridge description, sensor layout, file formats, naming conventions, and units |

## Dataset reference

The dataset itself ships with its own detailed README at
`AQUINAS_DATASET/README.md` -- see that file for sensor codes,
JSON schema, and channel layout.

## Deliverables (future)

The two-page methodology + results summary (due July 1, 2026)
will be added to this folder when written.

## Organizer Guidance Log

The official rules PDF and the AQUINAS handbook remain the primary
references. The organizer Q&A and shared helper script below are used
to clarify implementation choices where they add operational detail.

### April 2, 2026 email answers

| Topic | Organizer guidance | Affects | Reflected in |
|---|---|---|---|
| Trigger threshold | Triggering depends on strain range over a rolling 4-second window, not directly on vehicle load | docs only | This README and the preprocessing notebook narrative |
| Trigger duration | Each deck keeps a 5-second memory buffer, starts recording when triggered, and stops 5 seconds after the signal becomes quiet again | code now | Deck-specific event grouping and preserved raw durations in [preprocessing/README](../aquinas_toolkit/preprocessing/README.md) |

### April 9, 2026 meeting Q&A

| Topic | Organizer guidance | Affects | Reflected in |
|---|---|---|---|
| No labels | The challenge is unsupervised and teams must self-label or detect outliers/trends | docs only | [training/README](../aquinas_toolkit/training/README.md) and project methodology |
| Missing or uneven data | Teams may discard incomplete events if justified, but should keep the methodology honest and auditable | code now | Preprocess discard diagnostics and `event_manifest.csv` / `summary.json` outputs |
| Dataset coverage | Teams do not have to use every event if a subset is scientifically stronger | docs only | Methodology framing, not a runtime rule |
| Synchronization | The wired logger introduces millisecond offsets; use one sensor as reference and match others without interpolation | code now | Organizer-faithful `r_synchro` alignment with first-selected reference and two shrinking passes |
| Zeroing | The shared R script uses per-sensor endpoint-line subtraction before synchronization | code now | Pre-alignment `linear_endpoints` zeroing in preprocessing |
| Temperature | Fiber-optic strain sensors are not temperature-compensated; temperature mostly affects the slow baseline | code later | Temperature metadata is preserved now; normalization is deferred |
| OMA on short records | Short baseline-corrected records can be concatenated for later OMA experiments | later | Forward note in [training/README](../aquinas_toolkit/training/README.md) |

### April 9, 2026 `AQUINAS_Explorer.R`

| Topic | Organizer guidance | Affects | Reflected in |
|---|---|---|---|
| Event lookup | Query events by timestamp within a selected deck/sensor subset | code now | Strict-containment `find_events()` plus organizer-style `load_timestamp_query_frames()` / `load_event_group()` |
| Alignment concept | Build a shared timestamp grid from one reference signal | code now | Organizer `Synchro()` alignment with first-selected reference and two shrinking passes |
| CSV export | Export aligned sensor tables for downstream analysis | code now | Preprocess stage artifacts and `export_aligned_event()` |

### April 9, 2026 organizer follow-up email

| Topic | Organizer guidance | Affects | Reflected in |
|---|---|---|---|
| Damaged sensor | One sensor was damaged between SET3 and SET4 and should be discarded for SET4 and SET5 while kept for SET1-SET3 | code now | Config-driven exclusion policy and preprocess QC report |
| Local evidence | `OLD_S1_UP_SUP_STR` matches the warning: the TABLE `Range` field collapses to `0.0` in SET4 and SET5 while raw slices still vary and the baseline jumps to about `30` | code now | `sensor_qc_report.csv` and [preprocessing/README](../aquinas_toolkit/preprocessing/README.md) |

Implementation summary from that email:

- the repository does not treat the sensor as globally bad; it is kept
  for SET1-SET3 and excluded only for SET4-SET5
- the exclusion is declared in `configs/default.yaml`, not hardcoded in
  feature logic
- preprocessing applies the exclusion before alignment, so the damaged
  channel cannot become the reference sensor and does not appear in the
  aligned exports for the affected SETs
- the `Range = 0` anomaly is a TABLE metadata issue, not proof that the
  raw waveform file is flat; the concern is the inconsistency between
  the raw waveform and its summary metadata, plus the large baseline
  shift
- `sensor_qc_report.csv` exists specifically so the team can audit why
  this override is present

### Follow-up organizer answers, source date pending

These points are treated as confirmed guidance but should be updated
with the exact source date when available.

| Topic | Organizer guidance | Affects | Reflected in |
|---|---|---|---|
| Sensor technology | The strain channels use intensity-modulated fiber-optic "optical strand" sensors, not electrical strain gauges | code now | Terminology in preprocessing docs and notebooks |
| OMA reference | The organizer cited EVACES 2025 work showing that concatenated baseline-corrected short records can reproduce long-record OMA results | later | [training/README](../aquinas_toolkit/training/README.md) |
| Expected frequencies | Typical bridge frequencies for this structure are around 2-10 Hz | rationale only | Justifies simple non-interpolating synchronization for v1 |
| Geometry clarification | The cited 4 m spacing and 20-40 cm thickness refer to the two main concrete beams/webs rather than the road slab | docs only | Presentation and structural interpretation, not preprocessing code |

## Code Traceability

- The config glossary for active v1 settings lives in
  [configs/README](../configs/README.md).
- The implementation-facing ledger for these decisions lives in
  [preprocessing/README](../aquinas_toolkit/preprocessing/README.md).
- That file records which organizer notes were implemented immediately,
  which defaults they changed, how the Python behavior differs from
  `AQUINAS_Explorer.R`, and which items remain explicit TODOs.
