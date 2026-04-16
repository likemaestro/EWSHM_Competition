"""Move flat preprocess waveform artifacts into per-SET subfolders."""

from __future__ import annotations

import argparse

from aquinas_toolkit.preprocessing import migrate_preprocess_waveforms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move flat preprocess waveform artifacts into per-SET subfolders.",
    )
    parser.add_argument(
        "preprocess_stage_dir",
        help="Path to results/<run_id>/stages/preprocess or preprocess.sqlite",
    )
    args = parser.parse_args(argv)
    summary = migrate_preprocess_waveforms(args.preprocess_stage_dir)
    print(
        "Migration complete: "
        f"{summary['moved_events']} events moved, "
        f"{summary['already_migrated_events']} already migrated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
