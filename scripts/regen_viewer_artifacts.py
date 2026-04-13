"""One-off script: regenerate sensor_layout.json and bridge_geometry.json
in all existing result visualization folders, then copy updated viewer assets.

Run from the repo root:
    python scripts/regen_viewer_artifacts.py
"""

import json
import shutil
from pathlib import Path

from aquinas_toolkit.visualization.layout import build_bridge_geometry, build_sensor_layout

SRC_ASSETS = Path("aquinas_toolkit/visualization/viewer_assets")
VIEWER_FILES = ("index.html", "viewer.css", "viewer.js")


def main() -> None:
    result_dirs = sorted(Path("results").glob("*/visualization"))
    if not result_dirs:
        print("No result visualization folders found.")
        return

    for result_dir in result_dirs:
        sl_path = result_dir / "sensor_layout.json"
        if not sl_path.exists():
            print(f"  skip {result_dir} (no sensor_layout.json)")
            continue

        old_layout = json.loads(sl_path.read_text(encoding="utf-8"))
        sensor_names = [row["sensor_id"] for row in old_layout]

        new_layout = build_sensor_layout(sensor_names)
        new_geometry = build_bridge_geometry()

        sl_path.write_text(json.dumps(new_layout, indent=2), encoding="utf-8")
        (result_dir / "bridge_geometry.json").write_text(
            json.dumps(new_geometry, indent=2), encoding="utf-8"
        )

        for asset in VIEWER_FILES:
            shutil.copy(SRC_ASSETS / asset, result_dir / asset)

        print(f"  updated {result_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
