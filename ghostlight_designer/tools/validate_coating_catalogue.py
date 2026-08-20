"""Validate the bundled coating catalogue.

Checks that every bundled envelope under
``ghostlight_designer/resources/coatings`` parses into CatalogueCoating entries
and that every preset payload actually round-trips through the ghostlight
loader — by writing a one-surface lens carrying the coating modifier, loading
it, and confirming the coating model landed as expected.

Run from the repo with the ghostlight binding + designer on PYTHONPATH:

    python ghostlight_designer/tools/validate_coating_catalogue.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import ghostlight

from ghostlight_designer.coating_catalogue import CoatingCatalogue


# A minimal V2 lens carrying one coating modifier on the front surface. The
# glass catalogue + geometry are fixed; only the modifier varies per preset.
def _lens_doc(modifier: dict) -> dict:
    return {
        "format": "ghostlight-optical",
        "version": ghostlight.lens_format_version(),
        "glass_catalogue": {
            "N-BK7": {
                "name": "N-BK7",
                "dispersion": {
                    "model": "sellmeier",
                    "B": [1.03961212, 0.23179234, 1.01046945],
                    "C": [0.00600069867, 0.02001791440, 103.560653],
                },
            }
        },
        "optical_system": [
            {
                "type": "element",
                "name": "singlet",
                "transform": {"position": {"mode": "absolute", "x": 0, "y": 0, "z": 0}},
                "surfaces": [
                    {
                        "semi_aperture": 15.0,
                        "thickness": 5.0,
                        "form": {"type": "sphere", "radius": 50.0},
                        "modifiers": [modifier],
                    },
                    {
                        "semi_aperture": 15.0,
                        "thickness": 40.0,
                        "form": {"type": "sphere", "radius": -50.0},
                    },
                ],
                "materials": [{"glass": "N-BK7"}],
            }
        ],
        "pivots": [],
    }


# Expected runtime CoatingModel for each JSON coating discriminator.
def _expected_model(modifier: dict) -> int:
    if "layers" in modifier:
        return int(ghostlight.CoatingModel.SPECTRAL_ANGULAR)
    m = modifier.get("model")
    return {
        "simple": int(ghostlight.CoatingModel.SIMPLE),
        "artist": int(ghostlight.CoatingModel.ARTIST),
        "spectral": int(ghostlight.CoatingModel.SPECTRAL),
        "angular": int(ghostlight.CoatingModel.ANGULAR),
        "spectral_angular": int(ghostlight.CoatingModel.SPECTRAL_ANGULAR),
        "attenuator_gaussian": int(ghostlight.CoatingModel.ATTENUATOR_GAUSS),
    }.get(m, int(ghostlight.CoatingModel.SIMPLE))


def main() -> int:
    cat = CoatingCatalogue.load_bundled()
    if not len(cat):
        print("FAIL: no bundled coating presets found", file=sys.stderr)
        return 1

    failures = 0
    with tempfile.TemporaryDirectory() as d:
        for preset in cat.all():
            path = Path(d) / f"{preset.key}.lens"
            path.write_text(json.dumps(_lens_doc(preset.payload)), encoding="utf-8")
            try:
                sysm = ghostlight.OpticalSystem.load(str(path))
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL [{preset.key}]: load raised {exc}", file=sys.stderr)
                failures += 1
                continue

            got = int(sysm.surfaces[0].coating.model)
            want = _expected_model(preset.payload)
            # ar_layers=0 ("uncoated") legitimately stays SIMPLE with no
            # modifier round-tripped, so accept SIMPLE for it.
            if got != want:
                print(f"FAIL [{preset.key}]: coating model {got} != expected {want}",
                      file=sys.stderr)
                failures += 1
            else:
                print(f"ok   [{preset.key}]: {preset.display_name}")

    if failures:
        print(f"\n{failures} preset(s) failed validation", file=sys.stderr)
        return 1
    print(f"\nAll {len(cat)} coating presets validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
