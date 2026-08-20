# Coating Catalogue

The coating catalogue is the designer-side library of reusable coating
**presets**. It mirrors the [material catalogue](ghostlight_designer/resources/materials/README.md) in
shape, but is deliberately smaller and hand-authored: coatings are an
artistic/optical choice, not a scraped vendor database.

Applying a preset **copies** its coating-modifier payload onto the selected
surface. There is no live link: a `.lens` file has no coating-catalogue
section by design, because coatings are stored inline on each surface and a
lens must stay self-contained. The catalogue is purely an authoring
convenience.

## Where it lives

- **Model:** [`ghostlight_designer/coating_catalogue.py`](ghostlight_designer/coating_catalogue.py)
  — `CatalogueCoating` (frozen dataclass), `CoatingCatalogue` (container with
  `by_key` / `search` / `all`), `get_coating_catalogue()` singleton.
- **Bundled data:** [`ghostlight_designer/resources/coatings/*.json`](ghostlight_designer/resources/coatings/)
  — one or more envelope files, loaded alphabetically; later entries win on
  key collision, so a user catalogue can override a bundled preset.
- **Validator:** [`tools/validate_coating_catalogue.py`](tools/validate_coating_catalogue.py)
  — checks every bundled envelope parses and every payload round-trips
  through the ghostlight parser (load a synthetic lens carrying the coating).

## Envelope format

```json
{
  "format": "ghostlight-coating-catalogue",
  "version": { "major": 1, "minor": 0 },
  "source": "Built-in",
  "coatings": [
    {
      "key": "artist_vintage_amber",
      "display_name": "Vintage Amber (artist)",
      "source_vendor": "Built-in",
      "tags": ["artist", "warm", "vintage"],
      "description": "Warm amber-tinted ghosts …",
      "payload": {
        "type": "coating",
        "model": "artist",
        "tint": [1.0, 0.6, 0.2],
        "strength": 0.06
      }
    }
  ]
}
```

- `key` — stable, unique. Used for override precedence and lookup.
- `payload` — the **only** field that touches a lens. It is exactly one entry
  of a surface's `modifiers` array, in the same shape the `.lens` writer emits
  and the C++ parser reads. Every model is supported: `simple`, `artist`,
  `spectral`, `angular`, `spectral_angular`, a bare `layers` stack (TMM), and
  `attenuator_gaussian`.

## The three coating tiers (recap)

| Tier | Models | Use |
|---|---|---|
| **Simple** | `simple` (`ar_layers`) | Quick AR-layer count; V1-compatible. |
| **Artist** | `artist` (tint + strength) | Art-direct ghost colour directly — pick a hue and a reflection strength; the tracer synthesizes a smooth spectral curve. |
| **Scientific** | `spectral`, `angular`, `spectral_angular`, `layers` (TMM) | Manufacturer data or physically-modelled thin-film stacks (Transfer Matrix Method, baked to a λ×angle table at load). |

`attenuator_gaussian` is a positional amplitude modifier (apodization / pupil
shaping), not a reflectance coating; it composes as bare-Fresnel reflectance
plus a Gaussian transmission profile.

## Adding a preset

1. Add an entry to a bundled envelope (or a user envelope) under `coatings`.
2. Run `python tools/validate_coating_catalogue.py` to confirm it parses and
   round-trips.
3. It appears in the designer's coating-row **Preset** picker automatically.

## User catalogue

`CoatingCatalogue.load_with_user(path)` merges a user envelope on top of the
bundled data (user wins on key collision), matching the material catalogue's
bundled-first/user-last contract.
