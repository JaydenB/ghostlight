# Ghostlight Lens Files

This directory holds optical-system prescriptions in the **Ghostlight Optical** file format. A lens file describes a complete optical system — an ordered chain of elements, the surfaces that bound each element, the glasses that fill the gaps, and an optional post-process rig of pivots that translate or rotate groups of elements.

- **Extension:** `.lens`
- **Encoding:** UTF-8 JSON
- **Schema:** [`schema/lens.schema.json`](schema/lens.schema.json) — machine-readable validation, JSON Schema Draft 2020-12. There is exactly one schema, and `ghostlight/bindings/python/tests/test_lens_schema.py` validates every `.lens` in the repo against it.

Lens files are tool-generated, not hand-authored. Every element, surface, and pivot carries a UUID so render-time decisions (ghost identification, render-profile bindings, pivot membership) can reference physical objects by stable IDs instead of fragile array indices. Identity really is by UUID: the writer reassembles the flat surface array back into elements through these ids, so a document that omits them would collapse several surfaces onto one. The loader synthesizes stable placeholders when they are missing, but the schema requires them.

---

## Library Layout

```text
lenses/
  DoubleGauss.lens          Compact worked examples, used throughout this guide
  Helios44.lens             and by the test suite
  spherical/                Spherical prime prescriptions
  anamorphic/               Anamorphic systems, by squeeze factor and patent family
  experiments/              Research reconstructions and design studies
  schema/lens.schema.json   The one schema every .lens is validated against
```

Prescriptions under `spherical/` and `anamorphic/` are reconstructed from
published patent data. They are not vendor prescriptions: clear apertures,
coatings, and final image distances are reconstruction choices where the source
does not publish them, and each file records that in its `metadata`. Files under
`experiments/` are original designs and are labelled as such.

This directory holds designs only. The compact specimen the format
examples below are drawn from, `example_doublet.lens`, is not a design --
it exists to exercise every structural feature of the format at once -- so
it lives with the parser fixtures in
`ghostlight/bindings/python/tests/fixtures/`.

---

## File Format

```json
{
  "format": "ghostlight-optical",
  "version": { "major": 1, "minor": 0 },
  "metadata":        { ... },
  "glass_catalogue": { ... },
  "optical_system":  [ ... ],
  "pivots":          [ ... ]
}
```

| Key | Required | Description |
|---|---|---|
Every section is always present, in this order — the canonical form is a single
shape, not a set of optional combinations.

| Key | Required | Description |
|---|---|---|
| `format`          | yes | The literal string `"ghostlight-optical"`. |
| `version`         | yes | `{major, minor}`. The loader accepts `major` 1 only and warns on a newer `minor`. Mirrors `LENS_FORMAT_MAJOR` / `LENS_FORMAT_MINOR` in `ghostlight/src/optical_system.h`; read it from Python as `ghostlight.lens_format_version()` rather than writing the literal. |
| `metadata`        | yes | Provenance. The loader reads `name` and an optional `focal_length_mm` hint; everything else is informational and preserved verbatim. May be `{}`. |
| `glass_catalogue` | yes | Inline glass definitions, referenced by name from `materials[]`. May be `{}`. |
| `optical_system`  | yes | Ordered source → image-plane array of optical elements. |
| `pivots`          | yes | Post-process rig: named translation/rotation offsets layered on top of the optical system. May be `[]`. |

### Coordinate Conventions

- **Units:** millimetres for all distances, degrees for all rotations.
- **Axis:** `+z` is along the optical axis from the source toward the image plane.
- **Sensor:** lives at world `z = 0`. The optical system is laid out at `z ≤ 0`.
- **Radius sign:** positive radius = centre of curvature behind the surface (standard Zemax / ISO 10110 convention).
- **Flat surface:** the `"sphere"` form with `radius: 0.0`. There is no separate `"plane"` type.

---

## Glass Catalogue

Glasses are stored inline so the file is fully self-contained and portable. In practice, glasses are picked from a standalone catalogue (Schott, Ohara, etc.) and the relevant entries are embedded at save time. `catalogue_ref` records that provenance — it is not a live file reference.

```json
"glass_catalogue": {
  "N-BK7": {
    "name": "N-BK7",
    "catalogue_ref": "Schott:N-BK7",
    "dispersion": {
      "model": "sellmeier",
      "B": [1.03961212, 0.23179234, 1.01046945],
      "C": [0.00600069867, 0.02001791440, 103.560653]
    }
  },
  "custom_crown_1": {
    "name": "Custom Crown",
    "dispersion": {
      "model": "abbe",
      "nd": 1.5168,
      "Vd": 64.17
    }
  }
}
```

### Dispersion Models

| Model | Parameters | Notes |
|---|---|---|
| `"sellmeier"` | `B[3]`, `C[3]` | Industry standard. `n²(λ) = 1 + Σ Bᵢλ²/(λ²−Cᵢ)`, with λ in micrometres and `C` in µm². |
| `"abbe"`      | `nd`, `Vd`     | Two-parameter Cauchy approximation. Useful for artist-defined custom glasses and quick design work where Sellmeier coefficients are unknown. |

**Air is implicit** — `n = 1`, no dispersion. The literal strings `"air"` and `"AIR"` are reserved material names and resolve to air without a catalogue entry.

When a Sellmeier glass is loaded, the renderer also derives equivalent `n_d` and `V_d` values from the coefficients at the F (486.13 nm), d (587.56 nm) and C (656.27 nm) reference wavelengths, so every glass has a usable Cauchy approximation.

---

## Optical System

`optical_system` is an ordered array of **elements**, sorted source → image plane. Every item has `type: "element"`:

```json
"optical_system": [
  { "type": "element", ... },
  { "type": "element", ... },
  { "type": "element", ... }
]
```

Array order is authoritative. The loader walks elements in order when flattening to a sequential surface list.

### Elements

An element groups N surfaces sharing one rigid-body transform. It is the unit of geometry that the artist moves and tilts.

- **N surfaces + N−1 materials** model a piece of glass — singlet, cemented doublet, cemented triplet, and so on. `materials[i]` is the glass filling the gap between `surfaces[i]` and `surfaces[i+1]`. The cement interface in a doublet or triplet is just an ordinary surface — no special marker is needed.
- **1 surface + 0 materials** model a stop, mirror, or other interface in free air.

```json
{
  "type": "element",
  "id":   "e5f6a7b8-1234-5678-9abc-def012345678",
  "name": "Front Doublet",
  "transform": {
    "position": { "mode": "absolute", "x": 0.0, "y": 0.0, "z": 0.0 },
    "rotation": { "tilt_x": 0.0, "tilt_y": 0.0, "roll": 0.0 }
  },
  "surfaces":  [ ... ],
  "materials": [ ... ]
}
```

| Element kind          | Materials | Surfaces |
|---|:---:|:---:|
| Singlet               | 1 | 2 |
| Cemented doublet      | 2 | 3 |
| Cemented triplet      | 3 | 4 |
| Stop / mirror / iface | 0 | 1 |

The loader enforces `surfaces.length == materials.length + 1`.

Air gaps **between** elements are derived from the difference between element `z` positions and the cumulative axial thickness of surfaces within each element. They are not stored.

### Transforms

```json
"transform": {
  "position": { "mode": "absolute", "x": 0.0, "y": 0.0, "z": 0.0 },
  "rotation": { "tilt_x": 0.0, "tilt_y": 0.0, "roll": 0.0 },
  "pivot":    { "x": 0.0, "y": 0.0, "z": 0.0 }
}
```

Positions are in mm, rotations in degrees. The transform is a full six-DOF rigid body: `position.{x,y,z}` set the element origin in world space, and `rotation.{tilt_x,tilt_y,roll}` (degrees, Euler order `Ry·Rx·Rz`) rotate the element about that origin. Every surface in an element shares the element's composed transform, so a tilted element laterally displaces its rear surfaces as expected.

`pivot.{x,y,z}` moves the centre of rotation, in the element's own local frame, relative to the element origin — which is the **first surface's vertex**. Omitted or all-zero rotates about that front vertex (the historical behaviour, and what every file written before this field existed does). Setting `pivot.z` to half the element's axial thickness, for instance, tilts it about its middle rather than swinging the rear surfaces out.

The pivot affects `rotation` only: it never translates the element, and it is deliberately kept out of the nominal axial bookkeeping — relative position chaining, inter-element air gaps, and the sensor rebase all stay on the untilted layout, so a tilt or pivot never cascades into the sequential thickness chain.

Not to be confused with the top-level `pivots[]` array, which is a separate group-level rig applied over a set of elements after their own transforms resolve.

`position.mode` is an editor hint for tools that prefer to expose a "delta from the previous element" mode in their UI. The loader always flattens to absolute before tracing.

| Mode | Meaning |
|---|---|
| `"absolute"` (default) | `x`, `y`, `z` are in the world frame. |
| `"relative_to_preceding"` | `z` is interpreted as a delta from the previous element's resolved-absolute `z`. `x` and `y` are always absolute. The previous element's rotation and decenter are not taken into account — only its scalar z. A first element with `relative_to_preceding` is treated as absolute and the loader emits a warning. |

### Surfaces

```json
{
  "id":            "a3f9c2d1-aaaa-bbbb-cccc-dddddddddddd",
  "semi_aperture": 25.0,
  "is_stop":       false,
  "is_active":     true,
  "thickness":     5.3,
  "form":          { ... },
  "modifiers":     [ ... ]
}
```

| Field           | Meaning |
|---|---|
| `id`            | UUID — stable handle for cross-references. Required: identity is by UUID, not array position. |
| `semi_aperture` | Clear semi-diameter in mm. Used during raytracing to vignette rays that miss the surface. |
| `is_stop`       | `true` if this surface is the aperture stop. Omitted when false. |
| `is_active`     | `false` mutes the surface: the tracer skips it and it is excluded from ghost pairs. Defaults to `true` and is omitted when true. |
| `thickness`     | Axial distance to the next surface within the **same** element, in mm. On the last surface of an intermediate element it is recomputed from element-level z positions; on the last surface of the final element it is the back focal distance (gap from the back vertex to the sensor at z = 0). |
| `form`          | Surface shape. See below. |
| `modifiers`     | Ordered stack of optional per-surface effects (coatings, aperture shape). |

#### Aperture Stop

The aperture stop is marked directly on a surface with `"is_stop": true`. A stop sitting in free air between elements is a standalone single-surface element with `"sphere"` form, `radius: 0.0`, and `is_stop: true`.

#### Forms

##### Sphere

```json
{ "form": { "type": "sphere", "radius": 47.07 } }
```

Spheres cover spherical surfaces and flats. `radius: 0.0` means flat; positive radius means the centre of curvature is behind the surface.

##### Asphere

```json
{
  "form": {
    "type": "asphere",
    "radius": 47.07,
    "conic_constant": -1.0,
    "terms": [1.23e-7, -4.56e-11]
  }
}
```

General conic + polynomial sag. `terms` start at A4 (the r⁴ coefficient); the r² contribution is captured by `radius` + `conic_constant`, following the standard ISO 10110 convention.

- A pure sphere is `K = 0` with empty `terms`.
- A paraboloid is `K = −1` with empty `terms`.

Up to 8 polynomial terms (A4 through A18) are carried per surface; additional terms are truncated with a warning.

##### Cylindrical

```json
{
  "form": {
    "type": "cylindrical",
    "radius": 47.07,
    "axis": "x"
  }
}
```

`axis` is `"x"` or `"y"` and selects the cylinder axis orientation.

#### Modifier Stack

Modifiers are applied in order, first to last, and let multiple effects compose on a single surface.

**Coating.** At most one coating modifier is honoured per surface (last one
wins). Every coating declares an explicit `model` — a missing or unrecognised
one is a load error, not a silent fall back to uncoated. Omitting the modifier
entirely is equivalent to `ar_layers: 0`.

| `model` | Payload | Behaviour |
|---|---|---|
| `simple` | `ar_layers` | Scalar AR approximation; see the table below. |
| `artist` | `tint` (RGB), `strength` | Non-physical creative control: a flat reflectance `strength` tinted by `tint`. |
| `spectral` | `data[{lambda_nm, r}]`, `out_of_range` | Measured reflectance vs wavelength at normal incidence. |
| `angular` | `data[{angle_deg, r}]`, `out_of_range`, `angle_ref_ior` | Measured reflectance vs angle of incidence. |
| `spectral_angular` | `wavelengths_nm[]`, `angles_deg[]`, `r[][]`, `out_of_range`, `angle_ref_ior` | Full 2-D measured table. |
| `layers` | `layers[{material, thickness_nm, nk_table[]}]` | Physical thin-film stack, evaluated once at load time by the Transfer Matrix Method and baked to a spectral × angular table. The tracer never runs TMM. |
| `attenuator_gaussian` | `sigma`, `attenuation_background`, `attenuation_peak`, `decenter_x/y` | Not a reflectance model — a radial transmission attenuator. A surface is either a coating or an attenuator, not both. |

`out_of_range` is `"clamp"` (default) or `"discard"`.

```json
{ "type": "coating", "model": "simple", "ar_layers": 1 }
```

```json
{
  "type": "coating",
  "model": "layers",
  "layers": [
    {
      "material": "MgF2",
      "thickness_nm": 99.6,
      "nk_table": [ { "lambda_um": 0.55, "n": 1.38, "k": 0.0 } ]
    }
  ]
}
```

The `"simple"` coating model uses an integer layer count:

| `ar_layers` | Behaviour |
|:---:|---|
| `0` | Bare Fresnel — no coating (~4% reflectance at normal incidence for air/glass). |
| `1` | Single-layer MgF₂ AR coating. Fixed `n = 1.38`, quarter-wave optical thickness at λ = 550 nm (physical thickness ≈ 99.6 nm). Reflectance is wavelength-dependent via the Airy thin-film formula. |
| `2+` | Each layer beyond the first multiplies reflectance by ×0.25 — an empirical multi-coat approximation. |

**Aperture shape.** Describes the physical blade shape (used for diffraction), separate from `semi_aperture` on the surface itself (the raytracing vignetting limit).

```json
{ "type": "aperture", "shape": "circular" }
```

```json
{
  "type": "aperture",
  "shape": "polygon",
  "blades": 9,
  "rotation_deg": 15.0
}
```

```json
{
  "type": "aperture",
  "shape": "image",
  "image_path":    "iris_mask.png",
  "semi_diameter": 12.5
}
```

`semi_diameter` is meaningful **only** for `shape: "image"`, where it is the
world-space radius at the bitmap's boundary. On `circular` and `polygon` it was
once accepted and ignored; it is now rejected, since the surface's
`semi_aperture` is the single source of truth for the bounding radius.

For `shape: "image"`, the loader records the path and bounding semi-diameter; pixel data is decoded by a caller-supplied helper (typically `OpticalSystem.load_aperture_images()` on the Python side). Paths are resolved relative to the lens file's directory and survive a save/reload round-trip.

---

## Pivots

Pivots are the rig layer. The optical system is the bind pose; pivots are how artists move chunks of glass without needing to know whether a given element is a triplet or a cemented doublet. The loader composes each pivot's transform onto its targeted elements before flattening surfaces, so pivot offsets are baked into the final ray-traced geometry. Multiple pivots on the same element compose in array order.

```json
"pivots": [
  {
    "id":   "f1f2f3f4-aaaa-bbbb-cccc-dddddddddddd",
    "name": "Focus Group",
    "elements": [
      "33333333-3333-3333-3333-333333333333"
    ],
    "pivot_point": { "mode": "centroid", "x": 0, "y": 0, "z": 0 },
    "offset": {
      "position": { "x": 0.0, "y": 0.0, "z": 0.0 },
      "rotation": { "tilt_x": 0.0, "tilt_y": 0.0, "roll": 0.0 }
    },
    "exposed": [
      { "name": "focus", "attr": "offset.position.z", "min": -5.0, "max": 5.0 }
    ]
  }
]
```

| Field         | Meaning |
|---|---|
| `id`          | UUID — stable handle. |
| `name`        | Human-readable label, e.g. `"Focus Group"`. |
| `elements`    | UUIDs of the elements this pivot acts on. |
| `pivot_point` | The point of rotation. See below. |
| `offset`      | The translation + rotation this pivot currently applies. Mutated by artist controls. |
| `exposed`     | Optional list flagging which `offset` sub-fields are artist-facing controls. |

### Pivot Point

| Mode | Meaning |
|---|---|
| `"centroid"` (default) | Arithmetic mean of the targeted elements' resolved-absolute origins, evaluated at load time on the post-position-mode, post-prior-pivots state. |
| `"manual"` | The literal `x`, `y`, `z` are the pivot point, in world coordinates. |

A pivot point only matters when the pivot applies a rotation. Pure translations (no rotation offset) are invariant to the pivot point.

### Composition

For each targeted element, the pivot updates the element's transform as

```
P_e' = R_off · (P_e − P_p) + P_p + Δp
R_e' = R_off · R_e
```

where `P_e` / `R_e` are the element's resolved-absolute position and rotation, `P_p` is the pivot point, and `Δp` / `R_off` are the pivot's translation and rotation offsets. Stacked pivots iterate in array order, each one updating `(P_e, R_e)` for the next.

The loader does not clamp or reorder elements if a pivot offset pushes one element past another. The sequential ray trace still runs, but the result may be unphysical — that's the artist's responsibility.

### Exposed Controls

`exposed` flags which offset sub-fields a UI should surface as artist controls. Each entry has a `name`, an `attr` dotted path into the pivot's `offset` block, and optional `min` / `max` slider bounds.

Valid `attr` values:

- `offset.position.x`, `offset.position.y`, `offset.position.z`
- `offset.rotation.tilt_x`, `offset.rotation.tilt_y`, `offset.rotation.roll`

Coupled controls — one slider driving multiple pivots through a curve — are an application-level concern. The lens file describes the rig; the application maps high-level controls onto it.

---

## Load Pipeline

When the loader consumes a file, it:

1. **Resolves `position.mode`** across `optical_system` by rewriting any `relative_to_preceding` element's `z` to `prev_z + z`.
2. **Composes pivots.** For each pivot, computes its pivot point (centroid or manual) using the current resolved element origins, then composes the pivot's rotation/translation onto every targeted element. Multiple pivots stack in array order.
3. **Walks `optical_system` in order**, emitting one surface per geometry entry. Within-element axial distance comes from `surfaces[i].thickness`. The medium between surfaces *i* and *i+1* is `materials[i].glass`; the medium after the final surface of an element is air.
4. **Patches inter-element air gaps** from the difference between successive elements' post-pivot z positions.
5. **Resolves glasses** through `glass_catalogue`, populating each surface's "medium after" descriptor with Sellmeier or Abbe coefficients.
6. **Anchors the chain at the sensor** so the unpivoted layout ends at `z = 0`. Pivot offsets remain visible as shifts on top of that anchor — moving a focus group by +2 mm in z genuinely moves it +2 mm relative to the sensor; the loader does not re-anchor after applying pivots.

The result is a flat sequence of `Surface` records with absolute `z` positions, ready for the sequential ray trace. Pivots do not survive into the runtime — they live in the file and on the Python side for editing and round-trip.

---

## Worked Example

A cemented doublet (crown + flint) followed by a stop and a singlet, with a focus pivot on the rear singlet:

```json
{
  "format": "ghostlight-optical",
  "version": { "major": 1, "minor": 0 },
  "metadata": {
    "name": "Example 50mm",
    "focal_length_mm": 50.0
  },
  "glass_catalogue": {
    "N-BK7": {
      "name": "N-BK7",
      "catalogue_ref": "Schott:N-BK7",
      "dispersion": {
        "model": "sellmeier",
        "B": [1.03961212, 0.23179234, 1.01046945],
        "C": [0.00600069867, 0.02001791440, 103.560653]
      }
    },
    "SF5": {
      "name": "SF5",
      "catalogue_ref": "Schott:SF5",
      "dispersion": {
        "model": "sellmeier",
        "B": [1.46141885, 0.247713019, 0.949995832],
        "C": [0.0111826126, 0.0508594669, 112.041888]
      }
    }
  },
  "optical_system": [
    {
      "type": "element",
      "id":   "11111111-1111-1111-1111-111111111111",
      "name": "Front Doublet",
      "transform": { "position": { "mode": "absolute", "x": 0, "y": 0, "z": 0 } },
      "materials": [
        { "glass": "N-BK7" },
        { "glass": "SF5"   }
      ],
      "surfaces": [
        {
          "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
          "semi_aperture": 25.0,
          "thickness": 5.3,
          "form": { "type": "sphere", "radius": 47.07 },
          "modifiers": [{ "type": "coating", "model": "simple", "ar_layers": 1 }]
        },
        {
          "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
          "semi_aperture": 25.0,
          "thickness": 2.8,
          "form": { "type": "sphere", "radius": 184.28 }
        },
        {
          "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
          "semi_aperture": 24.0,
          "form": { "type": "sphere", "radius": -360.0 },
          "modifiers": [{ "type": "coating", "model": "simple", "ar_layers": 1 }]
        }
      ]
    },
    {
      "type": "element",
      "id":   "22222222-2222-2222-2222-222222222222",
      "name": "Aperture Stop",
      "transform": { "position": { "mode": "absolute", "x": 0, "y": 0, "z": 12.5 } },
      "materials": [],
      "surfaces": [
        {
          "id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
          "semi_aperture": 12.5,
          "is_stop": true,
          "form": { "type": "sphere", "radius": 0.0 }
        }
      ]
    },
    {
      "type": "element",
      "id":   "33333333-3333-3333-3333-333333333333",
      "name": "Rear Singlet",
      "transform": { "position": { "mode": "absolute", "x": 0, "y": 0, "z": 22.0 } },
      "materials": [{ "glass": "N-BK7" }],
      "surfaces": [
        {
          "id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
          "semi_aperture": 18.0,
          "thickness": 4.0,
          "form": { "type": "sphere", "radius": 58.0 },
          "modifiers": [{ "type": "coating", "model": "simple", "ar_layers": 1 }]
        },
        {
          "id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
          "semi_aperture": 18.0,
          "form": { "type": "sphere", "radius": -120.0 },
          "modifiers": [{ "type": "coating", "model": "simple", "ar_layers": 1 }]
        }
      ]
    }
  ],
  "pivots": [
    {
      "id":   "f1f2f3f4-aaaa-bbbb-cccc-dddddddddddd",
      "name": "Focus Group",
      "elements": ["33333333-3333-3333-3333-333333333333"],
      "pivot_point": { "mode": "centroid", "x": 0, "y": 0, "z": 0 },
      "offset": {
        "position": { "x": 0, "y": 0, "z": 0 },
        "rotation": { "tilt_x": 0, "tilt_y": 0, "roll": 0 }
      },
      "exposed": [
        { "name": "focus", "attr": "offset.position.z", "min": -5.0, "max": 5.0 }
      ]
    }
  ]
}
```

---

## Python API

The Python bindings expose lens files through three artist-facing types:

```python
import ghostlight

system = ghostlight.OpticalSystem.load("example.lens")

for el in system.elements:          # list[ghostlight.Element]
    print(el.name, el.position, el.position_mode)

for piv in system.pivots:           # list[ghostlight.Pivot]
    print(piv.name, piv.offset_position, piv.offset_rotation)
    piv.set_attr("offset.position.z", 2.0)

system.save("edited.lens")
```

- `OpticalSystem` is a Python subclass of the bound C++ runtime. It owns the flat surface array used by the tracer, the cached calibration and ghost-pair data, and convenience wrappers for the renderer calls.
- `Element` and `Pivot` are dataclasses reconstructed from the JSON at load time. Mutating them does not retrace anything on its own — persist via `OpticalSystem.save()` and reload to see the change in the surface chain.

---

## Out of Scope

These concerns live elsewhere in the pipeline:

| Concept | Lives where instead |
|---|---|
| Sensor / image plane size & position | Application-level (rendered scene, not lens design). |
| Environment / ambient medium         | Always air; not stored. |
| Focal length                         | Derived by tracing. `metadata.focal_length_mm` is an optional hint, not authoritative. |
| Ghost controls (muting, per-ghost RGB gain) | Render profile / application state, keyed by `(surface_id_a, surface_id_b)`. |
| Lens housing / barrel geometry       | Not defined. UUIDs provide the hook points when one is. |
| Dirt / image-pattern modifiers       | Render-pipeline concerns. |
| Zoom & focus motor curves            | The application maps user-facing controls onto pivot offsets; the lens file describes the rig, not the curve. |
