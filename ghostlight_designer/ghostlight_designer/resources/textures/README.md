# Example textures

Bundled example images for the designer's **Textures** panel. All are 8-bit
greyscale PNGs. Regenerate them with:

```
python ghostlight_designer/ghostlight_designer/resources/textures/generate_textures.py
```

The generator is seeded, so re-running reproduces the committed files exactly.
Adding a texture = writing one function and appending a `(name, fn, seed)` row
to `MASKS` or `DIRT`.

## The one mechanism, two uses

There is a single raster mechanism in Ghostlight — the `APERTURE_IMAGE` bitmap on
`OpticalSystem.aperture_images[i]`, parallel to `surfaces[i]`. Two code paths
read it, and they interpret it differently:

| Path | Where | Interpretation |
| --- | --- | --- |
| Ray trace | `trace.cpp` / `trace_cuda.h` | **Binary.** `pixel > 0.5` passes, else `VIGNETTED`. Greys are meaningless except to the SDF bake. |
| Diffraction pupil | `starburst_render.cu` | **Graded.** The sample multiplies pupil amplitude, so 0.5 = 50 % transmission. Gated on `DiffractionConfig.use_surface_textures`; uses the *front-most* surface carrying a texture. |

So `aperture_*.png` and `dirt_*.png` are the same file format authored for
different consumers — nothing stops you loading a dirt map as a hard matte, you
just get its 0.5-threshold silhouette.

## `aperture_*.png` — hard mattes

1024x1024, black/white, anti-aliased edges (the grey boundary pixels are what
the SDF bake and the MDFT pupil resample from). Every one keeps a black margin:
the GPU sampler uses `cudaAddressModeClamp`, so a mask running to the texture
edge would leak transmission outside the aperture.

| File | What it gives you |
| --- | --- |
| `aperture_iris_6blade.png` | Ordinary wide-open 6-blade iris — 6/12-point star |
| `aperture_iris_9blade_stopped.png` | Stopped down, near-straight blades — hard 18-point star |
| `aperture_iris_14blade_chipped.png` | Near-round 14-blade with a chipped blade and a rim nick — asymmetric spikes |
| `aperture_spider_vanes.png` | Mirror-lens spider vanes — clean 4-spike X |
| `aperture_star_5.png` | 5-point star gobo |
| `aperture_heart.png` | Heart gobo — and the orientation test asset, see below |
| `aperture_anamorphic_oval.png` | 2:1 oval slot — horizontally stretched flare |

Assign one in the Textures panel: pick the stop surface, set **Semi-diameter**
to the aperture's physical half-width in mm, then **Load**. The bitmap's full
width maps to `2 x semi_diameter` mm, so the shape fills the square — set the
semi-diameter to the *circumscribing* half-width, not the inscribed one.

Any of these on the stop also exercises the HURB image-aperture SDF path.

### The axial ray must survive

There is no true catadioptric annulus here, and that is deliberate. The
entrance-pupil solve needs a clear ray on axis: block the centre of the stop by
*any* amount and `calibration()` returns `entrance_pupil_semi = 0`, after which
the starburst pass bails with `degenerate first-order optics (no focal length /
pupil)` and nothing renders. `aperture_spider_vanes.png` therefore stops its
vanes short of the axis rather than joining them at a central obstruction; it
still gives the mirror-lens 4-spike. Bear the same constraint in mind when
authoring your own mattes — a mask with an opaque centre will not render.

### Orientation

The tracer maps `v = 0.5 + hit.y / (2 x semi_diameter)`, so texture row 0 lands
at world **-y** (bottom), while PNG row 0 is the **top** of the file. A matte is
therefore rendered vertically mirrored relative to how it looks in an image
viewer or in the panel's *Raw texture* preview.

Every mask here is vertically symmetric except `aperture_star_5.png` and
`aperture_heart.png`. The heart is deliberately kept file-upright: load it and
the render shows it upside down — that's the convention, not a bug, and it's the
fastest way to confirm which way round a mask of your own needs to be authored.

## `dirt_*.png` — front-glass transmission maps

2048x2048, near-white. 1.0 = clean glass; the map only ever attenuates (nothing
exceeds 1.0 — a value above 1 would amplify the pupil, which is unphysical).
No black margin here: a dark border would clamp-darken the pupil rim.

| File | Character |
| --- | --- |
| `dirt_dust_light.png` | Well-kept lens — sparse motes, two fibres |
| `dirt_dust_heavy.png` | Neglected — dense dust, lint, greasy film |
| `dirt_fingerprint.png` | Off-axis thumb print with its whorl ridges |
| `dirt_scratches_polish.png` | Cleaning swirls plus three deep gouges |
| `dirt_water_droplets.png` | Rain — dark bodies, bright meniscus rims |
| `dirt_grime_haze.png` | Gentle rim-weighted veil; softens rather than sparkles |

Assign these to the **front element's front surface**, not the stop — the pupil
path picks the front-most textured surface and projects the pupil ray forward
onto it (`tex_d0`), which is what makes the dirt slide across frame as the
source moves off-axis. Setting the shape to `IMAGE` also makes the tracer treat
the map as a binary matte, so keep the values well above 0.5 (all of these are)
unless you want dust motes vignetting rays as well as diffracting.

Turn `use_surface_textures` on (the checkbox in the panel) and watch the
*Composited pupil* / *Starburst sprite* views: dust and scratches broaden the
star's base and add fine radial hair, which is the whole reason to use them.

## Gotchas

* **16-bit input.** Both `_decode_image` in the panel and
  `OpticalSystem.load_aperture_images` normalise with a `peak > 1.5 -> /255`
  heuristic, so a 16-bit PNG/TIFF (0..65535) ends up at values up to 257.
  Author 8-bit, or pre-normalise to float 0..1.
* **EXR.** It's in the panel's file filter but plain Pillow won't decode it
  without a plugin. PNG and TIFF are the safe formats.
* **Paths, not pixels.** A saved `.lens` records `image_path`, resolved
  relative to the lens file's directory — the bitmap itself is never embedded.
  Move a lens without its textures and the assignment dangles.
* **No size constraints.** Nothing requires power-of-two or square images;
  `cudaMallocArray` takes whatever it is given. Square is just natural because
  UV maps to a square of half-width `semi_diameter`.
