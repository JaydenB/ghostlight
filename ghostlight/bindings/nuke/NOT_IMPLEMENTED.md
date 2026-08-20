# Nuke node — not implemented

There is no Nuke integration yet. This directory is a placeholder so the intent
is visible in the repo, and so the note below is not lost.

Nothing here is wired into the build. There is no stub in the bindings.

---

## The hook that already exists

The C++ side already carries the plumbing a Nuke node needs, and it is dormant
rather than missing. `render_ghost_pipeline`, `launch_ghost_render`, and the
ghost / starburst / veil scatter kernels all take four extra ints:

    fmt_w, fmt_h, fmt_x0_in_buf, fmt_y0_in_buf

Today the sole production caller (`SourceFlareRenderer.cpp`, which
`render_point_flare` also delegates through) passes the identity:

    render_ghost_pipeline(..., /*fmt_w=*/w, /*fmt_h=*/h,
                               /*fmt_x0_in_buf=*/0, /*fmt_y0_in_buf=*/0, ...);

so they are currently a no-op. **They are kept on purpose.** Deleting them would
look harmless in every current render and every value gate, while removing the
only extension point the node needs.

## Format vs. bounding box

Nuke separates a **format** (the nominal output resolution, e.g. 1920×1080) from
the **bounding box** of the data actually computed, which may be larger — a flare
can legitimately spill outside the frame and still matter when the result is
composited or re-framed.

Ghostlight mirrors that split:

- **Format** = `fmt_w × fmt_h`. The optics are *calibrated to the format*: the
  lens calibration (sensor half-extent → max field angle) and the source angular
  mapping are all relative to this rectangle.
- **Buffer / bbox** = `width × height`, the `out_r/g/b` arrays the kernel writes.
  This is the real allocation and can be larger than the format.
- **Placement** = `(fmt_x0_in_buf, fmt_y0_in_buf)`, the pixel coordinate *within
  the buffer* of the format's origin.

## The mapping

A sample landing at sensor position `pos` (millimetres, origin on the optical
axis, spanning `±sensor_half_w` × `±sensor_half_h`) becomes a buffer pixel as:

    px = (pos.x / (2 * sensor_half_w) + 0.5) * fmt_w + fmt_x0_in_buf
    py = (pos.y / (2 * sensor_half_h) + 0.5) * fmt_h + fmt_y0_in_buf

In three steps: sensor mm → normalised `[0,1]` across the **format**; then to
**format** pixels; then shifted into **buffer** pixels. The subsequent bounds
check is against `width`/`height` (the buffer), so a sample outside the format
but inside the buffer is still written.

## How a node would drive it

| Nuke concept                        | Ghostlight param                  |
|-------------------------------------|-----------------------------------|
| Format knob (output resolution)     | `fmt_w`, `fmt_h`                  |
| Node bbox (requested region)        | buffer `width`, `height`          |
| Format origin within the bbox       | `fmt_x0_in_buf`, `fmt_y0_in_buf`  |

Asked to render a bbox extending `L` pixels left and `B` below the format (Nuke
bboxes can have negative origins), the node allocates
`width = fmt_w + L + R`, `height = fmt_h + B + T`, and passes
`fmt_x0_in_buf = L`, `fmt_y0_in_buf = B`. The optics stay calibrated to
`fmt_w × fmt_h`; the border captures flare that spills past the frame.

## What must stay intact

- The four params on `render_ghost_pipeline`, `launch_ghost_render`, and the
  three kernels (`ghost_render.cu`, `starburst_render.cu`, `veil_render.cu`).
- The `(pos / (2*sensor_half) + 0.5) * fmt + fmt_x0` mapping — do **not**
  collapse it to `* width` even though the two are numerically equal today.
- The bounds check against `width`/`height` (buffer), not `fmt_w`/`fmt_h`.

When a node lands, the only new work is a non-identity caller. No kernel change.
