# Constellation Modding Guide

`constellations.json` lives next to `ShaderFun.exe` and is loaded at startup.  
Edit it to add, remove, or modify satellite types and constellation shells.  
The simulation falls back to built-in defaults if the file is missing or contains a JSON syntax error.

---

## Quick start

1. Open `constellations.json` in any text editor (VS Code gives autocomplete via the bundled `constellations.schema.json`).
2. Edit values, save, and restart the simulation.
3. If the simulation starts but your constellation doesn't appear, check the console output for `[SatelliteSim]` warnings.

---

## File structure

```jsonc
{
  "$schema": "./constellations.schema.json",   // gives VS Code autocomplete
  "version": 1,
  "satellite_types": [ ... ],                  // photometric models
  "constellations": [ ... ]                    // orbital shells
}
```

---

## Satellite types

Each entry in `satellite_types` defines how a group of satellites looks and reflects light.  
Constellations reference types **by name**.

```jsonc
{
  "name": "My Sat",               // referenced by constellation "type" field
  "base_color": [1.0, 0.9, 0.8], // RGB tint, linear [0–1]
  "cross_section_m2": 10.0,       // reflective area m²; brightness ∝ sqrt(area / 10)
  "primary":   { ... },           // dominant surface (required)
  "secondary": { ... },           // second surface (optional; weight:0 disables)
  "diffuse": 0.02,                // isotropic floor [0–1]; visible from all angles
  "mirror_frac": 0.05             // near-perfect mirror fraction [0–1]
}
```

### Surface spec

```jsonc
{
  "attitude": "NadirPointing",  // how the surface normal is oriented (see below)
  "spec_exp": 18.0,             // Phong exponent — 0=Lambertian, 200=near-mirror
  "weight": 1.0                 // contribution weight; secondary typically 0–0.5
}
```

### Attitude modes

| Value | Surface normal | Typical use |
|---|---|---|
| `NadirPointing` | Toward Earth's centre | Antenna/phased-array face (Starlink) |
| `SunTracking` | Toward the Sun | Solar panels (ISS, OneWeb) |
| `Tumbling` | Random spin around a fixed body axis | Debris, defunct satellites |
| `Perpendicular` | `cross(primary_normal, nadir)` — secondary only | Along-track radiators |
| `AntiNadir` | Away from Earth's centre — secondary only | Deep-space-facing radiators |
| `FlatMirror45` | `normalize(sunDir + nadir)` | Reflects sunlight straight down |
| `TargetedReflector` | `normalize(sunDir + toTarget)` | Focuses reflected beam on nearest night-side ground point |

### Brightness notes

- `cross_section_m2`: 10 m² → scale 1.0, 2376 m² → scale ~15.4. Use realistic areas for real constellations.
- `spec_exp`: 3 = broad glow (GEO comsat), 18 = sharp flash (solar panel), 200 = sub-degree spike (mirror).
- `mirror_frac`: adds a 300× intensity peak at perfect alignment. 0.97 gives the Reflect Mirror its extreme brightness. Keep at 0 for most types.
- `diffuse`: the fraction visible at all angles regardless of specular alignment. 0.02 gives a faint steady-state magnitude; 0 means the satellite is only visible during specular flares.

---

## Constellations

Each entry in `constellations` defines one orbital shell.

```jsonc
{
  "name": "My Constellation",    // shown in the Settings panel
  "type": "My Sat",              // must match a satellite_types name exactly
  "alt_km": 550.0,               // orbital altitude in km
  "incl_deg": 53.0,              // inclination in degrees (see distribution notes)
  "num_planes": 72,              // orbital planes (or total count for RandomShell)
  "per_plane": 61,               // satellites per plane — total = num_planes × per_plane
  "enabled": true,               // shown on startup; togglable in Settings
  "distribution": "Walker"       // orbit generation algorithm (see below)
}
```

> **Total satellite count** = `num_planes × per_plane` for **all** distributions.  
> Keep the total across all enabled constellations under `MAX_SATELLITES` (200,000).

### Distributions

#### Walker

Regular grid of `num_planes` planes × `per_plane` satellites. Standard for real mega-constellations.

```jsonc
{ "distribution": "Walker", "alt_km": 550, "incl_deg": 53, "num_planes": 72, "per_plane": 61 }
```

#### RandomShell

Random RAAN, random inclination in `[0, incl_deg]`, jittered altitude. Good for debris.

```jsonc
{
  "distribution": "RandomShell",
  "alt_km": 1000, "incl_deg": 180,  // incl_deg = max inclination drawn uniformly
  "num_planes": 100, "per_plane": 30,
  "alt_jitter_km": 500              // per-satellite ± altitude scatter
}
```

#### Disk

One or more concentric rings in a fixed orbital plane. Used for SSO/sun-synchronous shells.

```jsonc
{
  "distribution": "Disk",
  "alt_km": 1250, "incl_deg": 0,          // incl_deg ignored if align_terminator=true
  "num_planes": 200, "per_plane": 100,
  "num_rings": 10,                         // concentric rings
  "ring_spacing_km": 75.0,                 // altitude gap between rings
  "alt_jitter_km": 5.0,                    // per-satellite scatter within each ring
  "align_terminator": true,                // derive incl + RAAN from Sun; precesses at SSO rate
  "raan_deg": 0                            // ignored when align_terminator=true
}
```

`align_terminator: true` computes the SSO inclination from the J2 formula at the shell altitude, then anchors the RAAN to the dawn–dusk terminator and precesses it at ~1 revolution/year. This gives a sun-synchronous orbit that stays fixed relative to the day/night boundary regardless of simulation date.

---

## Adding a new satellite type + constellation

Minimum steps:

1. Add an entry to `satellite_types`:
   ```jsonc
   {
     "name": "My Type",
     "base_color": [1.0, 1.0, 1.0],
     "cross_section_m2": 15.0,
     "primary": { "attitude": "SunTracking", "spec_exp": 12.0, "weight": 1.0 },
     "diffuse": 0.03,
     "mirror_frac": 0.04
   }
   ```

2. Add an entry to `constellations`:
   ```jsonc
   {
     "name": "My Constellation",
     "type": "My Type",
     "alt_km": 600,
     "incl_deg": 45,
     "num_planes": 10,
     "per_plane": 20,
     "enabled": true,
     "distribution": "Walker"
   }
   ```

3. Restart. Check stderr output for any parse warnings.

---

## Adding new features (for developers)

When adding a new field to `SatelliteType` or `ConstellationConfig` in C++:

1. Add the field to the struct with a sensible default.
2. In `loadDefinitions()`, read it with `jt.value("new_field", default_value)` — missing keys silently return the default, so existing JSON files keep working.
3. Add the field to `loadHardcoded()` in `SatelliteSim.cpp`.
4. Update `constellations.json` in `data/` with the new field on relevant entries.
5. Update `constellations.schema.json`:
   - Add a property under `satellite_types.items.properties` or `constellations.items.properties`.
   - Include a `"description"` string — this is what users see as a tooltip in VS Code.
   - Add it to `"additionalProperties": false` sections if it's definitive.
6. Update this file (`CONSTELLATION_MODDING.md`) with a description.

When adding a new `AttitudeMode` value:
1. Add it to the `AttitudeMode` enum in `SatelliteSim.h`.
2. Add a case in `parseAttitudeMode()` in `SatelliteSim.cpp`.
3. Handle it in `updatePositions()`.
4. Add it to the `"enum"` array in `constellations.schema.json` under `$defs.surface_spec.properties.attitude`.
5. Document it in the **Attitude modes** table above.

---

## Limits and performance

| Limit | Value | Notes |
|---|---|---|
| `MAX_SATELLITES` | 200,000 | Hard cap; excess sats are truncated. Raise in `SatelliteSim.h` if needed (requires GPU buffer reallocation). |
| Constellations in UI | 256 | UI loop cap; all constellations load fine, but only the first 256 show toggle buttons. |
| Satellite types | Unlimited | Stored in `std::vector`; only limited by memory. |

CPU update time (updatePositions) scales linearly with active satellite count:

| Active sats | Approx. CPU time |
|---|---|
| 1,000 | ~0.1 ms |
| 10,000 | ~1 ms |
| 100,000 | ~10 ms (near 60 Hz budget) |

Disable large constellations (Starlink Gen2, Xingwang) when combining multiple custom shells near the cap.
