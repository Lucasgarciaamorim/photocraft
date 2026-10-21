//! A camera's rendering of one photo, learned from the camera's own JPEG of it, for raw data
//! that already carries a colour calibration (a DNG converted from a Canon CR3, whose sensor
//! data this crate does not decode itself).
//!
//! [`crate::fit_look`] serves raws without calibration with one matrix and one tone curve. With
//! a calibrated raw the colorimetry is already right; what separates the neutral development
//! from the camera's picture is its rendering: a tone curve, saturation, hue-dependent twists
//! (Canon's reds) and the lens corrections the camera applied to its JPEG. The model works on
//! the neutral development's code values (ProPhoto RGB, gamma 1.8, 0..1):
//!
//! ```text
//! y ≈ L(encode(g(r) · decode(d)))
//! ```
//!
//! `g(r) = 1 + a·r² + b·r⁴` is the peripheral-illumination (vignetting) correction, `r` the
//! distance from the frame centre over the half diagonal; `L` is a lookup table on a
//! [`LUT_EDGE`]³ lattice, trilinear, fitted by penalised least squares: the data term over a
//! grid of at most [`GRID_EDGE`] cells on the long edge, a thin-plate-like smoothness penalty
//! (the squared graph Laplacian of the lattice) so nodes without data continue their
//! neighbours, and a weak pull toward identity. The camera JPEG may be corrected for lens
//! distortion while the raw is not: its cells are paired with the raw's through a centred
//! radial scale `s + k·r²` chosen by luminance correlation. `g` comes from the cells away from
//! the centre, read through the tone relation the central cells show. One cell in three is held
//! out; the rendering is accepted only when it brings the held-out mean CIE76 ΔE (Lab D50) to
//! at most 85 % of the neutral development's.
//!
//! This is a per-file estimate of the camera's rendering, not a measured profile: local tone
//! mapping (Canon's Auto Lighting Optimizer and the like) is approximated by the global table,
//! and colours the photo does not show follow the smoothness penalty.

use crate::color;
use crate::develop::Developed;
use crate::error::RawError;
use crate::look::{LookRejected, Reference, reference_cell, srgb_to_linear};
use crate::par;

/// Long edge of the fit grid, in cells.
pub const GRID_EDGE: usize = 320;

/// Lattice nodes per axis.
pub const LUT_EDGE: usize = 17;

/// The development's encoding: ProPhoto RGB, gamma 1.8.
const GAMMA: f64 = 1.8;

/// Smoothness weight, relative to the mean data weight per lattice node.
const SMOOTHNESS: f64 = 0.01;

/// Pull toward identity, relative to the mean data weight per lattice node.
const ANCHOR: f64 = 1e-3;

/// Most conjugate-gradient iterations per channel: the final table, and the central tables the
/// vignetting is read through (they need less precision).
const MAX_ITERATIONS: usize = 400;
const PROBE_ITERATIONS: usize = 120;

/// A rendering fitted to the camera's JPEG: apply it to the neutral development with
/// [`CameraRendering::apply`].
#[derive(Debug, Clone, PartialEq)]
pub struct CameraRendering {
    /// Lattice nodes, [`LUT_EDGE`]³ with red fastest: output code values 0..1.
    nodes: Vec<[f32; 3]>,
    /// `a`, `b` of the vignetting correction `g(r) = 1 + a·r² + b·r⁴` (0, 0: none).
    pub vignette: [f64; 2],
    /// `s`, `k` of the radial scale pairing the JPEG's cells with the raw's (1, 0: none).
    pub alignment: [f64; 2],
    /// Held-out mean CIE76 ΔE against the camera JPEG with this rendering.
    pub delta_e: f64,
    /// The same for the neutral development.
    pub delta_e_neutral: f64,
}

impl CameraRendering {
    /// The vignetting correction at normalised radius `r`.
    fn gain(&self, r: f64) -> f64 {
        let r2 = r * r;
        1.0 + self.vignette[0] * r2 + self.vignette[1] * r2 * r2
    }

    /// The table at code values `u` (0..1), trilinear.
    fn eval(&self, u: [f32; 3]) -> [f32; 3] {
        let n = LUT_EDGE;
        let top = (n - 1) as f32;
        let mut base = [0usize; 3];
        let mut frac = [0f32; 3];
        for k in 0..3 {
            let p = if u[k].is_finite() { u[k].clamp(0.0, 1.0) * top } else { 0.0 };
            let i = (p as usize).min(n - 2);
            base[k] = i;
            frac[k] = p - i as f32;
        }
        let mut out = [0f32; 3];
        for corner in 0..8usize {
            let (dx, dy, dz) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let wgt = (if dx == 1 { frac[0] } else { 1.0 - frac[0] })
                * (if dy == 1 { frac[1] } else { 1.0 - frac[1] })
                * (if dz == 1 { frac[2] } else { 1.0 - frac[2] });
            if wgt <= 0.0 {
                continue;
            }
            let idx = (base[0] + dx) + n * ((base[1] + dy) + n * (base[2] + dz));
            let node = self.nodes.get(idx).copied().unwrap_or([0.0; 3]);
            for k in 0..3 {
                out[k] += wgt * node[k];
            }
        }
        out
    }

    /// `true` when the radial scale moves a corner of a `w × h` frame by more than half a pixel.
    fn corrects_distortion(&self, w: usize, h: usize) -> bool {
        let [s, k] = self.alignment;
        let half_diagonal = (w as f64 / 2.0).hypot(h as f64 / 2.0);
        ((s + k - 1.0) * half_diagonal).abs() > 0.5 || ((s - 1.0) * half_diagonal).abs() > 0.5
    }

    /// Renders a neutral development in place (its upright 16-bit ProPhoto code values): the
    /// camera's lens-distortion correction (the radial scale, bicubic), its vignetting correction,
    /// then the table.
    pub fn apply(&self, dev: &mut Developed) -> Result<(), RawError> {
        let (w, h) = (dev.width as usize, dev.height as usize);
        if w == 0 || h == 0 || w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(dev.rgb.len()) {
            return Err(RawError::malformed("the developed image's buffer does not match its size"));
        }
        if self.nodes.len() != LUT_EDGE * LUT_EDGE * LUT_EDGE {
            return Err(RawError::malformed("the camera rendering's table is incomplete"));
        }
        let vignette = self.vignette != [0.0, 0.0];
        let source = if self.corrects_distortion(w, h) { Some(dev.rgb.clone()) } else { None };
        let [s, k] = self.alignment;
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let half_diagonal = cx.hypot(cy).max(1.0);
        let band = par::band_rows(w);
        par::chunks_mut(&mut dev.rgb, band * w * 3, |b, chunk| {
            for (r, row) in chunk.chunks_exact_mut(w * 3).enumerate() {
                let y = b * band + r;
                let dy = y as f64 + 0.5 - cy;
                for (x, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let dx = x as f64 + 0.5 - cx;
                    let radius = dx.hypot(dy) / half_diagonal;
                    let code: [f32; 3] = match &source {
                        Some(src) => {
                            let f = s + k * radius * radius;
                            bicubic(src, w, h, cx + dx * f - 0.5, cy + dy * f - 0.5)
                        }
                        None => px.map(f32::from),
                    };
                    let u = if vignette {
                        let g = self.gain(radius);
                        code.map(|c| encode((f64::from(c) / 65535.0).max(0.0).powf(GAMMA) * g) as f32)
                    } else {
                        code.map(|c| c / 65535.0)
                    };
                    let v = self.eval(u);
                    *px = v.map(|c| (c.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16);
                }
            }
        });
        Ok(())
    }

    /// The note the import reports.
    pub fn note(&self) -> String {
        let g = self.gain(1.0);
        let mut lens = Vec::new();
        if (self.alignment[0] + self.alignment[1] - 1.0).abs() > 1e-3 || (self.alignment[0] - 1.0).abs() > 1e-3 {
            lens.push(format!("distortion corrected ({:+.1}% at the corners)", (self.alignment[0] + self.alignment[1] - 1.0) * 100.0));
        }
        if self.vignette != [0.0, 0.0] {
            lens.push(format!("corners brightened ×{g:.2}"));
        }
        let lens = if lens.is_empty() { String::new() } else { format!("; lens corrected like the camera's JPEG: {}", lens.join(", ")) };
        format!("colour rendering learned from the camera's own JPEG (mean ΔE {:.1} against it; {:.1} neutral){lens}", self.delta_e, self.delta_e_neutral)
    }
}

/// Catmull-Rom weights for the four taps around a sample `t` (0..1) past the second.
fn catmull_rom(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [(-t3 + 2.0 * t2 - t) / 2.0, (3.0 * t3 - 5.0 * t2 + 2.0) / 2.0, (-3.0 * t3 + 4.0 * t2 + t) / 2.0, (t3 - t2) / 2.0]
}

/// Interleaved 16-bit RGB at pixel position (`sx`, `sy`) (pixel centres on integers), bicubic,
/// edges extended.
fn bicubic(src: &[u16], w: usize, h: usize, sx: f64, sy: f64) -> [f32; 3] {
    let (sx, sy) = (if sx.is_finite() { sx.clamp(0.0, (w - 1) as f64) } else { 0.0 }, if sy.is_finite() { sy.clamp(0.0, (h - 1) as f64) } else { 0.0 });
    let (x0, y0) = (sx.floor() as isize, sy.floor() as isize);
    let (wx, wy) = (catmull_rom((sx - x0 as f64) as f32), catmull_rom((sy - y0 as f64) as f32));
    let mut out = [0f32; 3];
    for (j, wyj) in wy.iter().enumerate() {
        let yy = (y0 + j as isize - 1).clamp(0, h as isize - 1) as usize;
        for (i, wxi) in wx.iter().enumerate() {
            let xx = (x0 + i as isize - 1).clamp(0, w as isize - 1) as usize;
            let wgt = wyj * wxi;
            let base = (yy * w + xx) * 3;
            for (k, o) in out.iter_mut().enumerate() {
                *o += wgt * f32::from(src.get(base + k).copied().unwrap_or(0));
            }
        }
    }
    out.map(|v| v.clamp(0.0, 65535.0))
}

fn reject<T>(why: impl Into<String>) -> Result<T, LookRejected> {
    Err(LookRejected(why.into()))
}

/// Gamma-1.8 code value (0..1) of linear `v`.
fn encode(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 { v.min(1.0).powf(1.0 / GAMMA) } else { 0.0 }
}

/// Linear value of every 16-bit code value.
fn decode_table() -> Vec<f32> {
    (0..=u16::MAX).map(|c| (f64::from(c) / 65535.0).powf(GAMMA) as f32).collect()
}

/// The luminance (Y, D50) row of linear ProPhoto RGB.
fn y_row() -> [f64; 3] {
    color::rgb_to_xyz(color::ROMM, color::D50_XY)[1]
}

/// CIE L*a*b* (D50) of gamma-1.8 ProPhoto code values.
struct Lab {
    to_xyz: color::Mat3,
    white: [f64; 3],
}

impl Lab {
    fn new() -> Self {
        let to_xyz = color::rgb_to_xyz(color::ROMM, color::D50_XY);
        Lab { white: color::apply(&to_xyz, [1.0; 3]), to_xyz }
    }

    fn of_code(&self, u: [f64; 3]) -> [f64; 3] {
        let xyz = color::apply(&self.to_xyz, u.map(|c| if c.is_finite() { c.clamp(0.0, 1.0).powf(GAMMA) } else { 0.0 }));
        let f = |t: f64| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
        let [fx, fy, fz] = [0, 1, 2].map(|k| f(xyz[k] / self.white[k]));
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
    }

    fn delta_e(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        let (p, q) = (self.of_code(a), self.of_code(b));
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }
}

/// A reference grid cell.
#[derive(Debug, Clone, Copy)]
struct RefCell {
    /// The JPEG's colour, linear ProPhoto RGB.
    rgb: [f64; 3],
    lum: f64,
    /// Fraction of the cell's pixels with a channel at 250 or more.
    clipped: f64,
    /// Weight from the JPEG's local contrast (edges count less).
    edge_weight: f64,
}

/// Cell centres and their normalised radius on a `gw × gh` grid.
struct Geometry {
    gw: usize,
    gh: usize,
    cx: f64,
    cy: f64,
    half_diagonal: f64,
}

impl Geometry {
    fn new(gw: usize, gh: usize) -> Self {
        let (cx, cy) = (gw as f64 / 2.0, gh as f64 / 2.0);
        Geometry { gw, gh, cx, cy, half_diagonal: cx.hypot(cy).max(1.0) }
    }

    fn centre(&self, i: usize) -> (f64, f64) {
        ((i % self.gw) as f64 + 0.5, (i / self.gw) as f64 + 0.5)
    }

    fn radius(&self, i: usize) -> f64 {
        let (x, y) = self.centre(i);
        (x - self.cx).hypot(y - self.cy) / self.half_diagonal
    }
}

/// Mean linear ProPhoto of the developed image over each cell of a `gw × gh` grid.
fn dev_grid(dev: &Developed, lin: &[f32], gw: usize, gh: usize) -> Vec<[f64; 3]> {
    let (w, h) = (dev.width as usize, dev.height as usize);
    let rows = par::map(gh, |j| {
        let (y0, y1) = (j * h / gh, ((j + 1) * h / gh).max(j * h / gh + 1).min(h));
        let sy = (y1 - y0).div_ceil(8).max(1);
        (0..gw)
            .map(|i| {
                let (x0, x1) = (i * w / gw, ((i + 1) * w / gw).max(i * w / gw + 1).min(w));
                let sx = (x1 - x0).div_ceil(8).max(1);
                let (mut sum, mut n) = ([0.0f64; 3], 0usize);
                for y in (y0..y1).step_by(sy) {
                    for x in (x0..x1).step_by(sx) {
                        let Some(px) = dev.rgb.get((y * w + x) * 3..(y * w + x) * 3 + 3) else { continue };
                        for (s, c) in sum.iter_mut().zip(px) {
                            *s += f64::from(lin.get(usize::from(*c)).copied().unwrap_or(0.0));
                        }
                        n += 1;
                    }
                }
                if n == 0 { [0.0; 3] } else { sum.map(|s| s / n as f64) }
            })
            .collect::<Vec<_>>()
    });
    rows.into_iter().flatten().collect()
}

/// The camera JPEG over the same grid.
fn ref_grid(r: &Reference, gw: usize, gh: usize) -> Vec<Option<RefCell>> {
    let (rw, rh) = (r.width as usize, r.height as usize);
    let lut: Vec<f64> = (0..256).map(|v| srgb_to_linear(v as f64 / 255.0)).collect();
    let to_pro = color::mul(&color::xyz_d50_to_prophoto(), &color::srgb_to_xyz_d50());
    let y = y_row();
    let rows = par::map(gh, |j| {
        let (y0, y1) = (j * rh / gh, ((j + 1) * rh / gh).max(j * rh / gh + 1).min(rh));
        (0..gw)
            .map(|i| {
                let (x0, x1) = (i * rw / gw, ((i + 1) * rw / gw).max(i * rw / gw + 1).min(rw));
                reference_cell(r, &lut, x0, x1, y0, y1).map(|(srgb, clipped, edge_weight)| {
                    let rgb = color::apply(&to_pro, srgb).map(|v| v.max(0.0));
                    RefCell { rgb, lum: y[0] * rgb[0] + y[1] * rgb[1] + y[2] * rgb[2], clipped, edge_weight }
                })
            })
            .collect::<Vec<_>>()
    });
    rows.into_iter().flatten().collect()
}

/// The developed grid at the position the JPEG cell `i` shows under the radial scale `s + k·r²`
/// (bilinear between cell centres).
fn sample_aligned(dev: &[[f64; 3]], geo: &Geometry, i: usize, [s, k]: [f64; 2]) -> Option<[f64; 3]> {
    let (x, y) = geo.centre(i);
    let r = geo.radius(i);
    let f = s + k * r * r;
    let qx = (geo.cx + (x - geo.cx) * f - 0.5).clamp(0.0, (geo.gw - 1) as f64);
    let qy = (geo.cy + (y - geo.cy) * f - 0.5).clamp(0.0, (geo.gh - 1) as f64);
    let (x0, y0) = (qx.floor() as usize, qy.floor() as usize);
    let (fx, fy) = (qx - x0 as f64, qy - y0 as f64);
    let mut out = [0.0; 3];
    for (dx, dy, wgt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        if wgt <= 0.0 {
            continue;
        }
        let (cx, cy) = ((x0 + dx).min(geo.gw - 1), (y0 + dy).min(geo.gh - 1));
        let v = dev.get(cy * geo.gw + cx)?;
        for (o, c) in out.iter_mut().zip(v) {
            *o += wgt * c;
        }
    }
    Some(out)
}

/// Pearson correlation of `(a, b)` pairs; NaN for fewer than 100.
fn correlation(pairs: &[(f64, f64)]) -> f64 {
    if pairs.len() < 100 {
        return f64::NAN;
    }
    let n = pairs.len() as f64;
    let (ma, mb) = (pairs.iter().map(|p| p.0).sum::<f64>() / n, pairs.iter().map(|p| p.1).sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (a, b) in pairs {
        sab += (a - ma) * (b - mb);
        saa += (a - ma) * (a - ma);
        sbb += (b - mb) * (b - mb);
    }
    if saa <= 0.0 || sbb <= 0.0 { f64::NAN } else { sab / (saa * sbb).sqrt() }
}

/// The radial scale `[s, k]` that best pairs the JPEG's cells with the raw's (luminance
/// correlation, square-root domain), starting from none, and that correlation.
fn align(dev: &[[f64; 3]], refs: &[Option<RefCell>], geo: &Geometry) -> ([f64; 2], f64) {
    let y = y_row();
    let score = |sk: [f64; 2]| {
        let mut pairs = Vec::new();
        for i in (0..refs.len()).step_by(2) {
            let (Some(Some(rc)), Some(d)) = (refs.get(i), sample_aligned(dev, geo, i, sk)) else { continue };
            if rc.clipped >= 0.05 {
                continue;
            }
            let lum = (y[0] * d[0] + y[1] * d[1] + y[2] * d[2]).max(0.0);
            pairs.push((lum.sqrt(), rc.lum.max(0.0).sqrt()));
        }
        correlation(&pairs)
    };
    let mut best = [1.0, 0.0];
    let mut best_score = score(best);
    if !best_score.is_finite() {
        return (best, best_score);
    }
    let consider = |sk: [f64; 2], best: &mut [f64; 2], best_score: &mut f64| {
        let r = score(sk);
        if r.is_finite() && r > *best_score + 1e-5 {
            (*best, *best_score) = (sk, r);
        }
    };
    // Wide-angle kit zooms correct up to about 10% of barrel distortion at the corners.
    for si in 0..=16 {
        for ki in 0..=10 {
            consider([0.86 + 0.01 * f64::from(si), -0.14 + 0.02 * f64::from(ki)], &mut best, &mut best_score);
        }
    }
    let coarse = best;
    for si in -2..=2 {
        for ki in -2..=2 {
            consider([coarse[0] + 0.0025 * f64::from(si), coarse[1] + 0.005 * f64::from(ki)], &mut best, &mut best_score);
        }
    }
    (best, best_score)
}

/// Median of `v` (reordered); NaN when empty.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Gains tried per cell when reading the vignetting off a table: 0.6 to 2.5, log-spaced.
const GAIN_STEPS: usize = 41;

/// `[a, b]` of the vignetting correction the camera applied, read through `probe` (a table fitted
/// on the central cells): per cell away from the centre, the brightness gain on the raw that lets
/// the table reproduce the JPEG best; their medians per radius band, fitted by `g(r)`. `None` when
/// too few cells tell.
fn estimate_vignette(probe: &CameraRendering, support: &[u32], paired: &[Option<[f64; 3]>], refs: &[Option<RefCell>], geo: &Geometry) -> Option<[f64; 2]> {
    let gains: Vec<f64> = (0..GAIN_STEPS).map(|i| (0.6f64.ln() + (2.5f64.ln() - 0.6f64.ln()) * i as f64 / (GAIN_STEPS - 1) as f64).exp()).collect();
    let mut bands: Vec<Vec<f64>> = vec![Vec::new(); 7];
    for i in 0..refs.len() {
        let r = geo.radius(i);
        if r < 0.3 {
            continue;
        }
        let (Some(Some(d)), Some(Some(rc))) = (paired.get(i), refs.get(i)) else { continue };
        if !(rc.clipped < 0.01 && rc.edge_weight > 0.3 && rc.lum > 0.002 && rc.lum < 0.8) {
            continue;
        }
        let g0 = probe.gain(r);
        let target = rc.rgb.map(encode);
        let error = |m: f64| {
            let out = probe.eval(d.map(|c| encode(c * g0 * m) as f32));
            (0..3).map(|k| (f64::from(out[k]) - target[k]).powi(2)).sum::<f64>()
        };
        let Some((best, _)) = gains.iter().map(|&m| (m, error(m))).min_by(|a, b| a.1.total_cmp(&b.1)) else { continue };
        // Refine between the neighbouring steps.
        let step = (2.5f64 / 0.6).ln() / (GAIN_STEPS - 1) as f64;
        let fine = (-8..=8).map(|j| best * (step * f64::from(j) / 8.0).exp()).map(|m| (m, error(m))).min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((m, _)) = fine else { continue };
        let total = g0 * m;
        // Only where the central table has seen that colour: elsewhere it extrapolates.
        let seen = support.get(cube(d.map(|c| encode(c * total)))).is_some_and(|n| *n >= 4);
        // A cell pinned at either end of the search says nothing about the lens.
        if seen && m > gains.first().copied().unwrap_or(0.6) * 1.01 && m < gains.last().copied().unwrap_or(2.5) * 0.99 && total.is_finite() {
            let band = (((r - 0.3) / 0.1) as usize).min(6);
            if let Some(v) = bands.get_mut(band) {
                v.push(total);
            }
        }
    }
    // Weighted least squares of g − 1 = a·r² + b·r⁴ over the bands with enough cells; a·r² alone
    // when no band reaches the corners (the r⁴ term would extrapolate there unchecked).
    let (mut s44, mut s46, mut s66, mut b4, mut b6, mut used, mut reach) = (0.0, 0.0, 0.0, 0.0, 0.0, 0usize, 0.0f64);
    for (k, v) in bands.iter_mut().enumerate() {
        if v.len() < 30 {
            continue;
        }
        let n = v.len() as f64;
        let g = median(v) - 1.0;
        let r = 0.35 + 0.1 * k as f64;
        let (r2, r4) = (r * r, r.powi(4));
        s44 += n * r2 * r2;
        s46 += n * r2 * r4;
        s66 += n * r4 * r4;
        b4 += n * r2 * g;
        b6 += n * r4 * g;
        used += 1;
        reach = reach.max(r);
    }
    if used < 2 || s44 <= 0.0 {
        return None;
    }
    let det = s44 * s66 - s46 * s46;
    let (a, b) = if used >= 3 && reach >= 0.75 && det.is_finite() && det.abs() > 1e-12 {
        ((b4 * s66 - b6 * s46) / det, (s44 * b6 - s46 * b4) / det)
    } else {
        (b4 / s44, 0.0)
    };
    let gain = |r: f64| 1.0 + a * r * r + b * r.powi(4);
    let plausible = (0..=20).map(|i| gain(f64::from(i) / 20.0)).all(|g| g.is_finite() && (0.8..=3.0).contains(&g));
    let visible = (0..=20).map(|i| (gain(f64::from(i) / 20.0) - 1.0).abs()).fold(0.0, f64::max) >= 0.02;
    Some(if plausible && visible { [a, b] } else { [0.0, 0.0] })
}

/// The lattice cube (of `(LUT_EDGE − 1)³`) code values `u` fall in.
fn cube(u: [f64; 3]) -> usize {
    let n = LUT_EDGE - 1;
    let [x, y, z] = u.map(|c| ((if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 } * n as f64) as usize).min(n - 1));
    x + n * (y + n * z)
}

/// Fit samples per lattice cube.
fn support(samples: &[Sample]) -> Vec<u32> {
    let n = LUT_EDGE - 1;
    let mut count = vec![0u32; n * n * n];
    for s in samples {
        if let Some(c) = count.get_mut(cube(s.input)) {
            *c = c.saturating_add(1);
        }
    }
    count
}

/// One data point of the table fit.
struct Sample {
    /// Lattice nodes the input touches and their trilinear weights.
    nodes: [u32; 8],
    weights: [f32; 8],
    input: [f64; 3],
    target: [f64; 3],
    weight: f64,
}

fn sample(input: [f64; 3], target: [f64; 3], weight: f64) -> Sample {
    let n = LUT_EDGE;
    let top = (n - 1) as f64;
    let mut base = [0usize; 3];
    let mut frac = [0f64; 3];
    for k in 0..3 {
        let p = input[k].clamp(0.0, 1.0) * top;
        let i = (p as usize).min(n - 2);
        base[k] = i;
        frac[k] = p - i as f64;
    }
    let mut nodes = [0u32; 8];
    let mut weights = [0f32; 8];
    for corner in 0..8usize {
        let (dx, dy, dz) = (corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let w =
            (if dx == 1 { frac[0] } else { 1.0 - frac[0] }) * (if dy == 1 { frac[1] } else { 1.0 - frac[1] }) * (if dz == 1 { frac[2] } else { 1.0 - frac[2] });
        nodes[corner] = ((base[0] + dx) + n * ((base[1] + dy) + n * (base[2] + dz))) as u32;
        weights[corner] = w as f32;
    }
    Sample { nodes, weights, input, target, weight }
}

/// `L·x` for the lattice graph Laplacian (6 neighbours, free boundary).
fn laplacian(x: &[f64], out: &mut [f64]) {
    let n = LUT_EDGE;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let i = ix + n * (iy + n * iz);
                let xi = x.get(i).copied().unwrap_or(0.0);
                let mut acc = 0.0;
                let mut neighbour = |j: usize| {
                    acc += xi - x.get(j).copied().unwrap_or(xi);
                };
                if ix > 0 {
                    neighbour(i - 1);
                }
                if ix + 1 < n {
                    neighbour(i + 1);
                }
                if iy > 0 {
                    neighbour(i - n);
                }
                if iy + 1 < n {
                    neighbour(i + n);
                }
                if iz > 0 {
                    neighbour(i - n * n);
                }
                if iz + 1 < n {
                    neighbour(i + n * n);
                }
                if let Some(o) = out.get_mut(i) {
                    *o = acc;
                }
            }
        }
    }
}

/// The lattice node count each node is connected to.
fn degree(i: usize) -> f64 {
    let n = LUT_EDGE;
    let (ix, iy, iz) = (i % n, (i / n) % n, i / (n * n));
    [ix, iy, iz].iter().map(|&c| f64::from(u8::from(c > 0) + u8::from(c + 1 < n))).sum()
}

/// Fits one output channel of the table: minimises Σ wᵢ (L(uᵢ) − tᵢ)² + λ‖∇²L‖² + ε‖L − id‖²
/// with Jacobi-preconditioned conjugate gradients.
fn fit_channel(samples: &[Sample], channel: usize, smooth: f64, anchor: f64, diag: &[f64], iterations: usize) -> Vec<f64> {
    let n = LUT_EDGE;
    let count = n * n * n;
    let identity = |i: usize| {
        let c = [i % n, (i / n) % n, i / (n * n)];
        c.get(channel).copied().unwrap_or(0) as f64 / (n - 1) as f64
    };
    let apply = |x: &[f64], out: &mut [f64]| {
        out.iter_mut().for_each(|v| *v = 0.0);
        for s in samples {
            let mut v = 0.0;
            for (j, w) in s.nodes.iter().zip(s.weights) {
                v += f64::from(w) * x.get(*j as usize).copied().unwrap_or(0.0);
            }
            v *= s.weight;
            for (j, w) in s.nodes.iter().zip(s.weights) {
                if let Some(o) = out.get_mut(*j as usize) {
                    *o += v * f64::from(w);
                }
            }
        }
        let mut l1 = vec![0.0; count];
        let mut l2 = vec![0.0; count];
        laplacian(x, &mut l1);
        laplacian(&l1, &mut l2);
        for i in 0..count {
            if let (Some(o), Some(l), Some(xi)) = (out.get_mut(i), l2.get(i), x.get(i)) {
                *o += smooth * l + anchor * xi;
            }
        }
    };
    let mut b = vec![0.0; count];
    for s in samples {
        let t = s.target.get(channel).copied().unwrap_or(0.0) * s.weight;
        for (j, w) in s.nodes.iter().zip(s.weights) {
            if let Some(o) = b.get_mut(*j as usize) {
                *o += t * f64::from(w);
            }
        }
    }
    for (i, v) in b.iter_mut().enumerate() {
        *v += anchor * identity(i);
    }
    let mut x: Vec<f64> = (0..count).map(identity).collect();
    let mut ax = vec![0.0; count];
    apply(&x, &mut ax);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let precondition = |r: &[f64]| -> Vec<f64> { r.iter().zip(diag).map(|(r, d)| if *d > 0.0 { r / d } else { *r }).collect() };
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    let mut ap = vec![0.0; count];
    for _ in 0..iterations {
        apply(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if !pap.is_finite() || pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for i in 0..count {
            if let (Some(xi), Some(ri), Some(pi), Some(api)) = (x.get_mut(i), r.get_mut(i), p.get(i), ap.get(i)) {
                *xi += alpha * pi;
                *ri -= alpha * api;
            }
        }
        if r.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-7 * b_norm {
            break;
        }
        z = precondition(&r);
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        if !rz_new.is_finite() || rz <= 0.0 {
            break;
        }
        let beta = rz_new / rz;
        rz = rz_new;
        for (pi, zi) in p.iter_mut().zip(&z) {
            *pi = zi + beta * *pi;
        }
    }
    x
}

/// The fit's data: per cell whose radius `keep` accepts, the raw (corrected by `vignette`) and the
/// JPEG as gamma-1.8 code values, weighted; split into (fit, held-out check) one cell in three.
fn samples(
    paired: &[Option<[f64; 3]>],
    refs: &[Option<RefCell>],
    geo: &Geometry,
    vignette: [f64; 2],
    keep: impl Fn(f64) -> bool,
) -> (Vec<Sample>, Vec<Sample>) {
    let gain = |r: f64| 1.0 + vignette[0] * r * r + vignette[1] * r.powi(4);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for i in 0..refs.len() {
        let r = geo.radius(i);
        if !keep(r) {
            continue;
        }
        let (Some(Some(d)), Some(Some(rc))) = (paired.get(i), refs.get(i)) else { continue };
        let g = gain(r);
        let input = d.map(|c| encode(c * g));
        let target = rc.rgb.map(encode);
        // A channel at 255 is still the camera's rendering (saturated colours clip in sRGB); only
        // its information beyond is lost, so clipped cells count a little less, not much less.
        let weight = rc.edge_weight * (1.0 - 0.3 * rc.clipped.clamp(0.0, 1.0)) * if rc.lum < 0.002 { 0.3 } else { 1.0 };
        if !(weight.is_finite() && weight > 1e-3) || input.iter().chain(&target).any(|v| !v.is_finite()) {
            continue;
        }
        let (x, y) = (i % geo.gw, i / geo.gw);
        let s = sample(input, target, weight);
        if (x + 2 * y) % 3 == 0 {
            if rc.clipped < 0.01 && rc.edge_weight > 0.2 {
                test.push(s);
            }
        } else {
            train.push(s);
        }
    }
    (train, test)
}

/// The lattice fitted to `train`, three channels in parallel.
fn fit_table(train: &[Sample], iterations: usize) -> Result<Vec<[f32; 3]>, LookRejected> {
    let count = LUT_EDGE * LUT_EDGE * LUT_EDGE;
    let mass = train.iter().map(|s| s.weight).sum::<f64>() / count as f64;
    if !(mass.is_finite() && mass > 0.0) {
        return reject("no usable cells to fit");
    }
    let (smooth, anchor) = (SMOOTHNESS * mass, ANCHOR * mass);
    let mut diag = vec![0.0; count];
    for s in train {
        for (j, w) in s.nodes.iter().zip(s.weights) {
            if let Some(d) = diag.get_mut(*j as usize) {
                *d += s.weight * f64::from(w) * f64::from(w);
            }
        }
    }
    for (i, d) in diag.iter_mut().enumerate() {
        let k = degree(i);
        *d += smooth * (k * k + k) + anchor;
    }
    let channels = par::map(3, |c| fit_channel(train, c, smooth, anchor, &diag, iterations));
    let nodes: Vec<[f32; 3]> =
        (0..count).map(|i| [0, 1, 2].map(|c| channels.get(c).and_then(|ch| ch.get(i)).copied().unwrap_or(0.0).clamp(0.0, 1.0) as f32)).collect();
    if nodes.iter().flatten().any(|v| !v.is_finite()) {
        return reject("the table fit did not converge");
    }
    Ok(nodes)
}

/// Fits the camera's rendering of `dev` (a neutral development of calibrated raw data: upright
/// 16-bit ProPhoto code values) to its `reference` JPEG (upright 8-bit sRGB, any size of the
/// same shape).
pub fn fit_rendering(dev: &Developed, reference: &Reference) -> Result<CameraRendering, LookRejected> {
    let (w, h) = (dev.width as usize, dev.height as usize);
    if w < 64 || h < 64 || w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(dev.rgb.len()) {
        return reject("the developed image is too small or its buffer does not match its size");
    }
    let (rw, rh) = (reference.width as usize, reference.height as usize);
    if rw < 32 || rh < 32 || rw.checked_mul(rh).and_then(|n| n.checked_mul(3)) != Some(reference.rgb.len()) {
        return reject("the reference image is too small or its buffer does not match its size");
    }
    let aspect = (rw as f64 / rh as f64) / (w as f64 / h as f64);
    if !(aspect.is_finite() && aspect.ln().abs() <= 0.03) {
        return reject(format!("the reference ({rw}x{rh}) does not have the developed frame's shape ({w}x{h})"));
    }
    let long = w.max(h);
    let edge = GRID_EDGE.min(long / 4).min(rw.max(rh) / 2);
    let (gw, gh) = ((w * edge).div_ceil(long).max(1), (h * edge).div_ceil(long).max(1));
    if gw < 16 || gh < 16 {
        return reject("the image is too small to fit");
    }
    let lin = decode_table();
    let devs = dev_grid(dev, &lin, gw, gh);
    let refs = ref_grid(reference, gw, gh);
    let geo = Geometry::new(gw, gh);
    let (alignment, correlation) = align(&devs, &refs, &geo);
    if !correlation.is_finite() || correlation < 0.7 {
        return reject(format!("the reference does not show the same picture (lightness correlation {correlation:.2})"));
    }
    let paired: Vec<Option<[f64; 3]>> = (0..gw * gh).map(|i| sample_aligned(&devs, &geo, i, alignment)).collect();
    // Vignetting first: tables fitted on the central cells (corrected by the estimate so far),
    // read off the outer cells, three times.
    let mut vignette = [0.0, 0.0];
    for _ in 0..3 {
        let (centre, _) = samples(&paired, &refs, &geo, vignette, |r| r < 0.55);
        if centre.len() < 1500 {
            break;
        }
        let seen = support(&centre);
        let probe = CameraRendering { nodes: fit_table(&centre, PROBE_ITERATIONS)?, vignette, alignment, delta_e: f64::NAN, delta_e_neutral: f64::NAN };
        match estimate_vignette(&probe, &seen, &paired, &refs, &geo) {
            Some(v) => vignette = v,
            None => break,
        }
    }

    let (train, test) = samples(&paired, &refs, &geo, vignette, |_| true);
    if train.len() < 2000 || test.len() < 500 {
        return reject(format!("too few usable cells ({} to fit, {} to check)", train.len(), test.len()));
    }
    let mut rendering = CameraRendering { nodes: fit_table(&train, MAX_ITERATIONS)?, vignette, alignment, delta_e: f64::NAN, delta_e_neutral: f64::NAN };

    let lab = Lab::new();
    let (mut before, mut after) = (0.0, 0.0);
    for s in &test {
        before += lab.delta_e(s.input, s.target);
        let out = rendering.eval(s.input.map(|v| v as f32));
        after += lab.delta_e(out.map(f64::from), s.target);
    }
    let n = test.len() as f64;
    (rendering.delta_e_neutral, rendering.delta_e) = (before / n, after / n);
    if !(rendering.delta_e.is_finite() && rendering.delta_e <= 0.85 * rendering.delta_e_neutral) {
        return reject(format!("the learned rendering does not clearly help (mean ΔE {:.1}, {:.1} neutral)", rendering.delta_e, rendering.delta_e_neutral));
    }
    Ok(rendering)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawFormat;
    use crate::develop::RawInfo;

    fn linear_to_srgb(v: f64) -> f64 {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
    }

    fn developed(w: usize, h: usize, px: impl Fn(usize, usize) -> [f64; 3]) -> Developed {
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                rgb.extend(px(x, y).map(|c| (encode(c) * 65535.0 + 0.5) as u16));
            }
        }
        Developed {
            width: w as u32,
            height: h as u32,
            rgb,
            info: RawInfo {
                format: RawFormat::Dng,
                make: None,
                model: None,
                sensor_width: w,
                sensor_height: h,
                cfa: None,
                wb_multipliers: [1.0; 3],
                orientation: 1,
                baseline_exposure: 0.0,
            },
            warnings: Vec::new(),
        }
    }

    /// A smooth scene of moderate colours (inside sRGB once rendered), linear ProPhoto.
    fn scene(x: usize, y: usize, w: usize, h: usize) -> [f64; 3] {
        let (u, v) = (x as f64 / w as f64, y as f64 / h as f64);
        let base = 0.04 + 0.5 * u;
        let tint = [0.6 + 0.4 * (6.0 * v).sin().abs(), 0.7 + 0.3 * (4.0 * u + 1.0).cos().abs(), 0.6 + 0.4 * (5.0 * v + 2.0 * u).sin().abs()];
        tint.map(|t| base * t)
    }

    /// A "camera" rendering: more contrast and saturation, applied to code values.
    fn camera(code: [f64; 3]) -> [f64; 3] {
        let mean = (code[0] + code[1] + code[2]) / 3.0;
        code.map(|c| {
            let sat = mean + 1.3 * (c - mean);
            let s = sat.clamp(0.0, 1.0);
            (s * s * (3.0 - 2.0 * s) * 0.6 + 0.4 * s).clamp(0.0, 1.0)
        })
    }

    fn reference_of(dev: &Developed, render: impl Fn([f64; 3]) -> [f64; 3], vignette: impl Fn(f64) -> f64) -> Vec<u8> {
        let (w, h) = (dev.width as usize, dev.height as usize);
        let from_pro = color::invert(&color::mul(&color::xyz_d50_to_prophoto(), &color::srgb_to_xyz_d50())).unwrap();
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let hd = cx.hypot(cy);
        let mut out = Vec::with_capacity(w * h * 3);
        for (i, px) in dev.rgb.as_chunks::<3>().0.iter().enumerate() {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            let g = vignette((x - cx).hypot(y - cy) / hd);
            let code = [0, 1, 2].map(|k| encode((f64::from(px[k]) / 65535.0).powf(GAMMA) * g));
            let lin = render(code).map(|c| c.powf(GAMMA));
            let srgb = color::apply(&from_pro, lin);
            out.extend(srgb.map(|c| (linear_to_srgb(c) * 255.0 + 0.5) as u8));
        }
        out
    }

    #[test]
    fn learns_a_known_rendering() {
        let (w, h) = (480, 320);
        let dev = developed(w, h, |x, y| scene(x, y, w, h));
        let rgb = reference_of(&dev, camera, |_| 1.0);
        let r = fit_rendering(&dev, &Reference { width: w as u32, height: h as u32, rgb: &rgb }).unwrap();
        assert!(r.delta_e < 2.0, "{r:?}");
        assert!(r.delta_e < 0.3 * r.delta_e_neutral, "{} vs {}", r.delta_e, r.delta_e_neutral);
        assert_eq!(r.vignette, [0.0, 0.0], "no vignetting in the reference");
        let mut out = dev.clone();
        r.apply(&mut out).unwrap();
        let lab = Lab::new();
        let mut sum = 0.0;
        for (a, b) in out.rgb.as_chunks::<3>().0.iter().zip(dev.rgb.as_chunks::<3>().0) {
            let want = camera([0, 1, 2].map(|k| f64::from(b[k]) / 65535.0));
            sum += lab.delta_e([0, 1, 2].map(|k| f64::from(a[k]) / 65535.0), want);
        }
        let mean = sum / (w * h) as f64;
        assert!(mean < 2.0, "applied ΔE {mean}");
    }

    /// Patches of a few colours repeated over the whole frame, each with a gentle ramp, so every
    /// colour shows near the centre and in the corners alike.
    fn patches(x: usize, y: usize) -> [f64; 3] {
        const PALETTE: [[f64; 3]; 8] = [
            [0.30, 0.12, 0.08],
            [0.10, 0.22, 0.09],
            [0.07, 0.10, 0.28],
            [0.25, 0.24, 0.10],
            [0.20, 0.20, 0.20],
            [0.06, 0.06, 0.06],
            [0.35, 0.30, 0.28],
            [0.12, 0.20, 0.24],
        ];
        let tile = (x / 24 + 3 * (y / 24)) % PALETTE.len();
        let ramp = 0.6 + 0.8 * ((x % 24) as f64 / 24.0);
        PALETTE.get(tile).copied().unwrap_or([0.2; 3]).map(|c| c * ramp)
    }

    #[test]
    fn recovers_the_cameras_vignetting_correction() {
        let (w, h) = (480, 320);
        // The raw is darker toward the corners; the camera's JPEG is not.
        let dev = developed(w, h, |x, y| {
            let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
            let r = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy) / cx.hypot(cy);
            patches(x, y).map(|c| c / (1.0 + 0.5 * r * r))
        });
        let rgb = reference_of(&dev, camera, |r| 1.0 + 0.5 * r * r);
        let r = fit_rendering(&dev, &Reference { width: w as u32, height: h as u32, rgb: &rgb }).unwrap();
        let g = 1.0 + r.vignette[0] + r.vignette[1];
        assert!((g - 1.5).abs() < 0.12, "corner gain {g} from {:?}", r.vignette);
        assert!(r.delta_e < 2.5, "{r:?}");
    }

    #[test]
    fn hostile_sizes_are_errors_not_panics() {
        let dev = developed(100, 80, |x, y| scene(x, y, 100, 80));
        let rgb = vec![128u8; 100 * 80 * 3];
        let ok = Reference { width: 100, height: 80, rgb: &rgb };
        assert!(fit_rendering(&dev, &Reference { width: 100, height: 80, rgb: &rgb[..10] }).is_err());
        assert!(fit_rendering(&dev, &Reference { width: 80, height: 100, rgb: &rgb }).is_err(), "wrong shape");
        assert!(fit_rendering(&dev, &Reference { width: 0, height: 0, rgb: &[] }).is_err());
        let mut broken = dev.clone();
        broken.rgb.truncate(7);
        assert!(fit_rendering(&broken, &ok).is_err());
        let tiny = developed(8, 8, |_, _| [0.2; 3]);
        assert!(fit_rendering(&tiny, &Reference { width: 8, height: 8, rgb: &rgb[..192] }).is_err());
        // A flat grey reference does not show the picture: rejected rather than a grey table.
        let err = fit_rendering(&dev, &ok).unwrap_err();
        assert!(err.0.contains("same picture"), "{err}");
        // Applying to a mismatched buffer fails cleanly.
        let r = CameraRendering { nodes: vec![[0.5; 3]; LUT_EDGE.pow(3)], vignette: [0.0; 2], alignment: [1.0, 0.0], delta_e: 0.0, delta_e_neutral: 0.0 };
        assert!(r.apply(&mut broken).is_err());
        let incomplete = CameraRendering { nodes: Vec::new(), ..r };
        assert!(incomplete.apply(&mut dev.clone()).is_err());
    }
}
