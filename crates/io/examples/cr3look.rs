//! Measure how close a CR3's raw data, developed from a DNG copy of it, comes to the camera
//! maker's own rendering: each `NAME` needs `NAME.dng` (the CR3 converted by Adobe DNG
//! Converter), `NAME.CR3` (for its embedded camera JPEG, the colour fit's reference) and
//! `NAME.TIF` (the same photo developed by Canon Digital Photo Professional, the independent
//! reference every variant is measured against). Prints mean CIE76 ΔE (sRGB-clipped Lab D50) on
//! a 300-px grid over the whole frame, the central 60% (lens distortion and vignetting differ
//! least there) and the reds, for the CR3's embedded JPEG, the DNG's neutral rendering, the
//! DNG developed with a look fitted to the embedded JPEG (`fit_look`: one matrix and one tone
//! curve) and the DNG's calibrated development with a rendering learned from the embedded JPEG
//! (`fit_rendering`: a 3D table and the camera's vignetting correction).
//!
//! ```sh
//! cargo run --release -p photocraft-io --example cr3look -- --dng DIR --cr3 DIR --tif DIR [--png DIR] NAME...
//! ```

use std::time::Instant;

use photocraft_cms::{Builtin, Intent, Transform};
use photocraft_codecs::{self as codecs, ChannelLayout, EncodeOptions, Format, Image, SampleType};
use photocraft_raw::{DevelopOptions, develop_sensor};

/// Long edge of the comparison grid.
const GRID: usize = 300;

/// Resamples interleaved RGB to exactly `tw × th` by box averaging over the matching source area.
fn resample(px: &[f32], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tw * th * 3];
    for ty in 0..th {
        let (y0, y1) = (ty * h / th, ((ty + 1) * h / th).max(ty * h / th + 1));
        for tx in 0..tw {
            let (x0, x1) = (tx * w / tw, ((tx + 1) * w / tw).max(tx * w / tw + 1));
            let mut s = [0.0f32; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    for k in 0..3 {
                        s[k] += px[(y * w + x) * 3 + k];
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f32;
            for k in 0..3 {
                out[(ty * tw + tx) * 3 + k] = s[k] / n;
            }
        }
    }
    out
}

fn lin(v: f32) -> f64 {
    let v = f64::from(v);
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// Lab (D50) of display sRGB 0..1 (Bradford-adapted sRGB → XYZ D50).
fn lab(p: &[f32]) -> [f64; 3] {
    let m = [[0.4360747, 0.3850649, 0.1430804], [0.2225045, 0.7168786, 0.0606169], [0.0139322, 0.0971045, 0.7141733]];
    let l = [lin(p[0]), lin(p[1]), lin(p[2])];
    let xyz: [f64; 3] = std::array::from_fn(|i| m[i][0] * l[0] + m[i][1] * l[1] + m[i][2] * l[2]);
    let w = [0.9642, 1.0, 0.8249];
    let f = |t: f64| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
    let [fx, fy, fz] = [0, 1, 2].map(|k| f(xyz[k] / w[k]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Mean ΔE of `a` against the reference `r` (both `gw × gh` grids): whole frame, central 60%,
/// and the cells whose reference colour is a saturated red (hue 330°–45°, chroma above 30).
fn scores(a: &[f32], r: &[f32], gw: usize, gh: usize) -> [f64; 3] {
    let (mut all, mut center, mut reds) = ((0.0, 0usize), (0.0, 0usize), (0.0, 0usize));
    for y in 0..gh {
        for x in 0..gw {
            let i = (y * gw + x) * 3;
            let (p, q) = (lab(&a[i..i + 3]), lab(&r[i..i + 3]));
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            all = (all.0 + d, all.1 + 1);
            if x * 5 >= gw && x * 5 < gw * 4 && y * 5 >= gh && y * 5 < gh * 4 {
                center = (center.0 + d, center.1 + 1);
            }
            let (chroma, hue) = (q[1].hypot(q[2]), q[2].atan2(q[1]).to_degrees().rem_euclid(360.0));
            if chroma > 30.0 && !(45.0..330.0).contains(&hue) {
                reds = (reds.0 + d, reds.1 + 1);
            }
        }
    }
    [all, center, reds].map(|(s, n)| if n > 0 { s / n as f64 } else { f64::NAN })
}

/// Developed ProPhoto 16-bit → display sRGB 0..1.
fn prophoto_to_srgb(rgb: &[u16]) -> Vec<f32> {
    let t = Transform::new(Builtin::ProPhotoCompat.profile(), Builtin::Srgb.profile(), Intent::RelativeColorimetric, false).expect("transform");
    let mut out = vec![0u16; rgb.len()];
    t.convert_u16(rgb, 3, &mut out, 3, false);
    out.iter().map(|v| f32::from(*v) / 65535.0).collect()
}

/// An image's RGB as 0..1 floats (its values as stored: sRGB here).
fn floats(img: &Image) -> Vec<f32> {
    let img = img.converted(ChannelLayout::Rgb, SampleType::U16);
    img.data().as_chunks::<2>().0.iter().map(|b| f32::from(u16::from_ne_bytes(*b)) / 65535.0).collect()
}

/// The CR3's largest embedded JPEG, upright (its own EXIF orientation, else the raw's).
fn camera_jpeg(cr3: &[u8]) -> Image {
    let p = photocraft_raw::embedded_previews(cr3).into_iter().max_by_key(|p| p.width.max(p.height)).expect("no embedded JPEG");
    let img = codecs::decode_as_with(Format::Jpeg, p.jpeg, &codecs::DecodeOptions { keep_orientation: true, ..Default::default() }).expect("jpeg");
    let own = img.meta.exif.as_deref().map_or(1, codecs::exif_orientation);
    let o = if own != 1 { own } else { photocraft_raw::orientation(cr3).unwrap_or_else(|| codecs::exif_orientation(cr3)) };
    img.oriented(o).expect("orient")
}

/// The preview the import fits with: the smallest of at least 512 px, upright 8-bit RGB.
fn fit_reference(cr3: &[u8]) -> Image {
    let previews = photocraft_raw::embedded_previews(cr3);
    let long = |p: &photocraft_raw::Preview| p.width.max(p.height);
    let p = previews.iter().filter(|p| long(p) >= 512).min_by_key(|p| long(p)).or_else(|| previews.iter().max_by_key(|p| long(p))).expect("no preview");
    let img = codecs::decode_as_with(Format::Jpeg, p.jpeg, &codecs::DecodeOptions { keep_orientation: true, ..Default::default() }).expect("jpeg");
    let own = img.meta.exif.as_deref().map_or(1, codecs::exif_orientation);
    let o = if own != 1 { own } else { photocraft_raw::orientation(cr3).unwrap_or_else(|| codecs::exif_orientation(cr3)) };
    img.oriented(o).expect("orient").converted(ChannelLayout::Rgb, SampleType::U8).into_owned()
}

fn write_png(path: &str, px: &[f32], w: usize, h: usize) {
    let bytes: Vec<u8> = px.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
    let img = Image::from_u8(w as u32, h as u32, ChannelLayout::Rgb, bytes).expect("image");
    std::fs::write(path, codecs::encode(&img, Format::Png, &EncodeOptions::default()).expect("png")).expect("write");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut dng_dir, mut cr3_dir, mut tif_dir, mut png) = (String::new(), String::new(), String::new(), None);
    let mut names = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dng" => dng_dir = args.next().expect("--dng DIR"),
            "--cr3" => cr3_dir = args.next().expect("--cr3 DIR"),
            "--tif" => tif_dir = args.next().expect("--tif DIR"),
            "--png" => png = args.next(),
            _ => names.push(a),
        }
    }
    println!("| foto | prévia JPEG | DNG neutro | DNG + look | DNG + LUT | fit ms | develop ms |");
    println!("|---|---|---|---|---|---|---|");
    let mut totals = [[0.0f64; 3]; 4];
    for name in &names {
        let dng = std::fs::read(format!("{dng_dir}/{name}.dng")).expect("read dng");
        let cr3 = std::fs::read(format!("{cr3_dir}/{name}.CR3")).expect("read cr3");
        let tif = codecs::decode_as(Format::Tiff, &std::fs::read(format!("{tif_dir}/{name}.TIF")).expect("read tif")).expect("tif");
        let (rw, rh) = (tif.width() as usize, tif.height() as usize);
        let (gw, gh) = if rw >= rh { (GRID, (GRID * rh).div_ceil(rw)) } else { ((GRID * rw).div_ceil(rh), GRID) };
        let reference = resample(&floats(&tif), rw, rh, gw, gh);

        let jpeg = camera_jpeg(&cr3);
        let jpeg_grid = resample(&floats(&jpeg), jpeg.width() as usize, jpeg.height() as usize, gw, gh);

        let opts = DevelopOptions::default();
        let sensor = photocraft_raw::decode(&dng, &opts.limits).expect("decode dng");
        let neutral = develop_sensor(&sensor, &opts).expect("develop");
        let (dw, dh) = (neutral.width as usize, neutral.height as usize);
        let neutral_grid = resample(&prophoto_to_srgb(&neutral.rgb), dw, dh, gw, gh);

        let t = Instant::now();
        let fit = photocraft_io::raw::fit_preview_look(&cr3, &sensor, &opts);
        let fit_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let fitted = match &fit {
            Ok(look) => develop_sensor(&sensor, &DevelopOptions { camera_look: Some(look.clone()), ..opts.clone() }).expect("develop"),
            Err(e) => {
                eprintln!("{name}: {e}");
                neutral.clone()
            }
        };
        let dev_ms = t.elapsed().as_secs_f64() * 1e3;
        let fitted_grid = resample(&prophoto_to_srgb(&fitted.rgb), dw, dh, gw, gh);

        // The rendering learned from the camera JPEG on top of the DNG's own calibration.
        let small = fit_reference(&cr3);
        let t = Instant::now();
        let learned = photocraft_raw::fit_rendering(&neutral, &photocraft_raw::Reference { width: small.width(), height: small.height(), rgb: small.data() });
        let lut_fit_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut rendered = neutral.clone();
        let t = Instant::now();
        let note = match &learned {
            Ok(r) => {
                r.apply(&mut rendered).expect("apply");
                format!(
                    "vinheta {:?}, alinhamento {:?}, ΔE na prévia {:.1} → {:.1}",
                    r.vignette.map(|v| (v * 1000.0).round() / 1000.0),
                    r.alignment,
                    r.delta_e_neutral,
                    r.delta_e
                )
            }
            Err(e) => format!("rejeitado: {e}"),
        };
        let lut_apply_ms = t.elapsed().as_secs_f64() * 1e3;
        let rendered_srgb = prophoto_to_srgb(&rendered.rgb);
        let rendered_grid = resample(&rendered_srgb, dw, dh, gw, gh);
        eprintln!("{name}: LUT fit {lut_fit_ms:.0} ms, apply {lut_apply_ms:.0} ms; {note}");

        let s = [
            scores(&jpeg_grid, &reference, gw, gh),
            scores(&neutral_grid, &reference, gw, gh),
            scores(&fitted_grid, &reference, gw, gh),
            scores(&rendered_grid, &reference, gw, gh),
        ];
        for (t, s) in totals.iter_mut().zip(&s) {
            for k in 0..3 {
                t[k] += s[k];
            }
        }
        let cell = |s: &[f64; 3]| format!("{:.1} / {:.1} / {:.1}", s[0], s[1], s[2]);
        println!("| {name} | {} | {} | {} | {} | {fit_ms:.0} | {dev_ms:.0} |", cell(&s[0]), cell(&s[1]), cell(&s[2]), cell(&s[3]));
        if let Some(dir) = &png {
            let (pw, ph) = (dw.min(1200), (dh * dw.min(1200)) / dw);
            write_png(&format!("{dir}/{name}-dpp.png"), &resample(&floats(&tif), rw, rh, pw, ph), pw, ph);
            write_png(&format!("{dir}/{name}-lut.png"), &resample(&rendered_srgb, dw, dh, pw, ph), pw, ph);
        }
    }
    let n = names.len().max(1) as f64;
    let cell = |t: &[f64; 3]| format!("**{:.1} / {:.1} / {:.1}**", t[0] / n, t[1] / n, t[2] / n);
    println!("| média | {} | {} | {} | {} | | |", cell(&totals[0]), cell(&totals[1]), cell(&totals[2]), cell(&totals[3]));
    println!("\nΔE contra o TIFF do DPP: foto inteira / centro 60% / vermelhos.");
}
