//! Raw data `photocraft-raw` cannot decode itself (Canon CR3's CRX compression) through Adobe's
//! free DNG Converter, when it is installed: the raw is written to a private temporary folder,
//! converted to a lossless DNG without a preview (`-c -p0`) and read back; the folder is removed
//! afterwards, whatever happens. The converter is the program `PHOTOCRAFT_DNG_CONVERTER` names,
//! else its default install location on Windows and macOS. A conversion that takes longer than
//! [`TIMEOUT`] is stopped.
//!
//! Native only: the browser build cannot run programs, so it reports the converter as missing.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// The longest a conversion may take.
pub const TIMEOUT: Duration = Duration::from_secs(180);

/// Largest DNG read back (a 100 MP lossless DNG is about 250 MB).
const MAX_DNG_BYTES: u64 = 2 << 30;

/// Where the converter is installed by default.
fn default_locations() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if cfg!(windows) {
        for var in ["ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"] {
            if let Some(dir) = std::env::var_os(var) {
                out.push(PathBuf::from(dir).join("Adobe").join("Adobe DNG Converter").join("Adobe DNG Converter.exe"));
            }
        }
        out.push(PathBuf::from(r"C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe"));
    } else if cfg!(target_os = "macos") {
        out.push(PathBuf::from("/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter"));
    }
    out
}

/// The installed converter: `PHOTOCRAFT_DNG_CONVERTER`, else the first default location that
/// exists. `None` when there is none (always in the browser).
pub fn find() -> Option<PathBuf> {
    if cfg!(target_arch = "wasm32") {
        return None;
    }
    if let Some(p) = std::env::var_os("PHOTOCRAFT_DNG_CONVERTER").filter(|p| !p.is_empty()) {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    default_locations().into_iter().find(|p| p.is_file())
}

/// `raw` (a file with extension `extension`, such as `"CR3"`) as a DNG, through the installed
/// converter. `Err` says why not, in words a user can act on.
pub fn convert(raw: &[u8], extension: &str) -> Result<Vec<u8>, String> {
    let exe = find()
        .ok_or_else(|| "Adobe DNG Converter is not installed (it is free from Adobe; or set PHOTOCRAFT_DNG_CONVERTER to the program's path)".to_string())?;
    convert_with(&exe, raw, extension)
}

/// [`convert`] with an explicit converter program.
#[cfg(not(target_arch = "wasm32"))]
pub fn convert_with(exe: &Path, raw: &[u8], extension: &str) -> Result<Vec<u8>, String> {
    let ext: String = extension.chars().filter(char::is_ascii_alphanumeric).take(8).collect();
    if ext.is_empty() {
        return Err("the raw file's extension is missing".into());
    }
    let dir = TempDir::new()?;
    let input = dir.path.join(format!("raw.{ext}"));
    std::fs::write(&input, raw).map_err(|e| format!("cannot write a temporary copy of the raw file: {e}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("-c").arg("-p0").arg("-d").arg(&dir.path).arg(&input);
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: no console window flashes up for each conversion.
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().map_err(|e| format!("cannot start Adobe DNG Converter ({}): {e}", exe.display()))?;
    let started = std::time::Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < TIMEOUT => std::thread::sleep(Duration::from_millis(25)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("Adobe DNG Converter took longer than {} s and was stopped", TIMEOUT.as_secs()));
            }
            Err(e) => {
                let _ = child.kill();
                return Err(format!("lost track of Adobe DNG Converter: {e}"));
            }
        }
    };
    let output = dir.path.join("raw.dng");
    let size = std::fs::metadata(&output).map(|m| m.len()).map_err(|_| format!("Adobe DNG Converter wrote no DNG ({status})"))?;
    if size == 0 || size > MAX_DNG_BYTES {
        return Err(format!("Adobe DNG Converter wrote an unusable DNG ({size} bytes)"));
    }
    std::fs::read(&output).map_err(|e| format!("cannot read the converted DNG: {e}"))
}

/// [`convert`] with an explicit converter program (the browser runs no programs).
#[cfg(target_arch = "wasm32")]
pub fn convert_with(_exe: &Path, _raw: &[u8], _extension: &str) -> Result<Vec<u8>, String> {
    Err("Adobe DNG Converter cannot run in the browser".into())
}

/// A folder of our own under the system's temporary folder, removed when dropped.
#[cfg(not(target_arch = "wasm32"))]
struct TempDir {
    path: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl TempDir {
    fn new() -> Result<TempDir, String> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let base = std::env::temp_dir();
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let mut last = String::new();
        for _ in 0..8 {
            let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = base.join(format!("photocraft-dng-{}-{nanos:x}-{n}", std::process::id()));
            // `create_dir`, not `create_dir_all`: an existing folder is someone else's.
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(TempDir { path }),
                Err(e) => last = e.to_string(),
            }
        }
        Err(format!("cannot create a temporary folder for the conversion: {last}"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn a_missing_converter_is_an_error_and_leaves_no_folder_behind() {
        let before = temp_folders();
        let err = convert_with(Path::new("this-converter-does-not-exist.exe"), b"not a raw", "CR3").unwrap_err();
        assert!(err.contains("cannot start Adobe DNG Converter"), "{err}");
        assert_eq!(temp_folders(), before, "the temporary folder is removed");
    }

    #[test]
    fn a_bad_extension_is_refused_before_anything_runs() {
        assert!(convert_with(Path::new("x"), b"", "").is_err());
        assert!(convert_with(Path::new("x"), b"", "../..").is_err(), "only letters and digits are kept");
    }

    /// A program that exits without writing a DNG is reported, not trusted.
    #[test]
    fn a_converter_that_writes_nothing_is_reported() {
        let exe = if cfg!(windows) {
            PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_default()).join("System32").join("where.exe")
        } else {
            PathBuf::from("/bin/true")
        };
        if !exe.is_file() {
            return;
        }
        let err = convert_with(&exe, b"not a raw", "CR3").unwrap_err();
        assert!(err.contains("wrote no DNG"), "{err}");
    }

    fn temp_folders() -> usize {
        let prefix = format!("photocraft-dng-{}-", std::process::id());
        std::fs::read_dir(std::env::temp_dir()).map(|d| d.flatten().filter(|e| e.file_name().to_string_lossy().starts_with(&prefix)).count()).unwrap_or(0)
    }

    /// Converts a real CR3 when one is named (`PHOTOCRAFT_TEST_CR3`) and the converter is installed.
    #[test]
    #[ignore = "needs Adobe DNG Converter and a CR3 named by PHOTOCRAFT_TEST_CR3"]
    fn converts_a_real_cr3() {
        let path = std::env::var_os("PHOTOCRAFT_TEST_CR3").expect("PHOTOCRAFT_TEST_CR3");
        let dng = convert(&std::fs::read(path).unwrap(), "CR3").unwrap();
        assert!(photocraft_raw::identify(&dng) == Some(photocraft_raw::RawFormat::Dng));
    }
}
