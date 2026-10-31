//! Inline hook and the memory probes `libysmteam.so` uses beside KittyMemory.
//!
//! KittyMemory (`mem_write`, `Patch`, `syscall_mem_read`) already covers byte
//! patches and `process_vm_readv`. The mod library additionally:
//!
//! * plants a 16-byte absolute branch (`LDR X17, #8; BR X17; <addr>`) and runs
//!   the stolen instructions from an executable trampoline;
//! * asks `mincore` which pages of a mapping are resident;
//! * refuses a read or write that crosses one `/proc/self/maps` entry.
//!
//! [`syscall_mem_read`](crate::syscall_mem_read) remains the `process_vm_readv`
//! path on Android. This module does not call into the game.

use std::fmt;
use std::sync::Mutex;

use crate::reloc;

/// Bytes overwritten at the hook site. Four AArch64 instructions.
pub const PATCH_LEN: usize = 16;

/// Failure while hooking or probing memory.
#[derive(Debug)]
pub enum HookError {
    /// This process is not AArch64, so the absolute-branch detour cannot be planted.
    UnsupportedArch,
    /// `target` was null or not 4-byte aligned, or the byte slice was empty.
    Misaligned,
    /// `mmap` failed. `errno` is the raw OS code.
    Map { errno: i32 },
    /// `mprotect` failed. `errno` is the raw OS code.
    Protect { errno: i32 },
    /// An instruction in the stolen window is PC-relative and has no rewrite.
    Unrelocatable { pc: u64, insn: u32 },
    /// `mincore` failed. `errno` is the raw OS code.
    Mincore { errno: i32 },
    /// No `/proc/self/maps` entry contains the address.
    NotMapped { address: usize },
    /// The range spans more than one maps entry.
    CrossesMapping {
        address: usize,
        length: usize,
        mapping_end: usize,
    },
    /// Reading `/proc/self/maps` failed, or this OS has no such file.
    Maps(String),
}

impl fmt::Display for HookError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HookError::UnsupportedArch => write!(formatter, "inline hook requires aarch64"),
            HookError::Misaligned => {
                write!(formatter, "address is not 4-byte aligned or length is zero")
            }
            HookError::Map { errno } => write!(formatter, "mmap failed: errno {errno}"),
            HookError::Protect { errno } => write!(formatter, "mprotect failed: errno {errno}"),
            HookError::Unrelocatable { pc, insn } => {
                write!(
                    formatter,
                    "cannot relocate instruction {insn:#x} at {pc:#x}"
                )
            }
            HookError::Mincore { errno } => write!(formatter, "mincore failed: errno {errno}"),
            HookError::NotMapped { address } => {
                write!(formatter, "address {address:#x} is not mapped")
            }
            HookError::CrossesMapping {
                address,
                length,
                mapping_end,
            } => write!(
                formatter,
                "range {address:#x} len {length} crosses a mapping ending at {mapping_end:#x}"
            ),
            HookError::Maps(message) => write!(formatter, "maps: {message}"),
        }
    }
}

impl std::error::Error for HookError {}

/// How many pages in `[address, address + length)` are resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageResidency {
    pub pages: usize,
    pub resident: usize,
}

/// One `/proc/self/maps` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapRegion {
    pub start: usize,
    pub end: usize,
    pub permissions: String,
    pub pathname: String,
}

/// A planted detour. Dropping it writes the original bytes back.
///
/// The trampoline allocation is kept for the rest of the process. A thread
/// that is still inside the trampoline when [`InlineHook::unhook`] runs is a
/// race; stop callers before unhooking.
#[derive(Debug)]
pub struct InlineHook {
    target: usize,
    original: [u8; PATCH_LEN],
    trampoline: usize,
    active: bool,
}

impl InlineHook {
    /// Overwrite `target` with a branch to `replacement`.
    ///
    /// # Errors
    ///
    /// Returns [`HookError::UnsupportedArch`] off AArch64, [`HookError::Misaligned`]
    /// when `target` is not 4-byte aligned, [`HookError::Unrelocatable`] when one
    /// of the four stolen instructions cannot be rewritten, and [`HookError::Map`]
    /// or [`HookError::Protect`] when the OS refuses the trampoline or the patch.
    ///
    /// # Safety
    ///
    /// * `target` must be the start of an executable instruction stream that
    ///   stays mapped, and `target + 16` must be readable.
    /// * `replacement` must be an executable function with the same calling
    ///   convention as `target`.
    /// * No other thread may execute `target` while this writes it.
    /// * `X17` is scratch across the detour and inside the trampoline.
    pub unsafe fn install(target: usize, replacement: usize) -> Result<Self, HookError> {
        if cfg!(not(target_arch = "aarch64")) {
            return Err(HookError::UnsupportedArch);
        }
        if target == 0 || target % 4 != 0 || replacement == 0 {
            return Err(HookError::Misaligned);
        }
        let mut original = [0u8; PATCH_LEN];
        // SAFETY: caller guarantees `target` is readable for PATCH_LEN bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(target as *const u8, original.as_mut_ptr(), PATCH_LEN);
        }
        let trampoline = reserve_trampoline()?;
        let mut body = reloc::relocate(&original, target as u64, trampoline as u64)?;
        body.extend_from_slice(&reloc::encode_absolute_branch((target + PATCH_LEN) as u64));
        if body.len() > TRAMPOLINE_SLOT {
            return Err(HookError::Unrelocatable {
                pc: target as u64,
                insn: 0,
            });
        }
        write_executable(trampoline, &body)?;
        let patch = reloc::encode_absolute_branch(replacement as u64);
        write_executable(target, &patch)?;
        Ok(Self {
            target,
            original,
            trampoline,
            active: true,
        })
    }

    /// Address of the relocated original instructions. Call this to reach the unhooked body.
    pub fn trampoline(&self) -> usize {
        self.trampoline
    }

    pub fn original_bytes(&self) -> &[u8; PATCH_LEN] {
        &self.original
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Write the stolen bytes back to `target`.
    ///
    /// # Errors
    ///
    /// Returns [`HookError::Protect`] when the target page cannot be made writable.
    pub fn unhook(&mut self) -> Result<(), HookError> {
        if self.active {
            write_executable(self.target, &self.original)?;
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for InlineHook {
    fn drop(&mut self) {
        let _ = self.unhook();
    }
}

/// Plant the same detour as [`InlineHook::install`].
///
/// # Safety
///
/// Same contract as [`InlineHook::install`].
pub unsafe fn hook(target: usize, replacement: usize) -> Result<InlineHook, HookError> {
    // SAFETY: forwarded to the caller.
    unsafe { InlineHook::install(target, replacement) }
}

/// Count resident pages covering `[address, address + length)`.
///
/// # Errors
///
/// Returns [`HookError::Misaligned`] when `length` is zero and [`HookError::Mincore`]
/// when the OS rejects the range (not mapped, or not page-aligned after rounding).
pub fn page_residency(address: usize, length: usize) -> Result<PageResidency, HookError> {
    if length == 0 {
        return Err(HookError::Misaligned);
    }
    let page = page_size();
    let start = address & !(page - 1);
    let end = address.saturating_add(length).saturating_add(page - 1) & !(page - 1);
    let span = end.saturating_sub(start);
    let pages = span / page;
    let mut vector = vec![0u8; pages];
    // SAFETY: `start` is page-aligned, `span` covers exactly `vector.len()` pages,
    // and `mincore` only writes that vector.
    let rc = unsafe {
        libc::mincore(
            start as *mut libc::c_void,
            span,
            vector.as_mut_ptr() as *mut libc::c_char,
        )
    };
    if rc != 0 {
        return Err(HookError::Mincore {
            errno: std::io::Error::last_os_error().raw_os_error().unwrap_or(rc),
        });
    }
    let resident = vector.iter().filter(|byte| *byte & 1 != 0).count();
    Ok(PageResidency { pages, resident })
}

/// Read `/proc/self/maps`.
///
/// # Errors
///
/// Returns [`HookError::Maps`] on operating systems without that file, or when
/// the file cannot be read.
pub fn read_maps() -> Result<Vec<MapRegion>, HookError> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let text = std::fs::read_to_string("/proc/self/maps")
            .map_err(|error| HookError::Maps(error.to_string()))?;
        Ok(parse_maps(&text))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        Err(HookError::Maps("no /proc/self/maps on this os".to_string()))
    }
}

/// Require `[address, address + length)` to sit inside a single maps entry.
///
/// This is the check the mod's KittyMemory fork logs as a hard error when a
/// read or write crosses a mapping boundary.
///
/// # Errors
///
/// See [`HookError::NotMapped`], [`HookError::CrossesMapping`], and [`read_maps`].
pub fn require_single_mapping(address: usize, length: usize) -> Result<MapRegion, HookError> {
    if length == 0 {
        return Err(HookError::Misaligned);
    }
    let regions = read_maps()?;
    let region = regions
        .into_iter()
        .find(|region| address >= region.start && address < region.end)
        .ok_or(HookError::NotMapped { address })?;
    let range_end = address.saturating_add(length);
    if range_end > region.end {
        return Err(HookError::CrossesMapping {
            address,
            length,
            mapping_end: region.end,
        });
    }
    Ok(region)
}

pub(crate) fn parse_maps(text: &str) -> Vec<MapRegion> {
    let mut regions = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(span) = parts.next() else { continue };
        let Some((start_hex, end_hex)) = span.split_once('-') else {
            continue;
        };
        let Ok(start) = usize::from_str_radix(start_hex, 16) else {
            continue;
        };
        let Ok(end) = usize::from_str_radix(end_hex, 16) else {
            continue;
        };
        let Some(permissions) = parts.next() else {
            continue;
        };
        let _offset = parts.next();
        let _device = parts.next();
        let _inode = parts.next();
        let pathname = parts.collect::<Vec<_>>().join(" ");
        regions.push(MapRegion {
            start,
            end,
            permissions: permissions.to_string(),
            pathname,
        });
    }
    regions
}

const TRAMPOLINE_SLOT: usize = 256;

struct ExecPool {
    pages: Vec<ExecPage>,
}

struct ExecPage {
    base: usize,
    length: usize,
    cursor: usize,
}

fn pool() -> &'static Mutex<ExecPool> {
    static POOL: Mutex<ExecPool> = Mutex::new(ExecPool { pages: Vec::new() });
    &POOL
}
fn reserve_trampoline() -> Result<usize, HookError> {
    let mut pool = pool().lock().expect("hook pool poisoned");
    if let Some(page) = pool
        .pages
        .iter_mut()
        .rev()
        .find(|page| page.cursor + TRAMPOLINE_SLOT <= page.length)
    {
        let slot = page.base + page.cursor;
        page.cursor += TRAMPOLINE_SLOT;
        return Ok(slot);
    }
    let length = page_size().max(TRAMPOLINE_SLOT);
    let base = map_executable(length)?;
    let slot = base;
    pool.pages.push(ExecPage {
        base,
        length,
        cursor: TRAMPOLINE_SLOT,
    });
    Ok(slot)
}

fn page_size() -> usize {
    // SAFETY: `sysconf` with `_SC_PAGESIZE` does not touch caller memory.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size <= 0 {
        4096
    } else {
        size as usize
    }
}

fn map_executable(length: usize) -> Result<usize, HookError> {
    let mut flags = libc::MAP_PRIVATE | libc::MAP_ANON;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        flags |= libc::MAP_JIT;
    }
    // SAFETY: anonymous mapping; a null address lets the kernel choose the base.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            flags,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(-1);
        // Some Android kernels reject RWX. Map RW and switch to RX after the write.
        let readable = libc::PROT_READ | libc::PROT_WRITE;
        // SAFETY: same anonymous mapping, writable only.
        let base = unsafe { libc::mmap(std::ptr::null_mut(), length, readable, flags, -1, 0) };
        if base == libc::MAP_FAILED {
            return Err(HookError::Map { errno });
        }
        return Ok(base as usize);
    }
    Ok(base as usize)
}

fn write_executable(address: usize, bytes: &[u8]) -> Result<(), HookError> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        // Apple Silicon only executes writes to MAP_JIT pages, and only while
        // this thread's JIT write protect is off. Game hooks run on Android.
        begin_jit_write();
        // SAFETY: `address` is a MAP_JIT page this module just allocated, or a
        // caller target on a platform where the toggle is a no-op outside macOS.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len());
        }
        clear_instruction_cache(address, bytes.len());
        end_jit_write();
        return Ok(());
    }
    protect(address, bytes.len(), libc::PROT_READ | libc::PROT_WRITE).or_else(|_| {
        protect(
            address,
            bytes.len(),
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        )
    })?;
    // SAFETY: `protect` just made this span writable for `bytes.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len());
    }
    clear_instruction_cache(address, bytes.len());
    protect(address, bytes.len(), libc::PROT_READ | libc::PROT_EXEC)?;
    Ok(())
}

fn protect(address: usize, length: usize, prot: i32) -> Result<(), HookError> {
    let page = page_size();
    let start = address & !(page - 1);
    let end = address.saturating_add(length).saturating_add(page - 1) & !(page - 1);
    // SAFETY: `start`/`end` are page bounds covering `address`.
    let rc = unsafe { libc::mprotect(start as *mut libc::c_void, end - start, prot) };
    if rc != 0 {
        return Err(HookError::Protect {
            errno: std::io::Error::last_os_error().raw_os_error().unwrap_or(rc),
        });
    }
    Ok(())
}

fn begin_jit_write() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe {
        // SAFETY: toggles the calling thread's MAP_JIT write window. Paired with `end_jit_write`.
        pthread_jit_write_protect_np(0);
    }
}

fn end_jit_write() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe {
        // SAFETY: closes the write window opened by `begin_jit_write` on this thread.
        pthread_jit_write_protect_np(1);
    }
}

/// Drop instructions from the data cache and invalidate the instruction cache.
pub fn clear_instruction_cache(address: usize, length: usize) {
    if length == 0 {
        return;
    }
    extern "C" {
        fn __clear_cache(start: *mut libc::c_void, end: *mut libc::c_void);
    }
    // SAFETY: `__clear_cache` only maintains caches for this half-open byte range.
    unsafe {
        __clear_cache(
            address as *mut libc::c_void,
            (address + length) as *mut libc::c_void,
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_parser_keeps_the_pathname() {
        let text = "00400000-00401000 r-xp 00000000 00:00 1 /data/app/libil2cpp.so\n\
                    00401000-00402000 rw-p 00000000 00:00 0\n";
        let regions = parse_maps(text);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].pathname, "/data/app/libil2cpp.so");
        assert!(regions[1].pathname.is_empty());
        assert_eq!(regions[0].end, 0x0040_1000);
    }

    #[test]
    fn single_mapping_rejects_a_cross() {
        let text = "1000-2000 r--p 0 00:00 0 metadata\n2000-3000 r--p 0 00:00 0 other\n";
        let regions = parse_maps(text);
        assert!(regions[0].start == 0x1000);
        let owned = regions
            .into_iter()
            .find(|region| 0x1ff0 >= region.start && 0x1ff0 < region.end)
            .unwrap();
        assert!(0x1ff0 + 0x20 > owned.end);
    }

    #[test]
    fn residency_of_a_touched_page() {
        let length = page_size();
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED);
        unsafe { std::ptr::write_bytes(base as *mut u8, 1, length) };
        let residency = page_residency(base as usize, length).unwrap();
        assert_eq!(residency.pages, 1);
        assert_eq!(residency.resident, 1);
        unsafe { libc::munmap(base, length) };
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn detour_runs_replacement_and_trampoline_runs_original() {
        let page = map_executable(page_size()).unwrap();
        // mov x0, #40; nop; nop; nop; add x0, x0, #2; ret.
        // The add is past the 16-byte patch, so the trampoline must run the stolen mov and jump back.
        let program = [
            0xD280_0500u32,
            0xD503_201F,
            0xD503_201F,
            0xD503_201F,
            0x9100_0800,
            0xD65F_03C0,
        ];
        let mut bytes = Vec::new();
        for word in program {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        write_executable(page, &bytes).unwrap();
        let original: extern "C" fn() -> u64 = unsafe { std::mem::transmute(page) };
        assert_eq!(original(), 42);
        extern "C" fn replacement() -> u64 {
            7
        }
        let hooked = unsafe { hook(page, replacement as *const () as usize) }.unwrap();
        assert_eq!(original(), 7);
        let via_trampoline: extern "C" fn() -> u64 =
            unsafe { std::mem::transmute(hooked.trampoline()) };
        assert_eq!(via_trampoline(), 42);
        drop(hooked);
        assert_eq!(original(), 42);
    }
}
