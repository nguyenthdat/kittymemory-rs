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
    /// SELinux refuses to execute a modified file-backed mapping (`execmod`), which is the case for
    /// every `untrusted_app` process. Inline hooks cannot work there; use a GOT hook instead.
    SelinuxBlocked { context: String },
    /// The range spans more than one maps entry.
    CrossesMapping {
        address: usize,
        length: usize,
        mapping_end: usize,
    },
    /// Reading `/proc/self/maps` failed, or this OS has no such file.
    Maps(String),
    /// The site already carries a detour, so a second patch would corrupt it.
    AlreadyHooked { address: usize },
    /// The planted detour did not read back after the write.
    NotIntact { address: usize },
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
            HookError::AlreadyHooked { address } => {
                write!(formatter, "address {address:#x} already carries a detour")
            }
            HookError::NotIntact { address } => {
                write!(formatter, "the detour at {address:#x} did not land")
            }
            HookError::SelinuxBlocked { context } => write!(
                formatter,
                "selinux ({context}) denies execmod on modified code pages; use a GOT hook"
            ),
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

/// Permission bits of one `/proc/self/maps` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Perms {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Perms {
    /// Parses the 4-character permission column (`"r-xp"`, `"rw-s"`, ...); unknown characters are ignored.
    ///
    /// The share column (the last character, `p` for private or `s` for shared) is
    /// only checked for shape, because `mprotect` has no notion of sharing.
    /// Returns `None` when `perms` is shorter than the column or does not end in a
    /// share flag, i.e. when it cannot be a maps permission column at all.
    pub fn from_maps_str(perms: &str) -> Option<Self> {
        let bytes = perms.as_bytes();
        if bytes.len() < 4 || !matches!(bytes[3], b'p' | b's' | b'-') {
            return None;
        }
        Some(Self {
            read: bytes[0] == b'r',
            write: bytes[1] == b'w',
            execute: bytes[2] == b'x',
        })
    }

    /// `mprotect(2)` flags for these permissions.
    pub fn to_mprotect(self) -> i32 {
        let mut prot = 0;
        if self.read {
            prot |= libc::PROT_READ;
        }
        if self.write {
            prot |= libc::PROT_WRITE;
        }
        if self.execute {
            prot |= libc::PROT_EXEC;
        }
        prot
    }
}

/// How `write_code` makes an executable page writable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Never map a page writable *and* executable at once: `W` replaces `X` for the duration of the
    /// write. Stealthier; a thread executing that page during the window would fault.
    Split,
    /// `RWX` for the duration of the write. Compatible; visible to `/proc/self/maps` scans.
    Rwx,
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
    patch: [u8; PATCH_LEN],
    trampoline: usize,
    restored: Perms,
    active: bool,
}

impl InlineHook {
    /// Overwrite `target` with a branch to `replacement`.
    ///
    /// Refuses a site that already carries a detour (see [`looks_like_detour`]
    /// and the anonymous-target branch check) with [`HookError::AlreadyHooked`],
    /// pins the target page, patches it through [`write_code`] in
    /// [`WriteMode::Split`], and re-reads the site before reporting success.
    ///
    /// # Errors
    ///
    /// Returns [`HookError::UnsupportedArch`] off AArch64, [`HookError::Misaligned`]
    /// when `target` is not 4-byte aligned, [`HookError::AlreadyHooked`] when the
    /// site is already detoured, [`HookError::Unrelocatable`] when one of the four
    /// stolen instructions cannot be rewritten, [`HookError::Map`] or
    /// [`HookError::Protect`] when the OS refuses the trampoline or the patch, and
    /// [`HookError::NotIntact`] when the written detour did not read back.
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
        // Patching a file-backed code page and then executing it is refused by SELinux for app
        // domains; fail with the reason instead of faulting inside the hooked function later.
        if mapping_containing(target)?.2.execute && !execmod_allowed() {
            return Err(HookError::SelinuxBlocked {
                context: selinux_context().unwrap_or_else(|| "unknown".to_owned()),
            });
        }
        let mut original = [0u8; PATCH_LEN];
        // SAFETY: caller guarantees `target` is readable for PATCH_LEN bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(target as *const u8, original.as_mut_ptr(), PATCH_LEN);
        }
        let first = u32::from_le_bytes([original[0], original[1], original[2], original[3]]);
        if looks_like_detour(&original) || branches_outside_file_mapping(first, target) {
            return Err(HookError::AlreadyHooked { address: target });
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
        // Best effort: a resident page cannot be reclaimed and re-read from the
        // file underneath the patch. `pin` reports refusal instead of failing.
        let _ = pin(target, PATCH_LEN);
        let restored = write_code(target, &patch, WriteMode::Split)?;
        let installed = Self {
            target,
            original,
            patch,
            trampoline,
            restored,
            active: true,
        };
        if !installed.is_intact() {
            // `Drop` writes the original bytes back, so a site that did not take
            // the detour is left exactly as it was.
            drop(installed);
            return Err(HookError::NotIntact { address: target });
        }
        Ok(installed)
    }

    /// Address of the relocated original instructions. Call this to reach the unhooked body.
    pub fn trampoline(&self) -> usize {
        self.trampoline
    }

    pub fn original_bytes(&self) -> &[u8; PATCH_LEN] {
        &self.original
    }

    /// Permissions the target page had around the patch (restored afterwards).
    pub fn restored_perms(&self) -> Perms {
        self.restored
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Re-reads the hook site and compares it against the planted detour.
    ///
    /// Returns `false` once [`InlineHook::unhook`] ran, and also when anti-tamper
    /// code (or a page reclaim) replaced any of the 16 planted bytes.
    pub fn is_intact(&self) -> bool {
        if !self.active {
            return false;
        }
        let mut current = [0u8; PATCH_LEN];
        // SAFETY: `target` is the site this hook patched and stays mapped.
        unsafe {
            std::ptr::copy_nonoverlapping(self.target as *const u8, current.as_mut_ptr(), PATCH_LEN);
        }
        current == self.patch
    }

    /// Write the stolen bytes back to `target`.
    ///
    /// # Errors
    ///
    /// Returns [`HookError::Protect`] when the target page cannot be made writable
    /// or its original permissions cannot be restored.
    pub fn unhook(&mut self) -> Result<(), HookError> {
        if self.active {
            write_code(self.target, &self.original, patch_mode())?;
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

/// Current permissions of the mapping containing `address`.
///
/// On Linux and Android the permissions come from `/proc/self/maps`; on macOS
/// they come from `vm_region_64`. Other hosts return [`HookError::Maps`].
///
/// # Errors
///
/// Returns [`HookError::NotMapped`] when no mapping contains `address`, and
/// [`HookError::Maps`] when the OS cannot describe the mapping.
pub fn perms(address: usize) -> Result<Perms, HookError> {
    mapping_containing(address).map(|(_, _, perms)| perms)
}

/// Writes `bytes` into this process through `process_vm_writev(2)`.
///
/// This is the Android-safe path: patching a page with `mprotect` + `memcpy` turns a `MAP_PRIVATE`
/// file mapping into a *modified* mapping, and SELinux then refuses to execute it — the audit log
/// shows `avc: denied { execmod }` and the next call into the patched function dies with
/// `SEGV_ACCERR`. Writing through the kernel's process-memory interface leaves the mapping's
/// permission state untouched, so execution stays allowed. It is also how the reference mod patches
/// `libil2cpp.so`.
///
/// Returns `true` only when every byte was written.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn write_through_process_memory(address: usize, bytes: &[u8]) -> bool {
    use core::ffi::c_int;
    use libc::{c_ulong, c_void, iovec, pid_t, ssize_t};

    unsafe extern "C" {
        fn process_vm_writev(
            pid: pid_t,
            local_iov: *const iovec,
            local_count: c_ulong,
            remote_iov: *const iovec,
            remote_count: c_ulong,
            flags: c_ulong,
        ) -> ssize_t;
    }

    let local = iovec {
        iov_base: bytes.as_ptr() as *mut c_void,
        iov_len: bytes.len(),
    };
    let remote = iovec {
        iov_base: address as *mut c_void,
        iov_len: bytes.len(),
    };
    let _: c_int = 0;
    // SAFETY: both iovecs describe live buffers for the duration of the call; the remote range was
    // validated against `/proc/self/maps` by the caller.
    let written = unsafe {
        process_vm_writev(
            std::process::id() as pid_t,
            &local,
            1,
            &remote,
            1,
            0,
        )
    };
    written == bytes.len() as ssize_t
}

/// This process's SELinux label (`/proc/self/attr/current`), or `None` when SELinux is absent.
pub fn selinux_context() -> Option<String> {
    let raw = std::fs::read("/proc/self/attr/current").ok()?;
    let text = String::from_utf8_lossy(&raw);
    let trimmed = text.trim_matches(char::from(0)).trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Whether this process may execute a *modified* file-backed code page.
///
/// Android denies `execmod` to `untrusted_app*` (and to most other app domains) while SELinux is
/// enforcing, which makes any inline hook on a library's `.text` fault with `SEGV_ACCERR` on the
/// first call after the patch. Root, `priv_app` and permissive domains are allowed.
pub fn execmod_allowed() -> bool {
    let Some(context) = selinux_context() else {
        // No SELinux (desktop, or Android with SELinux off): the kernel has no such check.
        return true;
    };
    let enforcing = std::fs::read_to_string("/sys/fs/selinux/enforce")
        .map(|value| value.trim() != "0")
        // Unreadable means we cannot prove permissive; assume the strict case.
        .unwrap_or(true);
    if !enforcing {
        return true;
    }
    !context.contains(":untrusted_app")
}

/// Writes `bytes` at `address`, then restores the *original* permissions of the touched pages.
/// Returns the permissions that were restored. Never leaves `W|X` behind.
///
/// `WriteMode::Split` drops `X` for the duration of the write, so the page is
/// never writable and executable at the same time. `WriteMode::Rwx` keeps both
/// bits for the duration of the write and is visible to `/proc/self/maps` scans;
/// even then the bits are the caller's `R|W|X` window only, because the original
/// permissions are restored before returning.
///
/// # Errors
///
/// Returns [`HookError::Misaligned`] for a null address or an empty slice,
/// [`HookError::NotMapped`] when no mapping contains `address`,
/// [`HookError::Protect`] when `mprotect` refuses the write or the restore.
///
/// On Linux/Android the write goes through `process_vm_writev(2)` first. That keeps SELinux from
/// marking the page as a modified executable mapping (`execmod` denial) and needs no permission
/// change at all; `mprotect` is only the fallback.
pub fn write_code(address: usize, bytes: &[u8], mode: WriteMode) -> Result<Perms, HookError> {
    if address == 0 || bytes.is_empty() {
        return Err(HookError::Misaligned);
    }
    let length = bytes.len();
    // A 16-byte detour planted in the last 12 bytes of a page is a normal occurrence, so a range
    // that crosses into the next mapping is supported: every mapping is made writable on its own and
    // restored with the permissions it had.
    let spans = mappings_covering(address, length)?;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        if write_through_process_memory(address, bytes) {
            instruction_barrier();
            clear_instruction_cache(address, length);
            return Ok(spans[0].2);
        }
    }

    let mut writable: Vec<(usize, usize, Perms)> = Vec::with_capacity(spans.len());
    let mut failure: Option<HookError> = None;
    for (start, end, original) in &spans {
        let from = (*start).max(address);
        let to = (*end).min(address.saturating_add(length));
        let window = match mode {
            // `W` replaces `X`: the page is writable xor executable during the write.
            WriteMode::Split => Perms {
                read: original.read,
                write: true,
                execute: false,
            },
            // The only branch that asks for `W|X`; requested explicitly by the caller.
            WriteMode::Rwx => Perms {
                read: true,
                write: true,
                execute: true,
            },
        };
        match protect(from, to - from, window.to_mprotect()) {
            Ok(()) => writable.push((from, to - from, *original)),
            Err(err) => {
                failure = Some(err);
                break;
            }
        }
    }

    if failure.is_none() {
        begin_jit_write();
        // SAFETY: every mapping covering the range was just made writable.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, length);
        }
        instruction_barrier();
        clear_instruction_cache(address, length);
        end_jit_write();
    }

    // Restore in reverse order so a partial write never leaves a page writable.
    let mut restore_failure = None;
    for (from, span, original) in writable.iter().rev() {
        if let Err(err) = protect(*from, *span, original.to_mprotect()) {
            if restore_failure.is_none() {
                restore_failure = Some(err);
            }
        }
    }
    if let Some(err) = failure {
        return Err(err);
    }
    if let Some(err) = restore_failure {
        return Err(err);
    }
    Ok(spans[0].2)
}

/// Every mapping that covers `[address, address + length)`, in ascending order.
///
/// Fails when any byte of the range is unmapped, so a write can never touch a hole.
fn mappings_covering(
    address: usize,
    length: usize,
) -> Result<Vec<(usize, usize, Perms)>, HookError> {
    if address == 0 || length == 0 {
        return Err(HookError::Misaligned);
    }
    let end = address.saturating_add(length);
    let mut spans = Vec::new();

    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        for region in read_maps()? {
            if region.start < end && region.end > address {
                let permissions = Perms::from_maps_str(&region.permissions).ok_or_else(|| {
                    HookError::Maps(format!(
                        "unparsable permission column {:?} for {:#x}",
                        region.permissions, region.start
                    ))
                })?;
                spans.push((region.start, region.end, permissions));
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let mut cursor = address;
        while cursor < end {
            let (start, region_end, permissions) = darwin::region(cursor)?;
            spans.push((start, region_end, permissions));
            if region_end <= cursor {
                break;
            }
            cursor = region_end;
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        let _ = (&mut spans, end);
        return Err(HookError::Maps(
            "no mapping introspection on this os".to_string(),
        ));
    }

    // Contiguity: every byte from `address` to `end` must belong to one of the spans.
    let mut cursor = address;
    for (start, span_end, _) in &spans {
        if *start > cursor {
            return Err(HookError::NotMapped { address: cursor });
        }
        cursor = cursor.max(*span_end);
    }
    if spans.is_empty() || cursor < end {
        return Err(HookError::NotMapped { address: cursor });
    }
    Ok(spans)
}

/// Pre-faults and `mlock`s the pages covering `[address, address+length)` so a file-backed code page
/// cannot be dropped and re-read clean. Returns `false` when the kernel refused (never an error).
///
/// The range is checked against the process mappings first, so an unmapped
/// address is refused instead of faulting.
pub fn pin(address: usize, length: usize) -> bool {
    if address == 0 || length == 0 {
        return false;
    }
    // `mincore` alone is not enough: on Darwin it accepts an unmapped address,
    // so the page must be proven mapped before it is touched.
    let Ok((_, mapping_end, access)) = mapping_containing(address) else {
        return false;
    };
    if address.saturating_add(length) > mapping_end {
        return false;
    }
    let page = page_size();
    let start = address & !(page - 1);
    let end = address.saturating_add(length).saturating_add(page - 1) & !(page - 1);
    let span = end.saturating_sub(start);
    if span == 0 {
        return false;
    }
    if access.read || access.write {
        let mut cursor = start;
        while cursor < end {
            // SAFETY: the whole range sits inside one mapped, readable region.
            let faulted = unsafe { std::ptr::read_volatile(cursor as *const u8) };
            std::hint::black_box(faulted);
            cursor += page;
        }
    }
    // SAFETY: `start`/`span` are page-aligned and cover the faulted pages.
    unsafe { libc::mlock(start as *const libc::c_void, span) == 0 }
}

/// `LDR X17, #8`, the first instruction of the detour this module plants.
const DETOUR_LOAD_X17: u32 = 0x5800_0051;
/// `BR X17`, the second instruction of the same detour.
const DETOUR_BRANCH_X17: u32 = 0xD61F_0220;

/// `true` when the first instruction in `bytes` is already one of this module's detours.
///
/// This is the cheap half of the double-hook guard: an `LDR X17, #8` (or the
/// `BR X17` that follows it) at the site means somebody already planted the
/// same 16-byte absolute branch, and patching again would corrupt it.
pub fn looks_like_detour(bytes: &[u8]) -> bool {
    if bytes.len() < 4 {
        return false;
    }
    let first = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    first == DETOUR_LOAD_X17 || first == DETOUR_BRANCH_X17
}

/// Resolves the mapping containing `address` to `(start, end, permissions)`.
fn mapping_containing(address: usize) -> Result<(usize, usize, Perms), HookError> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let regions = read_maps()?;
        let region = regions
            .into_iter()
            .find(|region| address >= region.start && address < region.end)
            .ok_or(HookError::NotMapped { address })?;
        let permissions = Perms::from_maps_str(&region.permissions).ok_or_else(|| {
            HookError::Maps(format!(
                "unparsable permission column {:?} for {address:#x}",
                region.permissions
            ))
        })?;
        Ok((region.start, region.end, permissions))
    }
    #[cfg(target_os = "macos")]
    {
        darwin::region(address)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        let _ = address;
        Err(HookError::Maps(
            "no mapping introspection on this os".to_string(),
        ))
    }
}

#[cfg(target_os = "macos")]
mod darwin {
    use super::{HookError, Perms};

    /// `VM_REGION_BASIC_INFO_64`; `protection` is the first field of its info struct.
    const VM_REGION_BASIC_INFO_64: libc::c_int = 9;
    /// `vm_region_basic_info_data_64_t` is 9 `int`s wide on 64-bit Darwin.
    const BASIC_INFO_COUNT: libc::mach_msg_type_number_t = 9;

    extern "C" {
        fn vm_region_64(
            target_task: libc::mach_port_t,
            address: *mut libc::vm_address_t,
            size: *mut libc::vm_size_t,
            flavor: libc::c_int,
            info: *mut libc::c_int,
            info_count: *mut libc::mach_msg_type_number_t,
            object_name: *mut libc::mach_port_t,
        ) -> libc::kern_return_t;
    }

    #[allow(deprecated)] // libc deprecates both `mach_task_self` spellings; the port is stable ABI.
    pub(super) fn region(address: usize) -> Result<(usize, usize, Perms), HookError> {
        let mut start = address as libc::vm_address_t;
        let mut size: libc::vm_size_t = 0;
        // `protection` is the first field of every region flavor, so a raw
        // buffer avoids depending on Darwin's packed struct layout.
        let mut info = [0 as libc::c_int; 16];
        let mut count: libc::mach_msg_type_number_t = BASIC_INFO_COUNT;
        let mut object: libc::mach_port_t = 0;
        // SAFETY: `info` has room for `count` integers and `vm_region_64` only
        // writes to that buffer.
        let kr = unsafe {
            vm_region_64(
                libc::mach_task_self(),
                &mut start,
                &mut size,
                VM_REGION_BASIC_INFO_64,
                info.as_mut_ptr(),
                &mut count,
                &mut object,
            )
        };
        if kr != 0 {
            return Err(HookError::NotMapped { address });
        }
        let start = start as usize;
        let end = start.saturating_add(size as usize);
        if address < start || address >= end {
            return Err(HookError::NotMapped { address });
        }
        let bits = info[0];
        Ok((
            start,
            end,
            Perms {
                read: bits & libc::VM_PROT_READ != 0,
                write: bits & libc::VM_PROT_WRITE != 0,
                execute: bits & libc::VM_PROT_EXECUTE != 0,
            },
        ))
    }
}

/// `true` when the first instruction at `target` is a plain `B`/`BR` whose
/// destination cannot be inside any file-backed mapping.
///
/// A `B` is followed through its immediate. A `BR Xn` keeps its destination in
/// a register, so a routine that starts with one is already a thunk into
/// somebody else's trampoline. Sites that cannot be classified (no
/// `/proc/self/maps` on this OS) are not refused.
fn branches_outside_file_mapping(insn: u32, target: usize) -> bool {
    if insn & 0xFC00_0000 == 0x1400_0000 {
        // `B imm26`: sign-extended, scaled by 4.
        let offset = sign_extend((insn & 0x03FF_FFFF) as u64, 26).wrapping_mul(4);
        let destination = (target as u64).wrapping_add(offset as u64) as usize;
        return !in_file_backed_mapping(destination);
    }
    // `BR Xn`.
    insn & 0xFFFF_FC1F == 0xD61F_0000
}

fn in_file_backed_mapping(address: usize) -> bool {
    match read_maps() {
        Ok(regions) => regions.iter().any(|region| {
            address >= region.start && address < region.end && is_file_backed(region)
        }),
        // Without maps the branch destination cannot be classified; never refuse.
        Err(_) => true,
    }
}

fn is_file_backed(region: &MapRegion) -> bool {
    !region.pathname.is_empty() && !region.pathname.starts_with('[')
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits.min(64);
    ((value << shift) as i64) >> shift
}

/// Orders prior data writes before the instruction fetch that follows them.
fn instruction_barrier() {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // SAFETY: barriers only order memory and instruction accesses.
        core::arch::asm!("dsb sy", "isb", options(nostack, preserves_flags));
    }
}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "android")),
    allow(dead_code)
)] // only `read_maps` uses it, and that file does not exist off Linux/Android
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
        // Apple Silicon needs MAP_JIT before it will execute anonymous code.
        flags |= libc::MAP_JIT;
    }
    // No `PROT_EXEC`: the page is written while it is RW and flipped to RX by
    // `write_executable`, so it is never mapped writable and executable at once.
    // SAFETY: anonymous mapping; a null address lets the kernel choose the base.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        return Err(HookError::Map {
            errno: std::io::Error::last_os_error().raw_os_error().unwrap_or(-1),
        });
    }
    Ok(base as usize)
}

/// Writes a trampoline slot: `RW` -> write -> `RX`, never `RWX`.
///
/// Protection is page-granular, so while a second slot on a page is written the
/// page momentarily loses `X` for the already-installed slots on it as well.
fn write_executable(address: usize, bytes: &[u8]) -> Result<(), HookError> {
    if bytes.is_empty() {
        return Err(HookError::Misaligned);
    }
    let writable = Perms {
        read: true,
        write: true,
        execute: false,
    };
    protect(address, bytes.len(), writable.to_mprotect())?;
    begin_jit_write();
    // SAFETY: `protect` just made this span writable without execute.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len());
    }
    instruction_barrier();
    clear_instruction_cache(address, bytes.len());
    end_jit_write();
    let executable = Perms {
        read: true,
        write: false,
        execute: true,
    };
    protect(address, bytes.len(), executable.to_mprotect())?;
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

/// Write mode used when patching a live code page.
///
/// Android and Linux keep `X` for the duration of the write: the page usually holds this crate's own
/// instructions too, and dropping `X` there faults on the next instruction fetch. macOS refuses a
/// `W|X` `mprotect` outright (`EACCES`), so it takes the short non-executable window instead.
fn patch_mode() -> WriteMode {
    #[cfg(any(target_os = "android", target_os = "linux"))]
    {
        WriteMode::Rwx
    }
    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    {
        WriteMode::Split
    }
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
///
/// Bionic does not export `__clear_cache` (it lives in compiler-rt, which a `cdylib` does not link),
/// so Linux/Android runs the maintenance sequence directly: clean each line of `[address, end)` with
/// `dc cvau`, then invalidate it with `ic ivau`, separated by `dsb ish` barriers and closed with
/// `isb` so this core cannot execute stale instructions afterwards.
pub fn clear_instruction_cache(address: usize, length: usize) {
    if length == 0 {
        return;
    }

    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "android", target_os = "linux")
    ))]
    {
        // 64 bytes is the minimum I-cache line on every ARMv8 implementation.
        const LINE: usize = 64;
        let start = address & !(LINE - 1);
        let end = address.saturating_add(length);
        let mut line = start;
        while line < end {
            // SAFETY: cache maintenance instructions only touch the named cache line, which lies
            // inside the range the caller owns.
            unsafe { core::arch::asm!("dc cvau, {}", in(reg) line, options(nostack, preserves_flags)) };
            line += LINE;
        }
        // SAFETY: barrier only.
        unsafe { core::arch::asm!("dsb ish", options(nostack, preserves_flags)) };
        let mut line = start;
        while line < end {
            // SAFETY: as above; `ic ivau` invalidates the same line from the instruction cache.
            unsafe { core::arch::asm!("ic ivau, {}", in(reg) line, options(nostack, preserves_flags)) };
            line += LINE;
        }
        // SAFETY: barriers only.
        unsafe {
            core::arch::asm!("dsb ish", options(nostack, preserves_flags));
            core::arch::asm!("isb", options(nostack, preserves_flags));
        }
        return;
    }

    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "android", target_os = "linux")
    )))]
    {
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

    #[test]
    fn perms_parses_the_permission_column() {
        assert_eq!(
            Perms::from_maps_str("r-xp"),
            Some(Perms {
                read: true,
                write: false,
                execute: true
            })
        );
        assert_eq!(
            Perms::from_maps_str("rw-s"),
            Some(Perms {
                read: true,
                write: true,
                execute: false
            })
        );
        assert_eq!(
            Perms::from_maps_str("---p"),
            Some(Perms {
                read: false,
                write: false,
                execute: false
            })
        );
        // Unknown characters are ignored, not rejected.
        assert_eq!(
            Perms::from_maps_str("rq-p"),
            Some(Perms {
                read: true,
                write: false,
                execute: false
            })
        );
        // Garbage, truncated and empty columns cannot be a maps entry.
        assert_eq!(Perms::from_maps_str("garbage"), None);
        assert_eq!(Perms::from_maps_str("rw"), None);
        assert_eq!(Perms::from_maps_str(""), None);
    }

    #[test]
    fn mprotect_flags_follow_the_permission_bits() {
        assert_eq!(Perms::from_maps_str("---p").unwrap().to_mprotect(), 0);
        assert_eq!(
            Perms::from_maps_str("rw-p").unwrap().to_mprotect(),
            libc::PROT_READ | libc::PROT_WRITE
        );
        assert_eq!(
            Perms::from_maps_str("r-xp").unwrap().to_mprotect(),
            libc::PROT_READ | libc::PROT_EXEC
        );
        assert_eq!(
            Perms::from_maps_str("rwxp").unwrap().to_mprotect(),
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC
        );
    }

    #[test]
    fn detour_detector_matches_only_our_stub() {
        let stub = crate::reloc::encode_absolute_branch(0x1_0000);
        assert!(looks_like_detour(&stub));
        assert!(looks_like_detour(&stub[..4]));
        assert!(!looks_like_detour(&0xD503_201Fu32.to_le_bytes()));
        assert!(!looks_like_detour(&[0x51, 0x00, 0x00]));
        assert!(!looks_like_detour(&[]));
    }

    #[test]
    fn write_code_restores_the_original_permissions() {
        let length = page_size();
        // SAFETY: fresh anonymous mapping of one page.
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
        let base = base as usize;
        // Start RX so the write exercises the executable -> writable -> executable window.
        let executable = Perms {
            read: true,
            write: false,
            execute: true,
        };
        protect(base, length, executable.to_mprotect()).unwrap();
        let before = perms(base).unwrap();
        assert_eq!(before, executable);

        let payload = [0xA5u8; 16];
        let restored = write_code(base, &payload, WriteMode::Split).unwrap();
        assert_eq!(restored, before);

        // SAFETY: the mapping is live for `length` bytes and readable.
        let written = unsafe { std::slice::from_raw_parts(base as *const u8, payload.len()) };
        assert_eq!(written, &payload);

        let after = perms(base).unwrap();
        assert_eq!(after, before, "mprotect did not restore the maps permissions");
        assert!(!(after.write && after.execute), "left a W|X page behind");

        // SAFETY: the mapping is live and readable, and still executable.
        let reopened = unsafe { std::slice::from_raw_parts(base as *const u8, payload.len()) };
        assert_eq!(reopened, &payload);

        // SAFETY: `base`/`length` are exactly the mapping created above.
        unsafe { libc::munmap(base as *mut libc::c_void, length) };
    }

    #[test]
    fn pin_reports_a_bool_without_faulting() {
        let length = page_size();
        let heap = vec![0u8; length];
        let address = heap.as_ptr() as usize;
        // Either answer is valid (the kernel may refuse mlock); the call must not panic.
        let pinned: bool = pin(address, 16);
        std::hint::black_box(pinned);
        // Refused ranges come back as `false` instead of faulting.
        assert!(!pin(0, 16));
        assert!(!pin(address, 0));
        assert!(!pin(0x10, length));
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
        let mut hooked = unsafe { hook(page, replacement as *const () as usize) }.unwrap();
        assert!(hooked.is_intact());
        assert!(hooked.restored_perms().execute);
        // The trampoline page is flipped back to RX after the slot write.
        let trampoline_perms = perms(hooked.trampoline()).unwrap();
        assert!(trampoline_perms.execute);
        assert!(!trampoline_perms.write, "trampoline page left writable");
        assert_eq!(original(), 7);
        // A second install on the same site must not corrupt the trampoline.
        let again = unsafe { hook(page, replacement as *const () as usize) };
        assert!(matches!(again, Err(HookError::AlreadyHooked { .. })));
        let via_trampoline: extern "C" fn() -> u64 =
            unsafe { std::mem::transmute(hooked.trampoline()) };
        assert_eq!(via_trampoline(), 42);
        hooked.unhook().unwrap();
        assert!(!hooked.is_intact());
        assert_eq!(original(), 42);
        drop(hooked);
        assert_eq!(original(), 42);
    }
}
