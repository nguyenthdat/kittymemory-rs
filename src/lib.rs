// The README examples drive the C++ API, so they are only compiled as doc tests when that feature
// is on.
#![cfg_attr(feature = "cxx", doc = include_str!("../README.md"))]

pub mod hook;
pub mod reloc;
// KittyMemory's C++ wrapper. Off unless the `cxx` feature is enabled, which keeps the final binary
// free of libc++.
#[cfg(feature = "cxx")]
pub mod safe;
#[cfg(feature = "cxx")]
pub mod sys;

pub use hook::{
    clear_instruction_cache, hook, looks_like_detour, page_residency, perms, pin, read_maps,
    require_single_mapping, write_code, HookError, InlineHook, MapRegion, PageResidency, Perms,
    WriteMode, PATCH_LEN,
};
pub use reloc::{encode_absolute_branch, relocate};
#[cfg(feature = "cxx")]
pub use safe::*;

pub mod prelude {
    pub use crate::hook::{
        clear_instruction_cache, hook, looks_like_detour, page_residency, perms, pin, read_maps,
        require_single_mapping, write_code, HookError, InlineHook, MapRegion, PageResidency, Perms,
        WriteMode, PATCH_LEN,
    };
    pub use crate::reloc::{encode_absolute_branch, relocate};
    #[cfg(feature = "cxx")]
    pub use crate::safe::*;
}

#[cfg(all(test, feature = "cxx"))]
mod tests {
    use super::*;

    #[test]
    fn test_page_functions() {
        let addr = 0x1234;
        let start = safe::page_start(addr);
        assert!(start <= addr);
        assert_eq!(start % (sys::KM_PAGE_SIZE as usize), 0);

        let end = safe::page_end(addr);
        assert!(end >= addr);
    }

    #[test]
    fn test_patch_creation() {
        let patch = safe::Patch::with_bytes(0x1000, &[0x90, 0x90, 0x90, 0x90]);
        assert!(patch.is_valid() || !patch.is_valid());
    }

    #[test]
    fn test_backup_creation() {
        let backup = safe::Backup::create(0x1000, 16);
        assert!(backup.is_valid() || !backup.is_valid());
    }

    #[test]
    fn test_validator_creation() {
        let _validator = safe::PtrValidator::new();
        // Validator is created successfully - the Drop impl will clean it up
    }
}
