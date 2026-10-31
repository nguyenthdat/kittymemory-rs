//! Relocate AArch64 instructions stolen by a 16-byte absolute detour.
//!
//! The detour is `LDR X17, #8; BR X17; <addr>`. PC-relative opcodes in those
//! 16 bytes (`ADR`, `ADRP`, `B`, `BL`, `B.cond`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ`,
//! literal `LDR`) are rewritten so they still name the original target when
//! they run from the trampoline. `X17` is the scratch register, matching the
//! detour.

use crate::hook::HookError;

const NOP: u32 = 0xD503_201F;

/// `LDR X17, #8; BR X17` followed by a 64-bit absolute address.
pub fn encode_absolute_branch(destination: u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&0x5800_0051u32.to_le_bytes());
    out[4..8].copy_from_slice(&0xD61F_0220u32.to_le_bytes());
    out[8..16].copy_from_slice(&destination.to_le_bytes());
    out
}

/// Rewrite `src` (a multiple of 4) as it would execute at `dest_pc` instead of `source_pc`.
pub fn relocate(src: &[u8], source_pc: u64, dest_pc: u64) -> Result<Vec<u8>, HookError> {
    if src.len() % 4 != 0 {
        return Err(HookError::Misaligned);
    }
    let mut output = Vec::with_capacity(src.len().saturating_mul(4));
    let mut source = source_pc;
    let _ = dest_pc;
    for chunk in src.chunks_exact(4) {
        let insn = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let encoded = relocate_one(insn, source)?;
        output.extend_from_slice(&encoded);
        source = source.wrapping_add(4);
    }
    Ok(output)
}

fn relocate_one(insn: u32, source_pc: u64) -> Result<Vec<u8>, HookError> {
    if is_adr(insn) {
        let register = insn & 0x1f;
        let immediate = adr_immediate(insn);
        let target = if insn & 0x8000_0000 != 0 {
            (source_pc & !0xfff).wrapping_add(immediate)
        } else {
            source_pc.wrapping_add(immediate)
        };
        return Ok(mov_wide(register, target).to_vec());
    }
    if (insn & 0xfc00_0000) == 0x1400_0000 || (insn & 0xfc00_0000) == 0x9400_0000 {
        let immediate = sign_extend((insn & 0x03ff_ffff) as u64, 26).wrapping_mul(4);
        let target = source_pc.wrapping_add(immediate as u64);
        let link = insn & 0x8000_0000 != 0;
        return Ok(if link {
            absolute_call(target)
        } else {
            encode_absolute_branch(target).to_vec()
        });
    }
    if (insn & 0xff00_0010) == 0x5400_0000 {
        let condition = insn & 0xf;
        let immediate = sign_extend(((insn >> 5) & 0x7_ffff) as u64, 19).wrapping_mul(4);
        let target = source_pc.wrapping_add(immediate as u64);
        return Ok(conditional_absolute(condition, target));
    }
    if (insn & 0x7e00_0000) == 0x3400_0000 {
        let immediate = sign_extend(((insn >> 5) & 0x7_ffff) as u64, 19).wrapping_mul(4);
        let target = source_pc.wrapping_add(immediate as u64);
        return Ok(compare_and_branch(insn, target));
    }
    if (insn & 0x7e00_0000) == 0x3600_0000 {
        let immediate = sign_extend(((insn >> 5) & 0x3fff) as u64, 14).wrapping_mul(4);
        let target = source_pc.wrapping_add(immediate as u64);
        return Ok(test_and_branch(insn, target));
    }
    if (insn & 0x3f00_0000) == 0x1c00_0000 {
        return Err(HookError::Unrelocatable {
            pc: source_pc,
            insn,
        });
    }
    if (insn & 0x3f00_0000) == 0x1800_0000 {
        let opcode = insn >> 30;
        if opcode == 0b11 {
            return Ok(u32_le(NOP));
        }
        let immediate = sign_extend(((insn >> 5) & 0x7_ffff) as u64, 19).wrapping_mul(4);
        let literal = source_pc.wrapping_add(immediate as u64);
        let register = insn & 0x1f;
        return Ok(load_from_absolute(register, literal, opcode));
    }
    Ok(u32_le(insn))
}

fn is_adr(insn: u32) -> bool {
    insn & 0x1f00_0000 == 0x1000_0000
}

fn adr_immediate(insn: u32) -> u64 {
    let immlo = ((insn >> 29) & 0x3) as u64;
    let immhi = ((insn >> 5) & 0x7_ffff) as u64;
    let raw = (immhi << 2) | immlo;
    let byte = sign_extend(raw, 21) as u64;
    if insn & 0x8000_0000 != 0 {
        byte.wrapping_shl(12)
    } else {
        byte
    }
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

fn u32_le(insn: u32) -> Vec<u8> {
    insn.to_le_bytes().to_vec()
}

fn mov_wide(register: u32, value: u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    let words = [
        0xD280_0000 | (((value as u32) & 0xffff) << 5) | register,
        0xF2A0_0000 | ((((value >> 16) as u32) & 0xffff) << 5) | register,
        0xF2C0_0000 | ((((value >> 32) as u32) & 0xffff) << 5) | register,
        0xF2E0_0000 | ((((value >> 48) as u32) & 0xffff) << 5) | register,
    ];
    for (index, word) in words.iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

fn absolute_call(target: u64) -> Vec<u8> {
    // LDR X17, #12; BLR X17; B #8; .quad target
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&0x5800_0071u32.to_le_bytes());
    out.extend_from_slice(&0xD63F_0220u32.to_le_bytes());
    out.extend_from_slice(&0x1400_0002u32.to_le_bytes());
    out.extend_from_slice(&target.to_le_bytes());
    out
}

fn conditional_absolute(condition: u32, target: u64) -> Vec<u8> {
    // B.cond over the skip; B over the absolute branch; then LDR/BR/.quad.
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(&(0x5400_0040 | condition).to_le_bytes());
    out.extend_from_slice(&0x1400_0005u32.to_le_bytes());
    out.extend_from_slice(&encode_absolute_branch(target));
    out
}

fn compare_and_branch(insn: u32, target: u64) -> Vec<u8> {
    // Invert CBZ/CBNZ so the fall-through enters the absolute branch.
    let inverted = (insn ^ (1 << 24)) & !(0x7_ffff << 5);
    let skip = inverted | (5 << 5);
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&skip.to_le_bytes());
    out.extend_from_slice(&encode_absolute_branch(target));
    out
}

fn test_and_branch(insn: u32, target: u64) -> Vec<u8> {
    let inverted = (insn ^ (1 << 24)) & !(0x3fff << 5);
    let skip = inverted | (5 << 5);
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&skip.to_le_bytes());
    out.extend_from_slice(&encode_absolute_branch(target));
    out
}

fn load_from_absolute(register: u32, literal: u64, opcode: u32) -> Vec<u8> {
    let load = match opcode {
        0b00 => 0xB940_0000 | (17 << 5) | register,
        0b01 => 0xF940_0000 | (17 << 5) | register,
        _ => 0xB980_0000 | (17 << 5) | register,
    };
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&mov_wide(17, literal));
    out.extend_from_slice(&load.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_branch_matches_the_mod_stub() {
        let encoded = encode_absolute_branch(0x1000);
        assert_eq!(
            &encoded[0..8],
            &[0x51, 0x00, 0x00, 0x58, 0x20, 0x02, 0x1f, 0xd6]
        );
        assert_eq!(
            u64::from_le_bytes(encoded[8..16].try_into().unwrap()),
            0x1000
        );
    }

    #[test]
    fn adrp_becomes_the_original_page() {
        // ADRP X0, #+1 page at PC 0x1000 → page 0x2000.
        let adrp = 0xB000_0000u32;
        let relocated = relocate(&adrp.to_le_bytes(), 0x1000, 0x8000).unwrap();
        assert_eq!(relocated.len(), 16);
        assert_eq!(&relocated[0..4], &0xD284_0000u32.to_le_bytes());
        assert_eq!(&relocated[4..8], &0xF2A0_0000u32.to_le_bytes());
        assert_eq!(&relocated[8..12], &0xF2C0_0000u32.to_le_bytes());
        assert_eq!(&relocated[12..16], &0xF2E0_0000u32.to_le_bytes());
    }

    #[test]
    fn plain_ret_is_copied() {
        let ret = 0xD65F_03C0u32;
        let relocated = relocate(&ret.to_le_bytes(), 0x1000, 0x8000).unwrap();
        assert_eq!(relocated, ret.to_le_bytes());
    }

    #[test]
    fn branch_becomes_absolute() {
        // B #+8 at 0x1000 → 0x1008.
        let branch = 0x1400_0002u32;
        let relocated = relocate(&branch.to_le_bytes(), 0x1000, 0x8000).unwrap();
        assert_eq!(&relocated[0..8], &encode_absolute_branch(0)[0..8]);
        assert_eq!(
            u64::from_le_bytes(relocated[8..16].try_into().unwrap()),
            0x1008
        );
    }

    #[test]
    fn simd_literal_is_rejected() {
        // LDR S0, literal with a zero displacement. V=1 so this is not an integer LDR.
        let insn = 0x1C00_0000u32;
        let error = relocate(&insn.to_le_bytes(), 0x2000, 0x4000).unwrap_err();
        assert!(matches!(error, HookError::Unrelocatable { pc: 0x2000, .. }));
    }
}
