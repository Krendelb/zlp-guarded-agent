//! Process-local, path-free references into an immutable Lake view.
//!
//! This module does not own or copy the payload. A caller opens or maps the
//! Lake once, gives this view a borrowed byte slice, and then passes compact
//! references through the hot path. Resolving a reference returns a slice of
//! that same backing memory.

use core::mem::{align_of, size_of};
use core::ops::Range;

pub const LAKE_REF_MAGIC: u32 = 0x5A4C_5053; // "ZLPS"
pub const LAKE_REF_SCHEMA_VERSION: u16 = 1;
pub const LAKE_REF_KIND_SPAN: u8 = 1;

/// Compact capability for one range in one process-local Lake generation.
///
/// Fields are private deliberately. External code can obtain a reference only
/// from [`LakeReadViewV0::reference`], after bounds and identity validation.
/// The descriptor occupies exactly one 64-byte ZLP transport block.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LakeSpanRef64V1 {
    magic: u32,
    schema_version: u16,
    ref_kind: u8,
    flags: u8,
    lake_instance_id: u64,
    generation: u64,
    set_id: u64,
    offset: u64,
    length: u64,
    reserved: [u8; 12],
    integrity: u32,
}

const _: () = assert!(size_of::<LakeSpanRef64V1>() == 64);
const _: () = assert!(align_of::<LakeSpanRef64V1>() == 64);

impl LakeSpanRef64V1 {
    pub fn lake_instance_id(&self) -> u64 {
        self.lake_instance_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn set_id(&self) -> u64 {
        self.set_id
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn length(&self) -> u64 {
        self.length
    }

    pub fn verify_integrity(&self) -> bool {
        self.integrity == crc32_ieee(&self.encode_without_integrity())
    }

    pub fn to_le_bytes(&self) -> [u8; 64] {
        let mut out = self.encode_without_integrity();
        out[60..64].copy_from_slice(&self.integrity.to_le_bytes());
        out
    }

    fn seal(&mut self) {
        self.integrity = 0;
        self.integrity = crc32_ieee(&self.encode_without_integrity());
    }

    fn encode_without_integrity(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic.to_le_bytes());
        out[4..6].copy_from_slice(&self.schema_version.to_le_bytes());
        out[6] = self.ref_kind;
        out[7] = self.flags;
        out[8..16].copy_from_slice(&self.lake_instance_id.to_le_bytes());
        out[16..24].copy_from_slice(&self.generation.to_le_bytes());
        out[24..32].copy_from_slice(&self.set_id.to_le_bytes());
        out[32..40].copy_from_slice(&self.offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.length.to_le_bytes());
        out[48..60].copy_from_slice(&self.reserved);
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LakeRefError {
    InvalidLakeIdentity,
    InvalidSetId,
    EmptyRange,
    ViewTooLarge,
    RangeOverflow,
    OutOfBounds,
    ReferenceIntegrity,
    ReferenceSchema,
    LakeInstanceMismatch,
    StaleGeneration,
}

/// Borrowed, immutable window over an already-open Lake mapping.
///
/// It contains no path and performs no I/O. The lifetime prevents a resolved
/// slice from outliving the mapping supplied by the caller.
pub struct LakeReadViewV0<'lake> {
    bytes: &'lake [u8],
    lake_instance_id: u64,
    generation: u64,
}

impl<'lake> LakeReadViewV0<'lake> {
    pub fn open(
        bytes: &'lake [u8],
        lake_instance_id: u64,
        generation: u64,
    ) -> Result<Self, LakeRefError> {
        if lake_instance_id == 0 || generation == 0 {
            return Err(LakeRefError::InvalidLakeIdentity);
        }
        Ok(Self {
            bytes,
            lake_instance_id,
            generation,
        })
    }

    pub fn reference(
        &self,
        set_id: u64,
        range: Range<usize>,
    ) -> Result<LakeSpanRef64V1, LakeRefError> {
        if set_id == 0 {
            return Err(LakeRefError::InvalidSetId);
        }
        if range.start == range.end {
            return Err(LakeRefError::EmptyRange);
        }
        let length = range
            .end
            .checked_sub(range.start)
            .ok_or(LakeRefError::RangeOverflow)?;
        if range.end > self.bytes.len() {
            return Err(LakeRefError::OutOfBounds);
        }
        let offset = u64::try_from(range.start).map_err(|_| LakeRefError::ViewTooLarge)?;
        let length = u64::try_from(length).map_err(|_| LakeRefError::ViewTooLarge)?;

        let mut reference = LakeSpanRef64V1 {
            magic: LAKE_REF_MAGIC,
            schema_version: LAKE_REF_SCHEMA_VERSION,
            ref_kind: LAKE_REF_KIND_SPAN,
            flags: 0,
            lake_instance_id: self.lake_instance_id,
            generation: self.generation,
            set_id,
            offset,
            length,
            reserved: [0; 12],
            integrity: 0,
        };
        reference.seal();
        Ok(reference)
    }

    /// Resolve to the exact same backing bytes; the payload is not copied.
    pub fn resolve(&self, reference: &LakeSpanRef64V1) -> Result<&'lake [u8], LakeRefError> {
        if !reference.verify_integrity() {
            return Err(LakeRefError::ReferenceIntegrity);
        }
        if reference.magic != LAKE_REF_MAGIC
            || reference.schema_version != LAKE_REF_SCHEMA_VERSION
            || reference.ref_kind != LAKE_REF_KIND_SPAN
            || reference.flags != 0
            || reference.reserved != [0; 12]
        {
            return Err(LakeRefError::ReferenceSchema);
        }
        if reference.lake_instance_id != self.lake_instance_id {
            return Err(LakeRefError::LakeInstanceMismatch);
        }
        if reference.generation != self.generation {
            return Err(LakeRefError::StaleGeneration);
        }
        if reference.set_id == 0 {
            return Err(LakeRefError::InvalidSetId);
        }

        let start = usize::try_from(reference.offset).map_err(|_| LakeRefError::RangeOverflow)?;
        let length = usize::try_from(reference.length).map_err(|_| LakeRefError::RangeOverflow)?;
        let end = start
            .checked_add(length)
            .ok_or(LakeRefError::RangeOverflow)?;
        if reference.length == 0 {
            return Err(LakeRefError::EmptyRange);
        }
        self.bytes.get(start..end).ok_or(LakeRefError::OutOfBounds)
    }

    pub fn lake_instance_id(&self) -> u64 {
        self.lake_instance_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSTANCE_A: u64 = 0xa11c_e001;
    const INSTANCE_B: u64 = 0xb11c_e002;
    const GENERATION: u64 = 7;
    const SET_ID: u64 = 0x44;

    #[test]
    fn reference_is_exactly_one_aligned_64_byte_block() {
        assert_eq!(size_of::<LakeSpanRef64V1>(), 64);
        assert_eq!(align_of::<LakeSpanRef64V1>(), 64);
    }

    #[test]
    fn resolve_borrows_the_same_backing_memory_without_copying() {
        let lake = b"prefix|one immutable evidence body|suffix";
        let start = 7;
        let end = lake.len() - 7;
        let view = LakeReadViewV0::open(lake, INSTANCE_A, GENERATION).unwrap();
        let reference = view.reference(SET_ID, start..end).unwrap();
        let resolved = view.resolve(&reference).unwrap();

        assert_eq!(resolved, b"one immutable evidence body");
        assert_eq!(resolved.as_ptr(), lake[start..end].as_ptr());
        assert_eq!(reference.set_id(), SET_ID);
        assert_eq!(reference.offset(), start as u64);
        assert_eq!(reference.length(), (end - start) as u64);
        assert!(reference.verify_integrity());
    }

    #[test]
    fn reference_from_another_lake_instance_fails_closed() {
        let lake = b"immutable";
        let source = LakeReadViewV0::open(lake, INSTANCE_A, GENERATION).unwrap();
        let other = LakeReadViewV0::open(lake, INSTANCE_B, GENERATION).unwrap();
        let reference = source.reference(SET_ID, 0..lake.len()).unwrap();

        assert_eq!(
            other.resolve(&reference),
            Err(LakeRefError::LakeInstanceMismatch)
        );
    }

    #[test]
    fn stale_generation_fails_closed() {
        let lake = b"immutable";
        let old = LakeReadViewV0::open(lake, INSTANCE_A, GENERATION).unwrap();
        let current = LakeReadViewV0::open(lake, INSTANCE_A, GENERATION + 1).unwrap();
        let reference = old.reference(SET_ID, 0..lake.len()).unwrap();

        assert_eq!(
            current.resolve(&reference),
            Err(LakeRefError::StaleGeneration)
        );
    }

    #[test]
    fn invalid_identity_set_and_ranges_fail_closed() {
        let lake = b"immutable";
        assert!(matches!(
            LakeReadViewV0::open(lake, 0, GENERATION),
            Err(LakeRefError::InvalidLakeIdentity)
        ));

        let view = LakeReadViewV0::open(lake, INSTANCE_A, GENERATION).unwrap();
        assert_eq!(
            view.reference(0, 0..lake.len()),
            Err(LakeRefError::InvalidSetId)
        );
        assert_eq!(view.reference(SET_ID, 3..3), Err(LakeRefError::EmptyRange));
        let reversed = Range { start: 7, end: 6 };
        assert_eq!(
            view.reference(SET_ID, reversed),
            Err(LakeRefError::RangeOverflow)
        );
        assert_eq!(
            view.reference(SET_ID, 0..lake.len() + 1),
            Err(LakeRefError::OutOfBounds)
        );
    }

    #[test]
    fn repeated_resolution_is_deterministic_and_keeps_the_same_address() {
        let lake = b"observation|evidence|invariants";
        let view = LakeReadViewV0::open(lake, INSTANCE_A, GENERATION).unwrap();
        let reference = view.reference(SET_ID, 12..20).unwrap();
        let first = view.resolve(&reference).unwrap();
        let second = view.resolve(&reference).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.as_ptr(), second.as_ptr());
    }

    #[test]
    fn mutated_reference_integrity_fails_closed() {
        let lake = b"immutable";
        let view = LakeReadViewV0::open(lake, INSTANCE_A, GENERATION).unwrap();
        let mut reference = view.reference(SET_ID, 0..lake.len()).unwrap();
        reference.offset ^= 1;

        assert_eq!(
            view.resolve(&reference),
            Err(LakeRefError::ReferenceIntegrity)
        );
    }
}
