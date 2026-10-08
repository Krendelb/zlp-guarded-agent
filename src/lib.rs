//! Fixed-size binary boundary for untrusted local-LLM thought proposals.
//!
//! JSON is deliberately excluded from the runtime path. The programmatic model
//! boundary returns one 64-byte decision frame; the compact-line decoder remains
//! only for compatibility with the earlier CLI test harness. All trusted checks
//! operate on fixed-width integer fields.

use core::mem::{align_of, size_of};

pub mod agent_kernel;
pub mod lake_view;

pub use agent_kernel::{
    run_read_only_agent_v0, AgentKernelError, AgentStepV0, DecisionModuleV0, ModelInputViewV0,
};
pub use lake_view::{LakeReadViewV0, LakeRefError, LakeSpanRef64V1};

pub const FRAME_MAGIC: u32 = 0x5A4C_5043; // "ZLPC"
pub const RECEIPT_MAGIC: u32 = 0x5A4C_5052; // "ZLPR"
pub const DECISION_MAGIC: u32 = 0x5A4C_5044; // "ZLPD"
pub const TOOL_ADMISSION_MAGIC: u32 = 0x5A4C_5054; // "ZLPT"
pub const SCHEMA_VERSION: u16 = 1;

pub const KIND_TRUSTED_OBSERVATION: u8 = 1;
pub const KIND_UNTRUSTED_PROPOSAL: u8 = 2;
pub const KIND_UNTRUSTED_DECISION: u8 = 3;
pub const KIND_TOOL_ADMISSION: u8 = 4;

pub const FLAG_PUBLICATION_REQUESTED: u8 = 1 << 0;
pub const FLAG_CAN_EXECUTE: u8 = 1 << 1;

pub const UPDATE_COMPATIBLE: u8 = 1;
pub const UPDATE_SEMANTIC_BREAK: u8 = 2;
pub const UPDATE_UNKNOWN: u8 = 3;
pub const UPDATE_TAU_DROP: u8 = 4;

pub const DECISION_CONTINUE: u8 = 1;
pub const DECISION_INTERRUPT: u8 = 2;
pub const DECISION_QUARANTINE: u8 = 3;

pub const TOOL_NONE: u8 = 0;
pub const TOOL_READ_FILE: u8 = 1;
pub const TOOL_SEARCH_FILES: u8 = 2;
pub const TOOL_REQUEST_USER_INPUT: u8 = 3;

pub const TAU_FORMING: u8 = 1;
pub const TAU_STABLE: u8 = 2;
pub const TAU_DROP: u8 = 3;

pub const VERDICT_ADMIT_READ_ONLY: u8 = 1;
pub const VERDICT_QUARANTINE: u8 = 2;

pub const REASON_OBSERVATION_INTEGRITY: u64 = 1 << 0;
pub const REASON_PROPOSAL_INTEGRITY: u64 = 1 << 1;
pub const REASON_SCHEMA_OR_KIND: u64 = 1 << 2;
pub const REASON_BINDING_MISMATCH: u64 = 1 << 3;
pub const REASON_INVARIANT_MISMATCH: u64 = 1 << 4;
pub const REASON_EVIDENCE_MISMATCH: u64 = 1 << 5;
pub const REASON_UPDATE_CLASS_MISMATCH: u64 = 1 << 6;
pub const REASON_DECISION_MISMATCH: u64 = 1 << 7;
pub const REASON_TOOL_NOT_READ_ONLY: u64 = 1 << 8;
pub const REASON_PUBLICATION_FORBIDDEN: u64 = 1 << 9;
pub const REASON_EXECUTION_FORBIDDEN: u64 = 1 << 10;
pub const REASON_TAU_MISMATCH: u64 = 1 << 11;
pub const REASON_UNKNOWN_ERASED: u64 = 1 << 12;
pub const REASON_TOOL_ADMISSION_INTEGRITY: u64 = 1 << 13;
pub const REASON_TOOL_ADMISSION_BINDING: u64 = 1 << 14;
pub const REASON_TOOL_ADMISSION_MISMATCH: u64 = 1 << 15;
pub const REASON_TOOL_ADMISSION_SCHEMA: u64 = 1 << 16;

pub const TOOL_ADMISSION_POLICY_ID: u32 = 0x5A4C_5001;
pub const TOOL_ADMISSION_POLICY_VERSION: u16 = 1;

/// One cache-line-sized thought boundary.
///
/// Large meaning, evidence and hypothesis bodies remain in the Lake. This
/// frame carries only stable identities, versions, masks and Lake set IDs.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CognitiveFrame64V0 {
    pub magic: u32,
    pub schema_version: u16,
    pub frame_kind: u8,
    pub flags: u8,
    pub task_id: u64,
    pub object_id: u64,
    pub physical_version: u32,
    pub semantic_epoch: u32,
    pub invariant_set_id: u64,
    pub evidence_set_id: u64,
    pub unknown_mask: u32,
    pub hypothesis_set_id: u32,
    pub update_class: u8,
    pub decision: u8,
    pub requested_tool: u8,
    pub tau_state: u8,
    pub integrity: u32,
}

const _: () = assert!(size_of::<CognitiveFrame64V0>() == 64);
const _: () = assert!(align_of::<CognitiveFrame64V0>() == 64);

impl CognitiveFrame64V0 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_kind: u8,
        task_id: u64,
        object_id: u64,
        physical_version: u32,
        semantic_epoch: u32,
        invariant_set_id: u64,
        evidence_set_id: u64,
        unknown_mask: u32,
        hypothesis_set_id: u32,
        update_class: u8,
        decision: u8,
        requested_tool: u8,
        tau_state: u8,
        flags: u8,
    ) -> Self {
        let mut frame = Self {
            magic: FRAME_MAGIC,
            schema_version: SCHEMA_VERSION,
            frame_kind,
            flags,
            task_id,
            object_id,
            physical_version,
            semantic_epoch,
            invariant_set_id,
            evidence_set_id,
            unknown_mask,
            hypothesis_set_id,
            update_class,
            decision,
            requested_tool,
            tau_state,
            integrity: 0,
        };
        frame.seal();
        frame
    }

    pub fn seal(&mut self) {
        self.integrity = 0;
        self.integrity = crc32_ieee(&self.encode_without_integrity());
    }

    pub fn verify_integrity(&self) -> bool {
        self.integrity == crc32_ieee(&self.encode_without_integrity())
    }

    pub fn to_le_bytes(&self) -> [u8; 64] {
        let mut out = self.encode_without_integrity();
        out[60..64].copy_from_slice(&self.integrity.to_le_bytes());
        out
    }

    fn encode_without_integrity(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic.to_le_bytes());
        out[4..6].copy_from_slice(&self.schema_version.to_le_bytes());
        out[6] = self.frame_kind;
        out[7] = self.flags;
        out[8..16].copy_from_slice(&self.task_id.to_le_bytes());
        out[16..24].copy_from_slice(&self.object_id.to_le_bytes());
        out[24..28].copy_from_slice(&self.physical_version.to_le_bytes());
        out[28..32].copy_from_slice(&self.semantic_epoch.to_le_bytes());
        out[32..40].copy_from_slice(&self.invariant_set_id.to_le_bytes());
        out[40..48].copy_from_slice(&self.evidence_set_id.to_le_bytes());
        out[48..52].copy_from_slice(&self.unknown_mask.to_le_bytes());
        out[52..56].copy_from_slice(&self.hypothesis_set_id.to_le_bytes());
        out[56] = self.update_class;
        out[57] = self.decision;
        out[58] = self.requested_tool;
        out[59] = self.tau_state;
        out
    }
}

/// One 64-byte model decision bound to one trusted observation.
///
/// The LLM owns only class, decision, requested tool and hypothesis ID. The
/// program module copies trusted identity fields from the observation before
/// sealing the frame.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionFrame64V1 {
    magic: u32,
    schema_version: u16,
    frame_kind: u8,
    flags: u8,
    update_class: u8,
    decision: u8,
    requested_tool: u8,
    reserved0: u8,
    hypothesis_set_id: u32,
    task_id: u64,
    object_id: u64,
    physical_version: u32,
    semantic_epoch: u32,
    observation_integrity: u32,
    reserved1: [u8; 16],
    integrity: u32,
}

const _: () = assert!(size_of::<DecisionFrame64V1>() == 64);
const _: () = assert!(align_of::<DecisionFrame64V1>() == 64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionFrameDecodeError {
    Integrity,
    Schema,
}

impl DecisionFrame64V1 {
    pub fn new(
        observation: &CognitiveFrame64V0,
        update_class: u8,
        decision: u8,
        requested_tool: u8,
        hypothesis_set_id: u32,
    ) -> Self {
        let mut frame = Self {
            magic: DECISION_MAGIC,
            schema_version: SCHEMA_VERSION,
            frame_kind: KIND_UNTRUSTED_DECISION,
            flags: 0,
            update_class,
            decision,
            requested_tool,
            reserved0: 0,
            hypothesis_set_id,
            task_id: observation.task_id,
            object_id: observation.object_id,
            physical_version: observation.physical_version,
            semantic_epoch: observation.semantic_epoch,
            observation_integrity: observation.integrity,
            reserved1: [0; 16],
            integrity: 0,
        };
        frame.seal();
        frame
    }

    pub fn update_class(&self) -> u8 {
        self.update_class
    }

    pub fn decision(&self) -> u8 {
        self.decision
    }

    pub fn requested_tool(&self) -> u8 {
        self.requested_tool
    }

    pub fn hypothesis_set_id(&self) -> u32 {
        self.hypothesis_set_id
    }

    pub fn seal(&mut self) {
        self.integrity = 0;
        self.integrity = crc32_ieee(&self.encode_without_integrity());
    }

    pub fn verify_integrity(&self) -> bool {
        self.integrity == crc32_ieee(&self.encode_without_integrity())
    }

    pub fn to_le_bytes(&self) -> [u8; 64] {
        let mut out = self.encode_without_integrity();
        out[60..64].copy_from_slice(&self.integrity.to_le_bytes());
        out
    }

    pub fn from_le_bytes(bytes: [u8; 64]) -> Result<Self, DecisionFrameDecodeError> {
        let frame = Self {
            magic: u32::from_le_bytes(bytes[0..4].try_into().expect("four bytes")),
            schema_version: u16::from_le_bytes(bytes[4..6].try_into().expect("two bytes")),
            frame_kind: bytes[6],
            flags: bytes[7],
            update_class: bytes[8],
            decision: bytes[9],
            requested_tool: bytes[10],
            reserved0: bytes[11],
            hypothesis_set_id: u32::from_le_bytes(bytes[12..16].try_into().expect("four bytes")),
            task_id: u64::from_le_bytes(bytes[16..24].try_into().expect("eight bytes")),
            object_id: u64::from_le_bytes(bytes[24..32].try_into().expect("eight bytes")),
            physical_version: u32::from_le_bytes(bytes[32..36].try_into().expect("four bytes")),
            semantic_epoch: u32::from_le_bytes(bytes[36..40].try_into().expect("four bytes")),
            observation_integrity: u32::from_le_bytes(
                bytes[40..44].try_into().expect("four bytes"),
            ),
            reserved1: bytes[44..60].try_into().expect("sixteen bytes"),
            integrity: u32::from_le_bytes(bytes[60..64].try_into().expect("four bytes")),
        };
        if !frame.verify_integrity() {
            return Err(DecisionFrameDecodeError::Integrity);
        }
        if frame.magic != DECISION_MAGIC
            || frame.schema_version != SCHEMA_VERSION
            || frame.frame_kind != KIND_UNTRUSTED_DECISION
            || frame.flags != 0
            || frame.reserved0 != 0
            || frame.reserved1 != [0; 16]
        {
            return Err(DecisionFrameDecodeError::Schema);
        }
        Ok(frame)
    }

    fn encode_without_integrity(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic.to_le_bytes());
        out[4..6].copy_from_slice(&self.schema_version.to_le_bytes());
        out[6] = self.frame_kind;
        out[7] = self.flags;
        out[8] = self.update_class;
        out[9] = self.decision;
        out[10] = self.requested_tool;
        out[11] = self.reserved0;
        out[12..16].copy_from_slice(&self.hypothesis_set_id.to_le_bytes());
        out[16..24].copy_from_slice(&self.task_id.to_le_bytes());
        out[24..32].copy_from_slice(&self.object_id.to_le_bytes());
        out[32..36].copy_from_slice(&self.physical_version.to_le_bytes());
        out[36..40].copy_from_slice(&self.semantic_epoch.to_le_bytes());
        out[40..44].copy_from_slice(&self.observation_integrity.to_le_bytes());
        out[44..60].copy_from_slice(&self.reserved1);
        out
    }
}

/// Trusted, process-local authorization for exactly one read-only tool.
///
/// The frame does not invent a new semantic reason taxonomy. It binds the
/// already trusted observation unknown mask and exact requested tool to the
/// observation identity and integrity. All fields are private, and no public
/// constructor is exposed to model or calling code.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolAdmission64V1 {
    magic: u32,
    schema_version: u16,
    frame_kind: u8,
    allowed_tool: u8,
    policy_id: u32,
    policy_version: u16,
    reserved0: u16,
    unknown_mask: u32,
    observation_integrity: u32,
    task_id: u64,
    object_id: u64,
    physical_version: u32,
    semantic_epoch: u32,
    reserved1: [u8; 12],
    integrity: u32,
}

const _: () = assert!(size_of::<ToolAdmission64V1>() == 64);
const _: () = assert!(align_of::<ToolAdmission64V1>() == 64);

impl ToolAdmission64V1 {
    pub fn allowed_tool(&self) -> u8 {
        self.allowed_tool
    }

    pub fn unknown_mask(&self) -> u32 {
        self.unknown_mask
    }

    pub fn policy_id(&self) -> u32 {
        self.policy_id
    }

    pub fn policy_version(&self) -> u16 {
        self.policy_version
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
        out[6] = self.frame_kind;
        out[7] = self.allowed_tool;
        out[8..12].copy_from_slice(&self.policy_id.to_le_bytes());
        out[12..14].copy_from_slice(&self.policy_version.to_le_bytes());
        out[14..16].copy_from_slice(&self.reserved0.to_le_bytes());
        out[16..20].copy_from_slice(&self.unknown_mask.to_le_bytes());
        out[20..24].copy_from_slice(&self.observation_integrity.to_le_bytes());
        out[24..32].copy_from_slice(&self.task_id.to_le_bytes());
        out[32..40].copy_from_slice(&self.object_id.to_le_bytes());
        out[40..44].copy_from_slice(&self.physical_version.to_le_bytes());
        out[44..48].copy_from_slice(&self.semantic_epoch.to_le_bytes());
        out[48..60].copy_from_slice(&self.reserved1);
        out
    }
}

pub(crate) fn tool_admission_from_observation(
    observation: &CognitiveFrame64V0,
) -> ToolAdmission64V1 {
    let mut admission = ToolAdmission64V1 {
        magic: TOOL_ADMISSION_MAGIC,
        schema_version: SCHEMA_VERSION,
        frame_kind: KIND_TOOL_ADMISSION,
        allowed_tool: observation.requested_tool,
        policy_id: TOOL_ADMISSION_POLICY_ID,
        policy_version: TOOL_ADMISSION_POLICY_VERSION,
        reserved0: 0,
        unknown_mask: observation.unknown_mask,
        observation_integrity: observation.integrity,
        task_id: observation.task_id,
        object_id: observation.object_id,
        physical_version: observation.physical_version,
        semantic_epoch: observation.semantic_epoch,
        reserved1: [0; 12],
        integrity: 0,
    };
    admission.seal();
    admission
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssembleError {
    ObservationIntegrity,
    ObservationSchema,
    DecisionIntegrity,
    DecisionSchema,
    DecisionBinding,
}

/// Bind the untrusted decision frame to identities owned by the trusted frame.
pub fn assemble_proposal(
    observation: &CognitiveFrame64V0,
    decision_frame: &DecisionFrame64V1,
) -> Result<CognitiveFrame64V0, AssembleError> {
    if !observation.verify_integrity() {
        return Err(AssembleError::ObservationIntegrity);
    }
    if observation.magic != FRAME_MAGIC
        || observation.schema_version != SCHEMA_VERSION
        || observation.frame_kind != KIND_TRUSTED_OBSERVATION
    {
        return Err(AssembleError::ObservationSchema);
    }
    if !decision_frame.verify_integrity() {
        return Err(AssembleError::DecisionIntegrity);
    }
    if decision_frame.magic != DECISION_MAGIC
        || decision_frame.schema_version != SCHEMA_VERSION
        || decision_frame.frame_kind != KIND_UNTRUSTED_DECISION
        || decision_frame.flags != 0
        || decision_frame.reserved0 != 0
        || decision_frame.reserved1 != [0; 16]
    {
        return Err(AssembleError::DecisionSchema);
    }
    if decision_frame.task_id != observation.task_id
        || decision_frame.object_id != observation.object_id
        || decision_frame.physical_version != observation.physical_version
        || decision_frame.semantic_epoch != observation.semantic_epoch
        || decision_frame.observation_integrity != observation.integrity
    {
        return Err(AssembleError::DecisionBinding);
    }

    Ok(CognitiveFrame64V0::new(
        KIND_UNTRUSTED_PROPOSAL,
        observation.task_id,
        observation.object_id,
        observation.physical_version,
        observation.semantic_epoch,
        observation.invariant_set_id,
        observation.evidence_set_id,
        observation.unknown_mask,
        decision_frame.hypothesis_set_id,
        decision_frame.update_class,
        decision_frame.decision,
        decision_frame.requested_tool,
        observation.tau_state,
        decision_frame.flags,
    ))
}

/// Fixed-size result produced by the deterministic gate.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateReceipt64V0 {
    pub magic: u32,
    pub schema_version: u16,
    pub verdict: u8,
    pub flags: u8,
    pub reason_mask: u64,
    pub observation_integrity: u32,
    pub proposal_integrity: u32,
    pub task_id: u64,
    pub object_id: u64,
    pub physical_version: u32,
    pub semantic_epoch: u32,
    pub tool_admission_integrity: u32,
    pub tool_policy_id: u32,
    pub tool_policy_version: u16,
    pub allowed_tool: u8,
    pub reserved: u8,
    pub integrity: u32,
}

const _: () = assert!(size_of::<GateReceipt64V0>() == 64);
const _: () = assert!(align_of::<GateReceipt64V0>() == 64);

impl GateReceipt64V0 {
    fn new(
        observation: &CognitiveFrame64V0,
        proposal: &CognitiveFrame64V0,
        admission: &ToolAdmission64V1,
        reasons: u64,
    ) -> Self {
        let mut receipt = Self {
            magic: RECEIPT_MAGIC,
            schema_version: SCHEMA_VERSION,
            verdict: if reasons == 0 {
                VERDICT_ADMIT_READ_ONLY
            } else {
                VERDICT_QUARANTINE
            },
            flags: 0,
            reason_mask: reasons,
            observation_integrity: observation.integrity,
            proposal_integrity: proposal.integrity,
            task_id: observation.task_id,
            object_id: observation.object_id,
            physical_version: observation.physical_version,
            semantic_epoch: observation.semantic_epoch,
            tool_admission_integrity: admission.integrity,
            tool_policy_id: admission.policy_id,
            tool_policy_version: admission.policy_version,
            allowed_tool: admission.allowed_tool,
            reserved: 0,
            integrity: 0,
        };
        receipt.seal();
        receipt
    }

    pub fn admitted_read_only(&self) -> bool {
        self.verdict == VERDICT_ADMIT_READ_ONLY && self.reason_mask == 0
    }

    pub fn seal(&mut self) {
        self.integrity = 0;
        self.integrity = crc32_ieee(&self.encode_without_integrity());
    }

    pub fn verify_integrity(&self) -> bool {
        self.integrity == crc32_ieee(&self.encode_without_integrity())
    }

    pub fn to_le_bytes(&self) -> [u8; 64] {
        let mut out = self.encode_without_integrity();
        out[60..64].copy_from_slice(&self.integrity.to_le_bytes());
        out
    }

    fn encode_without_integrity(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.magic.to_le_bytes());
        out[4..6].copy_from_slice(&self.schema_version.to_le_bytes());
        out[6] = self.verdict;
        out[7] = self.flags;
        out[8..16].copy_from_slice(&self.reason_mask.to_le_bytes());
        out[16..20].copy_from_slice(&self.observation_integrity.to_le_bytes());
        out[20..24].copy_from_slice(&self.proposal_integrity.to_le_bytes());
        out[24..32].copy_from_slice(&self.task_id.to_le_bytes());
        out[32..40].copy_from_slice(&self.object_id.to_le_bytes());
        out[40..44].copy_from_slice(&self.physical_version.to_le_bytes());
        out[44..48].copy_from_slice(&self.semantic_epoch.to_le_bytes());
        out[48..52].copy_from_slice(&self.tool_admission_integrity.to_le_bytes());
        out[52..56].copy_from_slice(&self.tool_policy_id.to_le_bytes());
        out[56..58].copy_from_slice(&self.tool_policy_version.to_le_bytes());
        out[58] = self.allowed_tool;
        out[59] = self.reserved;
        out
    }
}

/// Deterministic fail-closed admission. No allocation and no text parsing.
pub fn evaluate_frame64(
    observation: &CognitiveFrame64V0,
    proposal: &CognitiveFrame64V0,
) -> GateReceipt64V0 {
    let admission = tool_admission_from_observation(observation);
    evaluate_frame64_with_admission(observation, proposal, &admission)
}

pub(crate) fn evaluate_frame64_with_admission(
    observation: &CognitiveFrame64V0,
    proposal: &CognitiveFrame64V0,
    admission: &ToolAdmission64V1,
) -> GateReceipt64V0 {
    let mut reasons = 0u64;

    if !observation.verify_integrity() {
        reasons |= REASON_OBSERVATION_INTEGRITY;
    }
    if !proposal.verify_integrity() {
        reasons |= REASON_PROPOSAL_INTEGRITY;
    }
    if !admission.verify_integrity() {
        reasons |= REASON_TOOL_ADMISSION_INTEGRITY;
    }
    if admission.magic != TOOL_ADMISSION_MAGIC
        || admission.schema_version != SCHEMA_VERSION
        || admission.frame_kind != KIND_TOOL_ADMISSION
        || admission.policy_id != TOOL_ADMISSION_POLICY_ID
        || admission.policy_version != TOOL_ADMISSION_POLICY_VERSION
        || admission.reserved0 != 0
        || admission.reserved1 != [0; 12]
        || admission.allowed_tool > TOOL_REQUEST_USER_INPUT
    {
        reasons |= REASON_TOOL_ADMISSION_SCHEMA;
    }
    if observation.magic != FRAME_MAGIC
        || proposal.magic != FRAME_MAGIC
        || observation.schema_version != SCHEMA_VERSION
        || proposal.schema_version != SCHEMA_VERSION
        || observation.frame_kind != KIND_TRUSTED_OBSERVATION
        || proposal.frame_kind != KIND_UNTRUSTED_PROPOSAL
    {
        reasons |= REASON_SCHEMA_OR_KIND;
    }

    if observation.task_id != proposal.task_id
        || observation.object_id != proposal.object_id
        || observation.physical_version != proposal.physical_version
        || observation.semantic_epoch != proposal.semantic_epoch
    {
        reasons |= REASON_BINDING_MISMATCH;
    }
    if observation.invariant_set_id != proposal.invariant_set_id {
        reasons |= REASON_INVARIANT_MISMATCH;
    }
    if observation.evidence_set_id != proposal.evidence_set_id {
        reasons |= REASON_EVIDENCE_MISMATCH;
    }
    if proposal.unknown_mask & observation.unknown_mask != observation.unknown_mask {
        reasons |= REASON_UNKNOWN_ERASED;
    }
    if observation.tau_state != proposal.tau_state {
        reasons |= REASON_TAU_MISMATCH;
    }
    if admission.task_id != observation.task_id
        || admission.object_id != observation.object_id
        || admission.physical_version != observation.physical_version
        || admission.semantic_epoch != observation.semantic_epoch
        || admission.observation_integrity != observation.integrity
        || admission.unknown_mask != observation.unknown_mask
        || admission.allowed_tool != observation.requested_tool
    {
        reasons |= REASON_TOOL_ADMISSION_BINDING;
    }

    let expected_class = if observation.tau_state == TAU_DROP {
        UPDATE_TAU_DROP
    } else {
        observation.update_class
    };
    let expected_decision = match expected_class {
        UPDATE_COMPATIBLE => DECISION_CONTINUE,
        UPDATE_SEMANTIC_BREAK => DECISION_INTERRUPT,
        UPDATE_UNKNOWN | UPDATE_TAU_DROP => DECISION_QUARANTINE,
        _ => {
            reasons |= REASON_UPDATE_CLASS_MISMATCH;
            DECISION_QUARANTINE
        }
    };

    if proposal.update_class != expected_class {
        reasons |= REASON_UPDATE_CLASS_MISMATCH;
    }
    if proposal.decision != expected_decision {
        reasons |= REASON_DECISION_MISMATCH;
    }
    if proposal.requested_tool > TOOL_REQUEST_USER_INPUT {
        reasons |= REASON_TOOL_NOT_READ_ONLY;
    }
    if proposal.requested_tool != admission.allowed_tool {
        reasons |= REASON_TOOL_ADMISSION_MISMATCH;
    }
    if proposal.flags & FLAG_PUBLICATION_REQUESTED != 0 {
        reasons |= REASON_PUBLICATION_FORBIDDEN;
    }
    if proposal.flags & FLAG_CAN_EXECUTE != 0 {
        reasons |= REASON_EXECUTION_FORBIDDEN;
    }

    GateReceipt64V0::new(observation, proposal, admission, reasons)
}

/// Compact one-line boundary emitted by the model before packing.
///
/// Format:
/// `Z0|task_hex|object_hex|physical|epoch|invariants_hex|evidence_hex|unknown_hex|hypotheses_hex|class|decision|tool|tau|flags_hex`
pub fn parse_compact_proposal(line: &str) -> Result<CognitiveFrame64V0, CompactParseError> {
    parse_compact_frame(line, KIND_UNTRUSTED_PROPOSAL)
}

/// Parse the only fields the model is allowed to own.
///
/// Format: `D0|class|decision|tool|hypothesis_hex`
pub fn parse_decision_frame(
    line: &str,
    observation: &CognitiveFrame64V0,
) -> Result<DecisionFrame64V1, CompactParseError> {
    let mut fields = line.trim().split('|');
    if fields.next() != Some("D0") {
        return Err(CompactParseError::Prefix);
    }
    let update_class = parse_dec_u8(fields.next())?;
    let decision = parse_dec_u8(fields.next())?;
    let requested_tool = parse_dec_u8(fields.next())?;
    let hypothesis_set_id = parse_hex_u32(fields.next())?;
    if fields.next().is_some() {
        return Err(CompactParseError::FieldCount);
    }
    Ok(DecisionFrame64V1::new(
        observation,
        update_class,
        decision,
        requested_tool,
        hypothesis_set_id,
    ))
}

pub fn parse_compact_frame(
    line: &str,
    frame_kind: u8,
) -> Result<CognitiveFrame64V0, CompactParseError> {
    let mut fields = line.trim().split('|');
    if fields.next() != Some("Z0") {
        return Err(CompactParseError::Prefix);
    }

    let task_id = parse_hex_u64(fields.next())?;
    let object_id = parse_hex_u64(fields.next())?;
    let physical_version = parse_dec_u32(fields.next())?;
    let semantic_epoch = parse_dec_u32(fields.next())?;
    let invariant_set_id = parse_hex_u64(fields.next())?;
    let evidence_set_id = parse_hex_u64(fields.next())?;
    let unknown_mask = parse_hex_u32(fields.next())?;
    let hypothesis_set_id = parse_hex_u32(fields.next())?;
    let update_class = parse_dec_u8(fields.next())?;
    let decision = parse_dec_u8(fields.next())?;
    let requested_tool = parse_dec_u8(fields.next())?;
    let tau_state = parse_dec_u8(fields.next())?;
    let flags = parse_hex_u8(fields.next())?;
    if fields.next().is_some() {
        return Err(CompactParseError::FieldCount);
    }

    Ok(CognitiveFrame64V0::new(
        frame_kind,
        task_id,
        object_id,
        physical_version,
        semantic_epoch,
        invariant_set_id,
        evidence_set_id,
        unknown_mask,
        hypothesis_set_id,
        update_class,
        decision,
        requested_tool,
        tau_state,
        flags,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactParseError {
    Prefix,
    FieldCount,
    Number,
}

fn parse_hex_u64(value: Option<&str>) -> Result<u64, CompactParseError> {
    u64::from_str_radix(value.ok_or(CompactParseError::FieldCount)?, 16)
        .map_err(|_| CompactParseError::Number)
}

fn parse_hex_u32(value: Option<&str>) -> Result<u32, CompactParseError> {
    u32::from_str_radix(value.ok_or(CompactParseError::FieldCount)?, 16)
        .map_err(|_| CompactParseError::Number)
}

fn parse_hex_u8(value: Option<&str>) -> Result<u8, CompactParseError> {
    u8::from_str_radix(value.ok_or(CompactParseError::FieldCount)?, 16)
        .map_err(|_| CompactParseError::Number)
}

fn parse_dec_u32(value: Option<&str>) -> Result<u32, CompactParseError> {
    value
        .ok_or(CompactParseError::FieldCount)?
        .parse()
        .map_err(|_| CompactParseError::Number)
}

fn parse_dec_u8(value: Option<&str>) -> Result<u8, CompactParseError> {
    value
        .ok_or(CompactParseError::FieldCount)?
        .parse()
        .map_err(|_| CompactParseError::Number)
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

    fn observation(update_class: u8, tau: u8) -> CognitiveFrame64V0 {
        CognitiveFrame64V0::new(
            KIND_TRUSTED_OBSERVATION,
            0x11,
            0x22,
            7,
            3,
            0x33,
            0x44,
            0b101,
            0x55,
            update_class,
            DECISION_QUARANTINE,
            TOOL_READ_FILE,
            tau,
            0,
        )
    }

    fn proposal(update_class: u8, decision: u8, tau: u8) -> CognitiveFrame64V0 {
        CognitiveFrame64V0::new(
            KIND_UNTRUSTED_PROPOSAL,
            0x11,
            0x22,
            7,
            3,
            0x33,
            0x44,
            0b101,
            0x66,
            update_class,
            decision,
            TOOL_READ_FILE,
            tau,
            0,
        )
    }

    #[test]
    fn all_runtime_transport_frames_are_64_bytes_or_multiples() {
        assert_eq!(size_of::<CognitiveFrame64V0>(), 64);
        assert_eq!(align_of::<CognitiveFrame64V0>(), 64);
        assert_eq!(size_of::<GateReceipt64V0>(), 64);
        assert_eq!(align_of::<GateReceipt64V0>(), 64);
        assert_eq!(size_of::<DecisionFrame64V1>(), 64);
        assert_eq!(align_of::<DecisionFrame64V1>(), 64);
        assert_eq!(size_of::<ToolAdmission64V1>(), 64);
        assert_eq!(align_of::<ToolAdmission64V1>(), 64);
        assert_eq!(size_of::<AgentStepV0>(), 192);
        assert_eq!(align_of::<AgentStepV0>(), 64);
    }

    #[test]
    fn unknown_quarantine_is_admitted_read_only() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);
        let receipt = evaluate_frame64(&obs, &prop);
        assert!(receipt.admitted_read_only());
        assert!(receipt.verify_integrity());
    }

    #[test]
    fn compatible_continue_is_admitted_read_only() {
        let obs = observation(UPDATE_COMPATIBLE, TAU_STABLE);
        let prop = proposal(UPDATE_COMPATIBLE, DECISION_CONTINUE, TAU_STABLE);
        assert!(evaluate_frame64(&obs, &prop).admitted_read_only());
    }

    #[test]
    fn semantic_break_requires_interrupt() {
        let obs = observation(UPDATE_SEMANTIC_BREAK, TAU_STABLE);
        let wrong = proposal(UPDATE_SEMANTIC_BREAK, DECISION_CONTINUE, TAU_STABLE);
        let receipt = evaluate_frame64(&obs, &wrong);
        assert_eq!(receipt.verdict, VERDICT_QUARANTINE);
        assert_ne!(receipt.reason_mask & REASON_DECISION_MISMATCH, 0);
    }

    #[test]
    fn tau_drop_requires_tau_drop_quarantine() {
        let obs = observation(UPDATE_COMPATIBLE, TAU_DROP);
        let prop = proposal(UPDATE_TAU_DROP, DECISION_QUARANTINE, TAU_DROP);
        assert!(evaluate_frame64(&obs, &prop).admitted_read_only());
    }

    #[test]
    fn cross_object_is_quarantined() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);
        prop.object_id = 0x99;
        prop.seal();
        let receipt = evaluate_frame64(&obs, &prop);
        assert_ne!(receipt.reason_mask & REASON_BINDING_MISMATCH, 0);
    }

    #[test]
    fn invariant_or_evidence_change_is_quarantined() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);
        prop.invariant_set_id ^= 1;
        prop.evidence_set_id ^= 1;
        prop.seal();
        let receipt = evaluate_frame64(&obs, &prop);
        assert_ne!(receipt.reason_mask & REASON_INVARIANT_MISMATCH, 0);
        assert_ne!(receipt.reason_mask & REASON_EVIDENCE_MISMATCH, 0);
    }

    #[test]
    fn model_cannot_erase_trusted_unknown_bits() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);
        prop.unknown_mask = 0b001;
        prop.seal();
        let receipt = evaluate_frame64(&obs, &prop);
        assert_ne!(receipt.reason_mask & REASON_UNKNOWN_ERASED, 0);
    }

    #[test]
    fn execution_and_publication_flags_are_quarantined() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);
        prop.flags = FLAG_CAN_EXECUTE | FLAG_PUBLICATION_REQUESTED;
        prop.seal();
        let receipt = evaluate_frame64(&obs, &prop);
        assert_ne!(receipt.reason_mask & REASON_EXECUTION_FORBIDDEN, 0);
        assert_ne!(receipt.reason_mask & REASON_PUBLICATION_FORBIDDEN, 0);
    }

    #[test]
    fn one_byte_mutation_breaks_integrity() {
        let frame = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);
        let mut bytes = frame.to_le_bytes();
        bytes[17] ^= 1;
        assert_ne!(crc32_ieee(&bytes[..60]), frame.integrity);
    }

    #[test]
    fn compact_line_decodes_without_json() {
        let line = "Z0|11|22|7|3|33|44|5|66|3|3|1|1|0";
        let frame = parse_compact_proposal(line).unwrap();
        assert_eq!(frame.task_id, 0x11);
        assert_eq!(frame.object_id, 0x22);
        assert_eq!(frame.unknown_mask, 0x5);
        assert_eq!(frame.update_class, UPDATE_UNKNOWN);
        assert_eq!(frame.decision, DECISION_QUARANTINE);
        assert!(frame.verify_integrity());
    }

    #[test]
    fn malformed_compact_line_fails_closed() {
        assert_eq!(
            parse_compact_proposal("not-zlp"),
            Err(CompactParseError::Prefix)
        );
    }

    #[test]
    fn decision_frame_is_bound_to_trusted_identity() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let decision_frame = parse_decision_frame("D0|3|3|1|66", &obs).unwrap();
        let prop = assemble_proposal(&obs, &decision_frame).unwrap();
        assert_eq!(prop.task_id, obs.task_id);
        assert_eq!(prop.object_id, obs.object_id);
        assert_eq!(prop.physical_version, obs.physical_version);
        assert_eq!(prop.semantic_epoch, obs.semantic_epoch);
        assert_eq!(prop.invariant_set_id, obs.invariant_set_id);
        assert_eq!(prop.evidence_set_id, obs.evidence_set_id);
        assert_eq!(prop.unknown_mask, obs.unknown_mask);
        assert_eq!(prop.tau_state, obs.tau_state);
        assert_eq!(prop.hypothesis_set_id, 0x66);
        assert!(evaluate_frame64(&obs, &prop).admitted_read_only());
    }

    #[test]
    fn wrong_decision_is_quarantined_after_assembly() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let decision_frame = parse_decision_frame("D0|3|1|0|0", &obs).unwrap();
        let prop = assemble_proposal(&obs, &decision_frame).unwrap();
        let receipt = evaluate_frame64(&obs, &prop);
        assert_ne!(receipt.reason_mask & REASON_DECISION_MISMATCH, 0);
    }

    #[test]
    fn decision_frame_from_another_observation_fails_closed() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut other = obs;
        other.object_id ^= 1;
        other.seal();
        let decision_frame =
            DecisionFrame64V1::new(&other, UPDATE_UNKNOWN, DECISION_QUARANTINE, TOOL_NONE, 0);

        assert_eq!(
            assemble_proposal(&obs, &decision_frame),
            Err(AssembleError::DecisionBinding)
        );
    }

    #[test]
    fn mutated_decision_frame_fails_integrity() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut decision_frame =
            DecisionFrame64V1::new(&obs, UPDATE_UNKNOWN, DECISION_QUARANTINE, TOOL_NONE, 0);
        decision_frame.object_id ^= 1;

        assert_eq!(
            assemble_proposal(&obs, &decision_frame),
            Err(AssembleError::DecisionIntegrity)
        );
    }

    #[test]
    fn malformed_decision_line_fails_closed() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        assert_eq!(
            parse_decision_frame("D0|3|3", &obs),
            Err(CompactParseError::FieldCount)
        );
        assert_eq!(
            parse_decision_frame("Z0|3|3|0|0", &obs),
            Err(CompactParseError::Prefix)
        );
    }

    #[test]
    fn tool_admission_is_one_private_64_byte_boundary() {
        assert_eq!(size_of::<ToolAdmission64V1>(), 64);
        assert_eq!(align_of::<ToolAdmission64V1>(), 64);
    }

    #[test]
    fn wrong_read_only_tool_is_quarantined() {
        let mut obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        obs.requested_tool = TOOL_SEARCH_FILES;
        obs.seal();
        let admission = tool_admission_from_observation(&obs);
        let prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);

        let receipt = evaluate_frame64_with_admission(&obs, &prop, &admission);

        assert!(!receipt.admitted_read_only());
        assert_ne!(receipt.reason_mask & REASON_TOOL_ADMISSION_MISMATCH, 0);
    }

    #[test]
    fn exact_admitted_tool_remains_read_only() {
        let mut obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        obs.requested_tool = TOOL_READ_FILE;
        obs.seal();
        let admission = tool_admission_from_observation(&obs);
        let prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);

        assert!(evaluate_frame64_with_admission(&obs, &prop, &admission).admitted_read_only());
    }

    #[test]
    fn stale_tool_admission_is_quarantined() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let admission = tool_admission_from_observation(&obs);
        let mut refreshed = obs;
        refreshed.physical_version += 1;
        refreshed.seal();
        let prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);

        let receipt = evaluate_frame64_with_admission(&refreshed, &prop, &admission);

        assert_ne!(receipt.reason_mask & REASON_TOOL_ADMISSION_BINDING, 0);
    }

    #[test]
    fn corrupt_tool_admission_is_quarantined() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut admission = tool_admission_from_observation(&obs);
        admission.object_id ^= 1;
        let prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);

        let receipt = evaluate_frame64_with_admission(&obs, &prop, &admission);

        assert_ne!(receipt.reason_mask & REASON_TOOL_ADMISSION_INTEGRITY, 0);
    }

    #[test]
    fn resealed_tool_substitution_is_quarantined() {
        let obs = observation(UPDATE_UNKNOWN, TAU_FORMING);
        let mut admission = tool_admission_from_observation(&obs);
        admission.allowed_tool = TOOL_SEARCH_FILES;
        admission.seal();
        let prop = proposal(UPDATE_UNKNOWN, DECISION_QUARANTINE, TAU_FORMING);

        let receipt = evaluate_frame64_with_admission(&obs, &prop, &admission);

        assert_ne!(receipt.reason_mask & REASON_TOOL_ADMISSION_BINDING, 0);
    }
}
