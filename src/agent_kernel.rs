//! Programmatic agent path from Lake references to the deterministic Gate.
//!
//! There is no prompt or JSON boundary here. The model module receives a
//! trusted fixed-size observation and borrowed immutable Lake slices, and may
//! return only one 64-byte decision frame.

use crate::{
    assemble_proposal, evaluate_frame64_with_admission, tool_admission_from_observation,
    AssembleError, CognitiveFrame64V0, DecisionFrame64V1, GateReceipt64V0, LakeReadViewV0,
    LakeRefError, LakeSpanRef64V1, ToolAdmission64V1, FRAME_MAGIC, KIND_TRUSTED_OBSERVATION,
    SCHEMA_VERSION,
};

/// The only input visible to a model implementation.
///
/// All fields are borrowed. Constructing this view does not allocate or copy
/// an observation, invariant body or evidence body.
pub struct ModelInputViewV0<'input> {
    observation: &'input CognitiveFrame64V0,
    invariants: &'input [u8],
    evidence: &'input [u8],
}

impl<'input> ModelInputViewV0<'input> {
    pub fn observation(&self) -> &'input CognitiveFrame64V0 {
        self.observation
    }

    pub fn invariants(&self) -> &'input [u8] {
        self.invariants
    }

    pub fn evidence(&self) -> &'input [u8] {
        self.evidence
    }
}

/// Program module implemented by a resident inference backend.
///
/// The backend cannot replace trusted identity fields. Its complete writable
/// result is one `DecisionFrame64V1` bound to the visible observation.
pub trait DecisionModuleV0 {
    type Error;

    fn decide(&mut self, input: ModelInputViewV0<'_>) -> Result<DecisionFrame64V1, Self::Error>;
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentStepV0 {
    pub proposal: CognitiveFrame64V0,
    pub tool_admission: ToolAdmission64V1,
    pub receipt: GateReceipt64V0,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentKernelError<ModelError> {
    ObservationIntegrity,
    ObservationSchema,
    InvariantSetMismatch,
    EvidenceSetMismatch,
    Lake(LakeRefError),
    Model(ModelError),
    Assemble(AssembleError),
}

/// Run one read-only agent step without payload copies.
///
/// Validation is deliberately completed before the model is called. A stale,
/// cross-Lake or wrongly bound reference therefore cannot consume inference
/// time or reach proposal assembly.
pub fn run_read_only_agent_v0<Model: DecisionModuleV0>(
    lake: &LakeReadViewV0<'_>,
    observation: &CognitiveFrame64V0,
    invariant_ref: &LakeSpanRef64V1,
    evidence_ref: &LakeSpanRef64V1,
    model: &mut Model,
) -> Result<AgentStepV0, AgentKernelError<Model::Error>> {
    if !observation.verify_integrity() {
        return Err(AgentKernelError::ObservationIntegrity);
    }
    if observation.magic != FRAME_MAGIC
        || observation.schema_version != SCHEMA_VERSION
        || observation.frame_kind != KIND_TRUSTED_OBSERVATION
    {
        return Err(AgentKernelError::ObservationSchema);
    }
    if invariant_ref.set_id() != observation.invariant_set_id {
        return Err(AgentKernelError::InvariantSetMismatch);
    }
    if evidence_ref.set_id() != observation.evidence_set_id {
        return Err(AgentKernelError::EvidenceSetMismatch);
    }

    let invariants = lake
        .resolve(invariant_ref)
        .map_err(AgentKernelError::Lake)?;
    let evidence = lake.resolve(evidence_ref).map_err(AgentKernelError::Lake)?;
    let delta = model
        .decide(ModelInputViewV0 {
            observation,
            invariants,
            evidence,
        })
        .map_err(AgentKernelError::Model)?;
    let proposal = assemble_proposal(observation, &delta).map_err(AgentKernelError::Assemble)?;
    let tool_admission = tool_admission_from_observation(observation);
    let receipt = evaluate_frame64_with_admission(observation, &proposal, &tool_admission);

    Ok(AgentStepV0 {
        proposal,
        tool_admission,
        receipt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DECISION_QUARANTINE, REASON_TOOL_ADMISSION_MISMATCH, TAU_FORMING, TOOL_NONE,
        TOOL_READ_FILE, TOOL_SEARCH_FILES, UPDATE_UNKNOWN,
    };

    const LAKE_INSTANCE: u64 = 0xa11c_e001;
    const GENERATION: u64 = 7;
    const INVARIANT_SET_ID: u64 = 0x33;
    const EVIDENCE_SET_ID: u64 = 0x44;

    fn observation() -> CognitiveFrame64V0 {
        CognitiveFrame64V0::new(
            KIND_TRUSTED_OBSERVATION,
            0x11,
            0x22,
            7,
            3,
            INVARIANT_SET_ID,
            EVIDENCE_SET_ID,
            0b101,
            0,
            UPDATE_UNKNOWN,
            DECISION_QUARANTINE,
            TOOL_NONE,
            TAU_FORMING,
            0,
        )
    }

    struct PointerCheckingModule {
        expected_invariants: *const u8,
        expected_evidence: *const u8,
        calls: usize,
    }

    impl DecisionModuleV0 for PointerCheckingModule {
        type Error = ();

        fn decide(
            &mut self,
            input: ModelInputViewV0<'_>,
        ) -> Result<DecisionFrame64V1, Self::Error> {
            self.calls += 1;
            assert_eq!(input.invariants().as_ptr(), self.expected_invariants);
            assert_eq!(input.evidence().as_ptr(), self.expected_evidence);
            assert_eq!(input.observation().object_id, 0x22);
            Ok(DecisionFrame64V1::new(
                input.observation(),
                UPDATE_UNKNOWN,
                DECISION_QUARANTINE,
                TOOL_NONE,
                0x66,
            ))
        }
    }

    fn fixture() -> (
        &'static [u8],
        LakeReadViewV0<'static>,
        LakeSpanRef64V1,
        LakeSpanRef64V1,
    ) {
        let bytes: &'static [u8] = b"stable invariant|stored evidence sentence";
        let lake = LakeReadViewV0::open(bytes, LAKE_INSTANCE, GENERATION).unwrap();
        let invariant_ref = lake.reference(INVARIANT_SET_ID, 0..16).unwrap();
        let evidence_ref = lake.reference(EVIDENCE_SET_ID, 17..bytes.len()).unwrap();
        (bytes, lake, invariant_ref, evidence_ref)
    }

    #[test]
    fn complete_agent_step_uses_borrowed_lake_bytes_and_decision_frame64() {
        let (bytes, lake, invariant_ref, evidence_ref) = fixture();
        let mut model = PointerCheckingModule {
            expected_invariants: bytes.as_ptr(),
            expected_evidence: bytes[17..].as_ptr(),
            calls: 0,
        };

        let step = run_read_only_agent_v0(
            &lake,
            &observation(),
            &invariant_ref,
            &evidence_ref,
            &mut model,
        )
        .unwrap();

        assert_eq!(model.calls, 1);
        assert_eq!(step.proposal.task_id, 0x11);
        assert_eq!(step.proposal.object_id, 0x22);
        assert_eq!(step.proposal.invariant_set_id, INVARIANT_SET_ID);
        assert_eq!(step.proposal.evidence_set_id, EVIDENCE_SET_ID);
        assert_eq!(step.proposal.hypothesis_set_id, 0x66);
        assert_eq!(step.tool_admission.allowed_tool(), TOOL_NONE);
        assert_eq!(step.receipt.allowed_tool, TOOL_NONE);
        assert_eq!(
            step.receipt.tool_admission_integrity,
            u32::from_le_bytes(
                step.tool_admission.to_le_bytes()[60..64]
                    .try_into()
                    .unwrap()
            )
        );
        assert!(step.receipt.admitted_read_only());
    }

    struct WrongReadOnlyToolModule;

    impl DecisionModuleV0 for WrongReadOnlyToolModule {
        type Error = ();

        fn decide(
            &mut self,
            input: ModelInputViewV0<'_>,
        ) -> Result<DecisionFrame64V1, Self::Error> {
            Ok(DecisionFrame64V1::new(
                input.observation(),
                UPDATE_UNKNOWN,
                DECISION_QUARANTINE,
                TOOL_READ_FILE,
                0,
            ))
        }
    }

    #[test]
    fn wrong_read_only_tool_is_quarantined_end_to_end() {
        let (_, lake, invariant_ref, evidence_ref) = fixture();
        let mut obs = observation();
        obs.requested_tool = TOOL_SEARCH_FILES;
        obs.seal();

        let step = run_read_only_agent_v0(
            &lake,
            &obs,
            &invariant_ref,
            &evidence_ref,
            &mut WrongReadOnlyToolModule,
        )
        .unwrap();

        assert_eq!(step.tool_admission.allowed_tool(), TOOL_SEARCH_FILES);
        assert_eq!(step.proposal.requested_tool, TOOL_READ_FILE);
        assert!(!step.receipt.admitted_read_only());
        assert_ne!(step.receipt.reason_mask & REASON_TOOL_ADMISSION_MISMATCH, 0);
    }

    #[test]
    fn wrong_set_binding_is_rejected_before_model_call() {
        let (bytes, lake, _, evidence_ref) = fixture();
        let wrong = lake.reference(0x99, 0..16).unwrap();
        let mut model = PointerCheckingModule {
            expected_invariants: bytes.as_ptr(),
            expected_evidence: bytes[17..].as_ptr(),
            calls: 0,
        };

        let result =
            run_read_only_agent_v0(&lake, &observation(), &wrong, &evidence_ref, &mut model);

        assert!(matches!(
            result,
            Err(AgentKernelError::InvariantSetMismatch)
        ));
        assert_eq!(model.calls, 0);
    }

    #[test]
    fn stale_reference_is_rejected_before_model_call() {
        let (bytes, _old_lake, invariant_ref, evidence_ref) = fixture();
        let current_lake = LakeReadViewV0::open(bytes, LAKE_INSTANCE, GENERATION + 1).unwrap();
        let mut model = PointerCheckingModule {
            expected_invariants: bytes.as_ptr(),
            expected_evidence: bytes[17..].as_ptr(),
            calls: 0,
        };
        let result = run_read_only_agent_v0(
            &current_lake,
            &observation(),
            &invariant_ref,
            &evidence_ref,
            &mut model,
        );

        assert!(matches!(
            result,
            Err(AgentKernelError::Lake(LakeRefError::StaleGeneration))
        ));
        assert_eq!(model.calls, 0);
    }

    #[test]
    fn corrupt_observation_is_rejected_before_model_call() {
        let (bytes, lake, invariant_ref, evidence_ref) = fixture();
        let mut corrupt = observation();
        corrupt.object_id ^= 1;
        let mut model = PointerCheckingModule {
            expected_invariants: bytes.as_ptr(),
            expected_evidence: bytes[17..].as_ptr(),
            calls: 0,
        };

        let result =
            run_read_only_agent_v0(&lake, &corrupt, &invariant_ref, &evidence_ref, &mut model);

        assert!(matches!(
            result,
            Err(AgentKernelError::ObservationIntegrity)
        ));
        assert_eq!(model.calls, 0);
    }

    struct FailingModule;

    impl DecisionModuleV0 for FailingModule {
        type Error = u8;

        fn decide(
            &mut self,
            _input: ModelInputViewV0<'_>,
        ) -> Result<DecisionFrame64V1, Self::Error> {
            Err(7)
        }
    }

    #[test]
    fn model_error_is_explicit_and_cannot_create_a_proposal() {
        let (_, lake, invariant_ref, evidence_ref) = fixture();
        let result = run_read_only_agent_v0(
            &lake,
            &observation(),
            &invariant_ref,
            &evidence_ref,
            &mut FailingModule,
        );

        assert_eq!(result, Err(AgentKernelError::Model(7)));
    }
}
