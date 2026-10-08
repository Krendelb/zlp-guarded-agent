use zlp_guarded_agent::{
    run_read_only_agent_v0, CognitiveFrame64V0, DecisionFrame64V1, DecisionModuleV0,
    LakeReadViewV0, ModelInputViewV0, DECISION_QUARANTINE, KIND_TRUSTED_OBSERVATION,
    REASON_TOOL_ADMISSION_MISMATCH, TAU_FORMING, TOOL_READ_FILE, TOOL_SEARCH_FILES, UPDATE_UNKNOWN,
};

// Deterministic stand-in for a model proposal. No model or tool is executed.
struct Proposal(u8);

impl DecisionModuleV0 for Proposal {
    type Error = ();

    fn decide(&mut self, input: ModelInputViewV0<'_>) -> Result<DecisionFrame64V1, ()> {
        Ok(DecisionFrame64V1::new(
            input.observation(),
            UPDATE_UNKNOWN,
            DECISION_QUARANTINE,
            self.0,
            0,
        ))
    }
}

fn main() {
    let payload = b"invariant|missing catalog location";
    let lake = LakeReadViewV0::open(payload, 1, 1).unwrap();
    let invariants = lake.reference(10, 0..9).unwrap();
    let evidence = lake.reference(20, 10..payload.len()).unwrap();
    let observation = CognitiveFrame64V0::new(
        KIND_TRUSTED_OBSERVATION,
        1,
        2,
        1,
        1,
        10,
        20,
        1,
        0,
        UPDATE_UNKNOWN,
        DECISION_QUARANTINE,
        TOOL_SEARCH_FILES,
        TAU_FORMING,
        0,
    );
    for (name, tool, expected) in [
        ("READ_FILE", TOOL_READ_FILE, false),
        ("SEARCH_FILES", TOOL_SEARCH_FILES, true),
    ] {
        let step = run_read_only_agent_v0(
            &lake,
            &observation,
            &invariants,
            &evidence,
            &mut Proposal(tool),
        )
        .unwrap();
        assert_eq!(step.receipt.admitted_read_only(), expected);
        if !expected {
            assert_ne!(step.receipt.reason_mask & REASON_TOOL_ADMISSION_MISMATCH, 0);
        }
        println!(
            "allowed=SEARCH_FILES proposed={name} verdict={}",
            if expected {
                "ADMIT_READ_ONLY"
            } else {
                "QUARANTINE"
            }
        );
    }
    println!("No tools executed. Both expected verdicts verified.");
}
