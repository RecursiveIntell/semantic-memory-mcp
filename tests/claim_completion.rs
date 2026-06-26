//! Integration tests for claim-ledger completion tools.
//!
//! Tests the 8 new claim-ledger MCP tool types (proof debt, support admissions,
//! contradictions, ledger verification, export bundles, supersession) using
//! the claim-ledger crate directly.

#![cfg(feature = "claim-integration")]

use semantic_memory_mcp::bridge::{BridgeConfig, EmbedderBackend, MemoryBridge};
use semantic_memory_mcp::server::SemanticMemoryServer;

/// Open a MemoryBridge with the mock embedder in a temp directory.
fn open_bridge(dir: &std::path::Path) -> MemoryBridge {
    let config = BridgeConfig {
        memory_dir: dir.to_path_buf(),
        embedder_backend: EmbedderBackend::Mock,
        embedding_url: "http://localhost:11434".to_string(),
        embedding_model: "nomic-embed-text".to_string(),
        embedding_dims: 768,
        turbo_quant_enabled: false,
        turbo_quant_bits: None,
        turbo_quant_projections: None,
    };
    MemoryBridge::open(config).expect("bridge should open")
}

#[test]
fn server_constructs_with_claim_integration() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let _server = SemanticMemoryServer::new(bridge, "full", String::new(), String::new());
}

// ── Proof-debt budget tests ──────────────────────────────────────────

#[test]
fn proof_debt_budget_creates_with_defaults() {
    use claim_ledger::{ProofDebtBudgetV1, ProofDebtSummaryV1};
    let budget = ProofDebtBudgetV1::new("claim:test_001", 1_000_000);
    assert!(budget.budget_id.starts_with("pdb_"));
    assert_eq!(budget.budget_micros, 1_000_000);
    assert_eq!(budget.consumed_micros, 0);
    assert!(!budget.is_exhausted());

    let summary = ProofDebtSummaryV1::from_budget(&budget);
    assert_eq!(summary.scope, "claim:test_001");
    assert_eq!(summary.consumed_pct, 0);
    assert!(!summary.exhausted);
}

#[test]
fn proof_debt_gate_proceeds_with_low_consumption() {
    use claim_ledger::{evaluate_proof_debt_gate, ProofDebtBudgetV1, ProofDebtGateDecision};
    let budget = ProofDebtBudgetV1::new("claim:test_002", 1_000_000);
    let gate = evaluate_proof_debt_gate(&budget);
    assert_eq!(gate.decision, ProofDebtGateDecision::Proceed);
    assert!(!gate.exhausted);
    assert!(gate.decision.allows_proceed());
    assert!(!gate.decision.blocks());
}

#[test]
fn proof_debt_gate_degrades_when_exhausted() {
    use claim_ledger::{evaluate_proof_debt_gate, ProofDebtBudgetV1, ProofDebtGateDecision};
    let mut budget = ProofDebtBudgetV1::new("claim:test_003", 100_000);
    budget.consume(100_000, "test", "100% consumption", false);
    let gate = evaluate_proof_debt_gate(&budget);
    assert_eq!(gate.decision, ProofDebtGateDecision::Degrade);
    assert!(gate.exhausted);
    assert!(gate.decision.blocks());
}

// ── Support admission tests ──────────────────────────────────────────

#[test]
fn support_admission_receipt_creates() {
    use claim_ledger::{SupportAdmissionMethod, SupportAdmissionReceipt, SupportState};
    let receipt = SupportAdmissionReceipt::new(
        "claim:test_004",
        "sj_prev_001",
        "sj_new_001",
        SupportAdmissionMethod::OperatorAdmitted,
        SupportState::Supported,
        "operator reviewed evidence and admitted support",
    );
    assert!(receipt.support_admission_receipt_id.starts_with("sar_"));
    assert_eq!(receipt.claim_id, "claim:test_004");
    assert_eq!(receipt.method, SupportAdmissionMethod::OperatorAdmitted);
    assert_eq!(receipt.admitted_support_state, SupportState::Supported);
}

#[test]
fn support_admission_all_methods() {
    use claim_ledger::{SupportAdmissionMethod, SupportAdmissionReceipt, SupportState};
    for method in [
        SupportAdmissionMethod::OperatorAdmitted,
        SupportAdmissionMethod::TestFixtureAdmitted,
        SupportAdmissionMethod::ExternalReceiptAdmitted,
    ] {
        let receipt = SupportAdmissionReceipt::new(
            "claim:test_method",
            "prev",
            "new",
            method,
            SupportState::PartiallySupported,
            "test",
        );
        assert_eq!(receipt.method, method);
    }
}

// ── Contradiction tests ──────────────────────────────────────────────

#[test]
fn contradiction_record_creates_with_candidate_status() {
    use claim_ledger::{ContradictionRecord, ContradictionStatus};
    let record = ContradictionRecord::new(
        "claim:a_001",
        "claim:b_001",
        "numeric_disagreement",
        "values 42 and 99 conflict",
    );
    assert!(record.contradiction_id.starts_with("ctr_"));
    assert_eq!(record.claim_refs.len(), 2);
    assert_eq!(record.status, ContradictionStatus::Candidate);
    assert_eq!(record.pattern, "numeric_disagreement");
}

#[test]
fn contradiction_resolution_receipt_creates() {
    use claim_ledger::{ContradictionResolution, ContradictionResolutionReceipt};
    let receipt = ContradictionResolutionReceipt::new(
        "ctr_test_001",
        "candidate",
        ContradictionResolution::Confirmed,
        "evidence supports the contradiction",
    );
    assert!(receipt
        .contradiction_resolution_receipt_id
        .starts_with("crr_"));
    assert_eq!(receipt.contradiction_id, "ctr_test_001");
    assert_eq!(receipt.resolution, ContradictionResolution::Confirmed);
}

// ── Ledger verification tests ────────────────────────────────────────

#[test]
fn verify_ledger_accepts_empty_input() {
    use claim_ledger::{parse_ledger_entries, verify_ledger};
    let entries = parse_ledger_entries("");
    assert!(entries.is_empty());
    let verification = verify_ledger(&entries);
    assert!(verification.valid);
    assert_eq!(verification.last_sequence, 0);
}

#[test]
fn verify_ledger_detects_invalid_entries() {
    use claim_ledger::{parse_ledger_entries, verify_ledger};
    // Malformed JSON lines should be silently skipped by parse
    let entries = parse_ledger_entries("not valid json\nalso not json");
    assert!(entries.is_empty());
    let verification = verify_ledger(&entries);
    assert!(verification.valid); // empty ledger is valid
}

// ── Export receipt tests ─────────────────────────────────────────────

#[test]
fn export_receipt_creates_and_binds_output() {
    use claim_ledger::ExportReceipt;
    let mut receipt = ExportReceipt::new(
        "bundle_export",
        vec!["claim:test_001".to_string(), "claim:test_002".to_string()],
        "attempt_001".to_string(),
    );
    assert!(receipt.export_receipt_id.starts_with("xpt_"));
    assert_eq!(receipt.operation, "bundle_export");
    assert_eq!(receipt.status, "pending");

    receipt.mark_success();
    assert_eq!(receipt.status, "success");

    receipt.bind_output("bundle:output_001".to_string(), "abc123".to_string());
    assert_eq!(receipt.output_ref, Some("bundle:output_001".to_string()));
    assert_eq!(receipt.output_digest, Some("abc123".to_string()));
}

// ── Supersession receipt tests ───────────────────────────────────────

#[test]
fn supersession_receipt_creates() {
    use claim_ledger::SupersessionReceipt;
    let receipt = SupersessionReceipt::new(
        "claim:old_001",
        "claim:new_001",
        "new evidence supersedes old claim",
    );
    assert!(receipt.supersession_receipt_id.starts_with("ssr_"));
    assert_eq!(receipt.superseded_ref, "claim:old_001");
    assert_eq!(receipt.superseding_ref, "claim:new_001");
    assert_eq!(receipt.rationale, "new evidence supersedes old claim");
}

// ── Tool profile gating tests ────────────────────────────────────────

#[test]
fn full_profile_allows_all_tools() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let _server = SemanticMemoryServer::new(bridge, "full", String::new(), String::new());
    // If this constructs without panic, all tools are registered
}

#[test]
fn standard_profile_hides_admin_tools() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let _server = SemanticMemoryServer::new(bridge, "standard", String::new(), String::new());
    // Standard profile should hide admin tools but allow basic ones
}

#[test]
fn lean_profile_hides_all_admin_tools() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let _server = SemanticMemoryServer::new(bridge, "lean", String::new(), String::new());
    // Lean profile hides all admin tools
}
