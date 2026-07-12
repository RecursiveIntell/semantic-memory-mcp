//! MCP server handler using rmcp's #[tool_router] macro.
//!
//! Each #[tool] method becomes an MCP tool that Hermes/Claude Desktop
//! can discover and call. The rmcp macro auto-generates JSON Schema
//! from the parameter structs in tools.rs.

use crate::bridge::MemoryBridge;
use crate::tools::*;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    tool, tool_handler, tool_router, ErrorData, ServerHandler,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::runtime::Handle;

static WITNESS_REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);
const ROUTING_POLICY_PERSIST_BATCH: usize = 10;

#[derive(Default)]
struct RoutingPolicyBatchState {
    policy: Option<semantic_memory::rl_routing::RoutingPolicy>,
    pending_outcomes: usize,
}

// Re-export the specific parameter types we use in tool signatures.
use crate::tools::{
    AddGraphEdgeParams, BenchmarkTrustParams, CommunityParams, FactorGraphParams,
    InvalidateGraphEdgeParams, ListGraphEdgesParams, RecordOutcomeParams, SearchProofDebtParams,
    SubgraphPruneParams, TopologyParams,
};

/// Process-local index from semantic-memory facts to claim-ledger support
/// judgments, consulted by search-time trust enrichment. Never a source of
/// truth itself — the claim ledger is; this is a pure in-memory projection
/// rebuilt from the ledger's hash-chained entries on every startup via
/// `rebuild_from_ledger`, so there is nothing here to persist independently.
#[cfg(feature = "claim-integration")]
#[derive(Default)]
struct ClaimTrustIndex {
    enabled: bool,
    fact_to_claim: HashMap<String, String>,
    claim_support: HashMap<String, claim_ledger::SupportState>,
    /// Reverse index from a claim's normalized content to its claim id, used
    /// to auto-link newly added facts whose content matches an existing
    /// claim without requiring an explicit sm_create_claim call.
    content_to_claim: HashMap<String, String>,
    /// Trigram → claim ids index over normalized claim content, used for
    /// fuzzy (Jaccard-similarity) auto-linking when no exact match exists.
    trigram_index: HashMap<String, Vec<String>>,
    /// Sequence number of the last ledger entry folded into this index, so
    /// `rebuild_from_ledger_incremental` only replays entries appended since
    /// the last checkpoint instead of rescanning the whole ledger.
    last_processed_sequence: u64,
}

#[cfg(feature = "claim-integration")]
impl ClaimTrustIndex {
    fn load_snapshot(&mut self, snapshot: &claim_ledger::LedgerSnapshot) {
        self.fact_to_claim.clear();
        self.claim_support.clear();
        self.content_to_claim.clear();
        self.trigram_index.clear();
        for link in &snapshot.content_to_claim_links {
            self.register_claim(link.claim_id.clone(), link.normalized_claim.clone());
        }
        for link in &snapshot.fact_to_claim_links {
            self.link_fact(link.fact_id.clone(), link.claim_id.clone());
        }
        for support in &snapshot.claim_support {
            self.record_judgment(support.claim_id.clone(), support.support_state);
        }
        self.last_processed_sequence = snapshot.last_compacted_sequence;
        self.enabled = true;
    }

    fn disable(&mut self) {
        *self = Self::default();
    }

    /// Fold a single ledger entry into the index. The ledger is the source
    /// of truth; this is the sole code path (used both at startup replay and
    /// at write time) that projects `ClaimAdded`, `SupportJudgment`, and
    /// `ContradictionCandidate` events into the process-local lookup cache.
    fn apply_entry(&mut self, entry: &claim_ledger::LedgerEntry) {
        use claim_ledger::{LedgerEvent, SupportState};
        match &entry.event {
            LedgerEvent::ClaimAdded {
                claim_id,
                source_id,
                normalized_claim,
                ..
            } => {
                if let Some(fact_id) = source_id.strip_prefix("semantic-memory:fact:") {
                    self.link_fact(fact_id.to_string(), claim_id.clone());
                }
                self.register_claim(claim_id.clone(), normalized_claim.clone());
            }
            LedgerEvent::SupportJudgment {
                claim_id,
                support_state,
                ..
            } => {
                self.record_judgment(claim_id.clone(), *support_state);
            }
            LedgerEvent::ContradictionCandidate { claim_refs, .. } => {
                for claim_ref in claim_refs {
                    self.record_judgment(claim_ref.clone(), SupportState::Contradicted);
                }
            }
            _ => {}
        }
        self.last_processed_sequence = entry.sequence;
    }

    /// Rebuild (or catch up) the index from the claim ledger's entries,
    /// replaying only entries with `sequence > last_processed_sequence`.
    /// On a fresh index this processes the whole ledger once; called again
    /// on an already-caught-up index it is O(new_entries).
    fn rebuild_from_ledger_incremental(&mut self, entries: &[claim_ledger::LedgerEntry]) {
        for entry in entries {
            if entry.sequence > self.last_processed_sequence {
                self.apply_entry(entry);
            }
        }
    }

    fn link_fact(&mut self, bare_fact_id: String, claim_id: String) {
        self.fact_to_claim.insert(bare_fact_id, claim_id);
    }

    /// Record a claim's normalized content in the reverse and trigram
    /// indexes so future facts with matching (or fuzzy-matching) content can
    /// be auto-linked without an explicit sm_create_claim call.
    fn register_claim(&mut self, claim_id: String, normalized_content: String) {
        if normalized_content.is_empty() {
            return;
        }
        for trigram in Self::trigrams(&normalized_content) {
            self.trigram_index
                .entry(trigram)
                .or_default()
                .push(claim_id.clone());
        }
        self.content_to_claim.insert(normalized_content, claim_id);
    }

    /// Character 3-grams of `text`, lowercased. Text shorter than 3 chars
    /// yields the whole (lowercased) text as its only "trigram".
    fn trigrams(text: &str) -> HashSet<String> {
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() < 3 {
            let mut set = HashSet::new();
            if !lower.is_empty() {
                set.insert(lower);
            }
            return set;
        }
        chars
            .windows(3)
            .map(|w| w.iter().collect::<String>())
            .collect()
    }

    fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        let intersection = a.intersection(b).count();
        let union = a.union(b).count();
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Best-effort: if `bare_fact_id` isn't already linked to a claim, look
    /// for the best fuzzy (trigram-Jaccard) content match among known
    /// claims, falling back to an exact normalized-text match. No-op if the
    /// fact is already linked or no candidate reaches the similarity bar.
    fn auto_link_content(&mut self, bare_fact_id: &str, content: &str) -> Option<String> {
        if !self.enabled {
            return None;
        }
        if self.fact_to_claim.contains_key(bare_fact_id) {
            return None;
        }
        let normalized = claim_ledger::ids::normalize_text(content);
        if normalized.is_empty() {
            return None;
        }

        const SIMILARITY_THRESHOLD: f64 = 0.7;
        let query_trigrams = Self::trigrams(&normalized);
        let mut candidate_ids: HashSet<&String> = HashSet::new();
        for trigram in &query_trigrams {
            if let Some(ids) = self.trigram_index.get(trigram) {
                candidate_ids.extend(ids.iter());
            }
        }

        let mut best: Option<(String, f64)> = None;
        for claim_id in &candidate_ids {
            if let Some((existing_content, existing_claim_id)) = self
                .content_to_claim
                .iter()
                .find(|(_, cid)| *cid == *claim_id)
            {
                let similarity = Self::jaccard(&query_trigrams, &Self::trigrams(existing_content));
                if best.as_ref().map(|(_, s)| similarity > *s).unwrap_or(true) {
                    best = Some((existing_claim_id.clone(), similarity));
                }
            }
        }

        let claim_id = match best {
            Some((claim_id, similarity)) if similarity >= SIMILARITY_THRESHOLD => claim_id,
            _ => self.content_to_claim.get(&normalized)?.clone(),
        };
        self.link_fact(bare_fact_id.to_string(), claim_id.clone());
        Some(claim_id)
    }

    fn record_judgment(&mut self, claim_id: String, state: claim_ledger::SupportState) {
        self.claim_support.insert(claim_id, state);
    }

    fn trust_for_fact(&self, bare_fact_id: &str) -> String {
        if !self.enabled {
            return "trust_enrichment_disabled".to_string();
        }
        let Some(claim_id) = self.fact_to_claim.get(bare_fact_id) else {
            return "persisted_unjudged".to_string();
        };
        let Some(state) = self.claim_support.get(claim_id) else {
            return "persisted_unjudged".to_string();
        };
        support_state_label(*state).to_string()
    }

    /// The claim linked to this fact (if any) and its recorded support
    /// judgment (if any judgment has been made yet). Distinguishes "no claim
    /// at all" (no proof debt to speak of) from "claim exists but unjudged"
    /// (real proof debt, `SupportState::Unknown`).
    fn claim_and_state_for_fact(
        &self,
        bare_fact_id: &str,
    ) -> Option<(String, Option<claim_ledger::SupportState>)> {
        if !self.enabled {
            return None;
        }
        let claim_id = self.fact_to_claim.get(bare_fact_id)?;
        Some((claim_id.clone(), self.claim_support.get(claim_id).copied()))
    }
}

/// Map a claim's support judgment to the proof-debt obligations it incurs.
/// Supported/PartiallySupported claims carry no debt. Unjudged, heuristic, or
/// unsupported claims lack a source basis. Contradicted claims carry that
/// same missing-source-basis debt plus a missing-repro debt, so the real
/// claim-ledger weights (not a hardcoded number) rank them strictly worse.
#[cfg(feature = "claim-integration")]
fn proof_debts_for_support_state(
    state: claim_ledger::SupportState,
) -> Vec<claim_ledger::ProofDebt> {
    use claim_ledger::{ProofDebt, SupportState};
    match state {
        SupportState::Supported | SupportState::PartiallySupported => Vec::new(),
        SupportState::Unknown | SupportState::HeuristicOnly | SupportState::Unsupported => {
            vec![ProofDebt::MissingSourceBasis]
        }
        SupportState::Contradicted => vec![ProofDebt::MissingSourceBasis, ProofDebt::MissingRepro],
    }
}

#[cfg(feature = "claim-integration")]
#[derive(serde::Serialize, serde::Deserialize)]
struct ClaimLedgerManifest {
    format_version: String,
    generation: String,
    snapshot_path: String,
    tail_path: String,
    receipt_path: String,
}

#[cfg(feature = "claim-integration")]
struct ClaimLedgerCompactionConfig {
    dry_run: bool,
    max_entries: usize,
    max_bytes: u64,
    retain_tail_entries: usize,
    max_backups: usize,
}

/// Hash-chained claim-ledger store. Before the first compaction `path` is the
/// legacy JSONL file. Afterwards an atomically replaced manifest selects one
/// verified snapshot/retained-tail generation; the snapshot is a checkpoint,
/// never an independent source of truth.
#[cfg(feature = "claim-integration")]
struct ClaimLedgerStore {
    entries: Vec<claim_ledger::LedgerEntry>,
    path: std::path::PathBuf,
    legacy_path: std::path::PathBuf,
    snapshot: Option<claim_ledger::LedgerSnapshot>,
    receipt: Option<claim_ledger::CompactionReceipt>,
    trust_enabled: bool,
}

#[cfg(feature = "claim-integration")]
impl ClaimLedgerStore {
    fn open(path: std::path::PathBuf) -> Self {
        match Self::open_verified(&path) {
            Ok(store) => store,
            Err(error) => {
                tracing::error!(error = %error, "CRITICAL: claim ledger verification failed; trust enrichment disabled");
                Self {
                    entries: Vec::new(),
                    path: path.clone(),
                    legacy_path: path,
                    snapshot: None,
                    receipt: None,
                    trust_enabled: false,
                }
            }
        }
    }

    fn open_verified(legacy_path: &std::path::Path) -> Result<Self, String> {
        let memory_dir = legacy_path
            .parent()
            .ok_or("claim ledger has no parent directory")?;
        let manifest_path = memory_dir.join("claim_ledger.active_compaction.json");
        if manifest_path.exists() {
            let manifest: ClaimLedgerManifest = serde_json::from_slice(
                &std::fs::read(&manifest_path)
                    .map_err(|e| format!("failed to read compaction manifest: {e}"))?,
            )
            .map_err(|e| format!("invalid compaction manifest: {e}"))?;
            if manifest.format_version != "claim-ledger.active-compaction.v1" {
                return Err(format!(
                    "unsupported compaction manifest format {}",
                    manifest.format_version
                ));
            }
            let snapshot_path = Self::resolve_relative(memory_dir, &manifest.snapshot_path)?;
            let tail_path = Self::resolve_relative(memory_dir, &manifest.tail_path)?;
            let receipt_path = Self::resolve_relative(memory_dir, &manifest.receipt_path)?;
            let snapshot: claim_ledger::LedgerSnapshot = serde_json::from_slice(
                &std::fs::read(&snapshot_path)
                    .map_err(|e| format!("failed to read ledger snapshot: {e}"))?,
            )
            .map_err(|e| format!("invalid ledger snapshot: {e}"))?;
            let receipt: claim_ledger::CompactionReceipt = serde_json::from_slice(
                &std::fs::read(&receipt_path)
                    .map_err(|e| format!("failed to read compaction receipt: {e}"))?,
            )
            .map_err(|e| format!("invalid compaction receipt: {e}"))?;
            let tail_contents = std::fs::read_to_string(&tail_path)
                .map_err(|e| format!("failed to read retained ledger tail: {e}"))?;
            let entries = claim_ledger::parse_ledger_entries(&tail_contents)
                .map_err(|e| format!("invalid retained ledger tail: {e}"))?;
            let verification = claim_ledger::verify_compaction(&snapshot, &entries, &receipt)
                .map_err(|e| e.to_string())?;
            tracing::info!(
                generation = %manifest.generation,
                entries = verification.last_sequence,
                "claim ledger snapshot and retained tail verified"
            );
            return Ok(Self {
                entries,
                path: tail_path,
                legacy_path: legacy_path.to_path_buf(),
                snapshot: Some(snapshot),
                receipt: Some(receipt),
                trust_enabled: true,
            });
        }

        let contents = match std::fs::read_to_string(legacy_path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(format!("failed to read claim ledger: {error}")),
        };
        let entries = claim_ledger::parse_ledger_entries(&contents)
            .map_err(|e| format!("invalid claim ledger: {e}"))?;
        let expected_head = match entries.last() {
            None => claim_ledger::ExpectedLedgerHead::Empty,
            Some(last) => claim_ledger::ExpectedLedgerHead::Entry {
                sequence: last.sequence,
                entry_digest: last.entry_digest.clone(),
            },
        };
        let verification =
            claim_ledger::verify_ledger(&entries, &expected_head).map_err(|e| e.to_string())?;
        tracing::info!(
            entries = verification.last_sequence,
            "claim ledger verified"
        );
        Ok(Self {
            entries,
            path: legacy_path.to_path_buf(),
            legacy_path: legacy_path.to_path_buf(),
            snapshot: None,
            receipt: None,
            trust_enabled: true,
        })
    }

    fn resolve_relative(
        base: &std::path::Path,
        relative: &str,
    ) -> Result<std::path::PathBuf, String> {
        let path = std::path::Path::new(relative);
        if path.is_absolute()
            || path.components().any(|component| {
                !matches!(
                    component,
                    std::path::Component::Normal(_) | std::path::Component::CurDir
                )
            })
        {
            return Err(format!("unsafe path in compaction manifest: {relative}"));
        }
        Ok(base.join(path))
    }

    fn next_sequence(&self) -> u64 {
        self.entries
            .last()
            .map(|entry| entry.sequence)
            .or_else(|| {
                self.snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.last_compacted_sequence)
            })
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn last_digest(&self) -> Option<String> {
        self.entries
            .last()
            .map(|entry| entry.entry_digest.clone())
            .or_else(|| {
                self.snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.last_compacted_entry_digest.clone())
            })
    }

    /// Append a new entry to the in-memory chain and the backing file.
    /// Returns the entry's digest on success.
    fn append(&mut self, entry: claim_ledger::LedgerEntry) -> Result<String, String> {
        if !self.trust_enabled {
            return Err("claim ledger is disabled after verification failure".into());
        }
        if entry.sequence != self.next_sequence()
            || entry.previous_entry_digest != self.last_digest()
        {
            return Err("claim ledger append does not continue the verified head".into());
        }
        let line = claim_ledger::serialize_entry(&entry).map_err(|e| e.to_string())?;
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("failed to open claim ledger file: {e}"))?;
        writeln!(file, "{line}").map_err(|e| format!("failed to write claim ledger entry: {e}"))?;
        file.sync_data()
            .map_err(|e| format!("failed to fsync claim ledger entry: {e}"))?;
        let digest = entry.entry_digest.clone();
        self.entries.push(entry);
        Ok(digest)
    }

    fn compact(
        &mut self,
        config: ClaimLedgerCompactionConfig,
    ) -> Result<serde_json::Value, String> {
        if !self.trust_enabled {
            return Err("claim ledger trust is disabled; refusing compaction".into());
        }
        let current_bytes = std::fs::metadata(&self.path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let threshold_exceeded =
            self.entries.len() > config.max_entries || current_bytes > config.max_bytes;
        if !threshold_exceeded {
            return Ok(serde_json::json!({
                "ok": true,
                "dry_run": config.dry_run,
                "compacted": false,
                "reason": "threshold_not_exceeded",
                "entries": self.entries.len(),
                "bytes": current_bytes,
                "max_entries": config.max_entries,
                "max_bytes": config.max_bytes,
            }));
        }

        let compacted = claim_ledger::compact_ledger_from_snapshot(
            self.snapshot.as_ref(),
            &self.entries,
            &claim_ledger::CompactionPolicy {
                retain_tail_entries: config.retain_tail_entries,
                unprojectable_events: claim_ledger::UnprojectableEventPolicy::Retain,
            },
        )
        .map_err(|e| e.to_string())?;
        let compacted_entries = self
            .entries
            .len()
            .saturating_sub(compacted.retained_tail.len());
        let result = serde_json::json!({
            "ok": true,
            "dry_run": config.dry_run,
            "compacted": !config.dry_run && compacted_entries > 0,
            "entries_before": self.entries.len(),
            "bytes_before": current_bytes,
            "entries_checkpointed": compacted_entries,
            "retained_tail_entries": compacted.retained_tail.len(),
            "snapshot_sequence": compacted.snapshot.last_compacted_sequence,
            "snapshot_digest": compacted.snapshot.snapshot_digest,
            "receipt": compacted.receipt,
        });
        if config.dry_run || compacted_entries == 0 {
            return Ok(result);
        }

        let memory_dir = self
            .legacy_path
            .parent()
            .ok_or("claim ledger has no parent directory")?;
        let generation = compacted.receipt.receipt_digest[..16].to_string();
        let generations_dir = memory_dir.join("claim_ledger_generations");
        std::fs::create_dir_all(&generations_dir)
            .map_err(|e| format!("failed to create ledger generation directory: {e}"))?;
        let final_dir = generations_dir.join(&generation);
        let temp_dir = generations_dir.join(format!(".tmp-{generation}"));
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir)
                .map_err(|e| format!("failed to clear stale compaction temp directory: {e}"))?;
        }
        std::fs::create_dir(&temp_dir)
            .map_err(|e| format!("failed to create compaction temp directory: {e}"))?;

        let snapshot_bytes = serde_json::to_vec_pretty(&compacted.snapshot)
            .map_err(|e| format!("failed to serialize ledger snapshot: {e}"))?;
        let receipt_bytes = serde_json::to_vec_pretty(&compacted.receipt)
            .map_err(|e| format!("failed to serialize compaction receipt: {e}"))?;
        let mut tail_bytes = Vec::new();
        for entry in &compacted.retained_tail {
            tail_bytes.extend_from_slice(
                claim_ledger::serialize_entry(entry)
                    .map_err(|e| e.to_string())?
                    .as_bytes(),
            );
            tail_bytes.push(b'\n');
        }
        Self::write_synced(&temp_dir.join("snapshot.json"), &snapshot_bytes)?;
        Self::write_synced(&temp_dir.join("tail.jsonl"), &tail_bytes)?;
        Self::write_synced(&temp_dir.join("receipt.json"), &receipt_bytes)?;
        Self::sync_directory(&temp_dir)?;
        std::fs::rename(&temp_dir, &final_dir)
            .map_err(|e| format!("failed to publish ledger generation: {e}"))?;
        Self::sync_directory(&generations_dir)?;

        let manifest = ClaimLedgerManifest {
            format_version: "claim-ledger.active-compaction.v1".into(),
            generation: generation.clone(),
            snapshot_path: format!("claim_ledger_generations/{generation}/snapshot.json"),
            tail_path: format!("claim_ledger_generations/{generation}/tail.jsonl"),
            receipt_path: format!("claim_ledger_generations/{generation}/receipt.json"),
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| format!("failed to serialize compaction manifest: {e}"))?;
        let manifest_path = memory_dir.join("claim_ledger.active_compaction.json");
        let manifest_temp = memory_dir.join(format!(".claim_ledger.manifest.{generation}.tmp"));
        Self::write_synced(&manifest_temp, &manifest_bytes)?;
        std::fs::rename(&manifest_temp, &manifest_path)
            .map_err(|e| format!("failed to atomically activate compaction: {e}"))?;
        Self::sync_directory(memory_dir)?;

        let previous_path = self.path.clone();
        self.path = final_dir.join("tail.jsonl");
        self.entries = compacted.retained_tail;
        self.snapshot = Some(compacted.snapshot);
        self.receipt = Some(compacted.receipt);

        if previous_path == self.legacy_path && previous_path.exists() {
            let backups_dir = memory_dir.join("claim_ledger_backups");
            if std::fs::create_dir_all(&backups_dir).is_ok() {
                let backup_path = backups_dir.join(format!("{generation}-pre-compaction.jsonl"));
                if let Err(error) = std::fs::rename(&previous_path, &backup_path) {
                    tracing::error!(error = %error, "failed to rotate legacy claim ledger backup");
                }
            }
        }
        Self::rotate_backups(memory_dir, &generation, config.max_backups);
        Ok(result)
    }

    fn write_synced(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("failed to create {}: {e}", path.display()))?;
        file.write_all(bytes)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
        file.sync_all()
            .map_err(|e| format!("failed to fsync {}: {e}", path.display()))
    }

    fn sync_directory(path: &std::path::Path) -> Result<(), String> {
        std::fs::File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(|e| format!("failed to fsync directory {}: {e}", path.display()))
    }

    fn rotate_backups(memory_dir: &std::path::Path, active: &str, max_backups: usize) {
        let generations_dir = memory_dir.join("claim_ledger_generations");
        let mut generations: Vec<_> = std::fs::read_dir(&generations_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
            .filter(|entry| entry.file_name() != active)
            .collect();
        generations.sort_by_key(|entry| {
            entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        let remove_count = generations.len().saturating_sub(max_backups);
        for entry in generations.into_iter().take(remove_count) {
            if let Err(error) = std::fs::remove_dir_all(entry.path()) {
                tracing::error!(error = %error, "failed to remove old claim-ledger generation");
            }
        }

        let backups_dir = memory_dir.join("claim_ledger_backups");
        let mut backups: Vec<_> = std::fs::read_dir(&backups_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| {
                entry
                    .file_type()
                    .map(|kind| kind.is_file())
                    .unwrap_or(false)
            })
            .collect();
        backups.sort_by_key(|entry| {
            entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        let remove_count = backups.len().saturating_sub(max_backups);
        for entry in backups.into_iter().take(remove_count) {
            if let Err(error) = std::fs::remove_file(entry.path()) {
                tracing::error!(error = %error, "failed to remove old claim-ledger backup");
            }
        }
    }
}

#[cfg(feature = "claim-integration")]
fn support_state_label(state: claim_ledger::SupportState) -> &'static str {
    use claim_ledger::SupportState;
    match state {
        SupportState::Supported => "supported",
        SupportState::PartiallySupported => "partially_supported",
        SupportState::Unsupported => "unsupported",
        SupportState::Contradicted => "contradicted",
        SupportState::HeuristicOnly => "heuristic_only",
        SupportState::Unknown => "persisted_unjudged",
    }
}

pub struct SemanticMemoryServer {
    bridge: Arc<MemoryBridge>,
    tool_router: ToolRouter<Self>,
    routing_policy_batch: Mutex<RoutingPolicyBatchState>,
    #[cfg(feature = "claim-integration")]
    claim_trust: Mutex<ClaimTrustIndex>,
    #[cfg(feature = "claim-integration")]
    claim_ledger_store: Mutex<ClaimLedgerStore>,
}

impl SemanticMemoryServer {
    pub fn new(bridge: MemoryBridge, tool_profile: &str) -> Self {
        let mut router = Self::tool_router();

        match tool_profile {
            "full" => { /* all tools visible */ }
            "stable" => {
                let allowed: HashSet<&str> = [
                    "sm_search",
                    "sm_search_witnessed",
                    "sm_stats",
                    "sm_list_namespaces",
                    "sm_get_fact",
                    "sm_get_fact_neighbors",
                    "sm_graph_path",
                    "sm_search_conversations",
                    "sm_add_fact",
                    "sm_supersede_fact",
                    "sm_add_graph_edge",
                    "sm_decide_assertion_authority",
                    "sm_decide_action_authority",
                ]
                .into_iter()
                .collect();
                let names: Vec<_> = router
                    .list_all()
                    .into_iter()
                    .map(|tool| tool.name.into_owned())
                    .collect();
                for name in names {
                    if !allowed.contains(name.as_str()) {
                        router.disable_route(name);
                    }
                }
            }
            "lean" => {
                // Autonomous/lean: governed read-only surface only.
                let allowed: HashSet<&str> = [
                    "sm_search_witnessed",
                    "sm_replay_search",
                    "sm_decide_assertion_authority",
                    "sm_decide_action_authority",
                ]
                .into_iter()
                .collect();
                let names: Vec<_> = router
                    .list_all()
                    .into_iter()
                    .map(|tool| tool.name.into_owned())
                    .collect();
                for name in names {
                    if !allowed.contains(name.as_str()) {
                        router.disable_route(name);
                    }
                }
            }
            "standard" => {
                let allowed: HashSet<&str> = [
                    "sm_search",
                    "sm_search_witnessed",
                    "sm_replay_search",
                    "sm_stats",
                    "sm_list_namespaces",
                    "sm_get_fact",
                    "sm_get_fact_neighbors",
                    "sm_graph_path",
                    "sm_search_conversations",
                    "sm_add_fact",
                    "sm_supersede_fact",
                    "sm_add_graph_edge",
                    "sm_decide_assertion_authority",
                    "sm_decide_action_authority",
                    "sm_update_fact",
                    "sm_set_provenance",
                    "sm_list_facts",
                ]
                .into_iter()
                .collect();
                let names: Vec<_> = router
                    .list_all()
                    .into_iter()
                    .map(|tool| tool.name.into_owned())
                    .collect();
                for name in names {
                    if !allowed.contains(name.as_str()) {
                        router.disable_route(name);
                    }
                }
            }
            "agent" => {
                let allowed: HashSet<&str> = [
                    "sm_add_fact",
                    "sm_add_graph_edge",
                    "sm_decide_action_authority",
                    "sm_decide_assertion_authority",
                    "sm_get_fact",
                    "sm_get_fact_neighbors",
                    "sm_get_search_receipt",
                    "sm_graph_path",
                    "sm_list_namespaces",
                    "sm_replay_search",
                    "sm_search_conversations",
                    "sm_search_witnessed",
                    "sm_set_provenance",
                    "sm_stats",
                    "sm_supersede_fact",
                    "sm_update_fact",
                ]
                .into_iter()
                .collect();
                let names: Vec<_> = router
                    .list_all()
                    .into_iter()
                    .map(|tool| tool.name.into_owned())
                    .collect();
                for name in names {
                    if !allowed.contains(name.as_str()) {
                        router.disable_route(name);
                    }
                }
            }
            _ => panic!(
                "Unknown tool profile '{}'. Must be one of: stable, lean, standard, agent, full",
                tool_profile
            ),
        }

        eprintln!(
            "Tool profile: {} ({} tools visible)",
            tool_profile,
            router.list_all().len()
        );

        #[cfg(feature = "claim-integration")]
        let claim_ledger_store =
            ClaimLedgerStore::open(bridge.memory_dir.join("claim_ledger.jsonl"));
        #[cfg(feature = "claim-integration")]
        let mut claim_trust = ClaimTrustIndex::default();
        #[cfg(feature = "claim-integration")]
        if claim_ledger_store.trust_enabled {
            claim_trust.enabled = true;
            if let Some(snapshot) = &claim_ledger_store.snapshot {
                claim_trust.load_snapshot(snapshot);
            }
            claim_trust.rebuild_from_ledger_incremental(&claim_ledger_store.entries);
        } else {
            claim_trust.disable();
        }

        Self {
            bridge: Arc::new(bridge),
            tool_router: router,
            routing_policy_batch: Mutex::new(RoutingPolicyBatchState::default()),
            #[cfg(feature = "claim-integration")]
            claim_trust: Mutex::new(claim_trust),
            #[cfg(feature = "claim-integration")]
            claim_ledger_store: Mutex::new(claim_ledger_store),
        }
    }

    pub fn exposes_tool(&self, name: &str) -> bool {
        self.tool_router
            .list_all()
            .iter()
            .any(|tool| tool.name == name)
    }

    pub fn exposed_tool_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .tool_router
            .list_all()
            .into_iter()
            .map(|tool| tool.name.into_owned())
            .collect();
        names.sort();
        names
    }

    pub fn tool_annotations(&self, name: &str) -> Option<rmcp::model::ToolAnnotations> {
        self.tool_router
            .list_all()
            .into_iter()
            .find(|tool| tool.name == name)
            .and_then(|tool| tool.annotations)
    }

    fn decide_governed_authority(
        &self,
        params: GovernedDecisionParams,
        purpose: semantic_memory::GovernedAccessPurposeV1,
    ) -> Result<String, ErrorData> {
        use semantic_memory::{
            AudienceV1, CallerPrincipalV1, DelegationElevationLeaseV1, GovernedAccessPurposeV1,
            GovernedAccessRequestV1, NamespaceScopeV1, SubjectPrincipalV1,
        };

        let GovernedDecisionParams {
            fact_id,
            caller,
            subject,
            audiences,
            scope,
            delegation_or_elevation,
        } = params;
        let caller = CallerPrincipalV1::new(caller)
            .map_err(|error| ErrorData::invalid_params(error, None))?;
        let subject = SubjectPrincipalV1::new(subject)
            .map_err(|error| ErrorData::invalid_params(error, None))?;
        let scope = NamespaceScopeV1 {
            namespace: scope.namespace,
            domain: scope.domain,
            workspace_id: scope.workspace_id,
            repo_id: scope.repo_id,
        };
        let mut request =
            GovernedAccessRequestV1::for_principals(caller, subject, audiences, purpose, scope);
        if let Some(lease) = delegation_or_elevation {
            let lease_scope = NamespaceScopeV1 {
                namespace: lease.scope.namespace,
                domain: lease.scope.domain,
                workspace_id: lease.scope.workspace_id,
                repo_id: lease.scope.repo_id,
            };
            let purposes = lease
                .purposes
                .into_iter()
                .map(|purpose| match purpose {
                    GovernedAccessPurposeParam::Recall => GovernedAccessPurposeV1::Recall,
                    GovernedAccessPurposeParam::Assertion => GovernedAccessPurposeV1::Assertion,
                    GovernedAccessPurposeParam::Action => GovernedAccessPurposeV1::Action,
                    GovernedAccessPurposeParam::Export => GovernedAccessPurposeV1::Export,
                    GovernedAccessPurposeParam::Replay => GovernedAccessPurposeV1::Replay,
                    GovernedAccessPurposeParam::Admin => GovernedAccessPurposeV1::Admin,
                })
                .collect();
            request = request.with_delegation_or_elevation(DelegationElevationLeaseV1 {
                lease_id: lease.lease_id,
                delegator: SubjectPrincipalV1::new(lease.delegator)
                    .map_err(|error| ErrorData::invalid_params(error, None))?,
                delegatee: CallerPrincipalV1::new(lease.delegatee)
                    .map_err(|error| ErrorData::invalid_params(error, None))?,
                purposes,
                scope: lease_scope,
                audience: AudienceV1::new(lease.audiences),
                expires_at: lease.expires_at,
                revoked: lease.revoked,
                elevation: lease.elevation,
            });
        }

        let fact_id = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let access = tokio::task::block_in_place(|| {
            Handle::current().block_on(
                self.bridge
                    .store
                    .authority()
                    .get_fact_governed(&fact_id, request),
            )
        })
        .map_err(|error| {
            ErrorData::internal_error(format!("governed authority decision error: {error}"), None)
        })?;

        // Deliberately serialize only the canonical typed receipt. `access.fact`
        // and `access.origin` are never part of this MCP decision surface.
        serde_json::to_string(&access.decision).map_err(|error| {
            ErrorData::internal_error(
                format!("decision receipt serialization error: {error}"),
                None,
            )
        })
    }
}

/// Helper: load all stored graph edges from the store as GraphEdgeRef tuples
/// for discord scoring.
fn load_stored_edge_refs(
    store: &semantic_memory::MemoryStore,
) -> Result<Vec<semantic_memory::discord::GraphEdgeRef>, ErrorData> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let refs = edges
        .iter()
        .filter_map(|edge| {
            let parsed_type = match edge
                .edge_type_parsed
                .clone()
                .or_else(|| serde_json::from_str(&edge.edge_type).ok())
            {
                Some(t) => t,
                None => {
                    eprintln!("WARNING: Skipping edge with unparseable type between {} and {} (edge_type={})", edge.source, edge.target, edge.edge_type);
                    return None;
                }
            };
            let type_str = match parsed_type {
                semantic_memory::GraphEdgeType::Semantic { .. } => "semantic",
                semantic_memory::GraphEdgeType::Temporal { .. } => "temporal",
                semantic_memory::GraphEdgeType::Causal { .. } => "causal",
                semantic_memory::GraphEdgeType::Entity { .. } => "entity",
            };
            Some(semantic_memory::discord::GraphEdgeRef {
                source: edge.source.clone(),
                target: edge.target.clone(),
                edge_type: type_str.to_string(),
                weight: edge.weight,
            })
        })
        .collect();
    Ok(refs)
}

/// Helper: load all stored graph edges from the store as raw factor graph
/// edge tuples (source, target, GraphEdgeType, weight, metadata_json).
fn load_stored_factor_edges(
    store: &semantic_memory::MemoryStore,
) -> Result<Vec<FactorEdgeTuple>, ErrorData> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let raw = edges
        .iter()
        .map(|edge| {
            let parsed_type = edge
                .edge_type_parsed
                .clone()
                .or_else(|| serde_json::from_str(&edge.edge_type).ok())
                .unwrap_or(semantic_memory::GraphEdgeType::Entity {
                    relation: "unknown".to_string(),
                });
            (
                edge.source.clone(),
                edge.target.clone(),
                parsed_type,
                edge.weight,
                edge.metadata.clone(),
            )
        })
        .collect();
    Ok(raw)
}

/// Helper: load all stored graph edges as (source, target) pairs.
fn load_stored_edge_pairs(
    store: &semantic_memory::MemoryStore,
) -> Result<Vec<(String, String)>, ErrorData> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let pairs = edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect();
    Ok(pairs)
}

/// Helper: load graph edges for a neighborhood around the given seed node IDs.
/// Uses BFS expansion with max_hops=2 and max_nodes=200 by default.
/// Falls back to full graph load if seeds are empty.
fn load_neighborhood_edge_pairs(
    store: &semantic_memory::MemoryStore,
    seed_ids: &[String],
) -> Result<Vec<(String, String)>, ErrorData> {
    if seed_ids.is_empty() {
        return load_stored_edge_pairs(store);
    }
    let edges = tokio::task::block_in_place(|| {
        Handle::current().block_on(store.list_graph_edges_for_neighborhood(
            seed_ids.to_vec(),
            2,
            200,
        ))
    })
    .map_err(|e| {
        ErrorData::internal_error(format!("Failed to load neighborhood edges: {e}"), None)
    })?;
    let pairs = edges
        .iter()
        .map(|edge| (edge.source.clone(), edge.target.clone()))
        .collect();
    Ok(pairs)
}

/// Helper: load graph edges for a neighborhood as GraphEdgeRef vec.
fn load_neighborhood_edge_refs(
    store: &semantic_memory::MemoryStore,
    seed_ids: &[String],
) -> Result<Vec<semantic_memory::discord::GraphEdgeRef>, ErrorData> {
    if seed_ids.is_empty() {
        return load_stored_edge_refs(store);
    }
    let edges = tokio::task::block_in_place(|| {
        Handle::current().block_on(store.list_graph_edges_for_neighborhood(
            seed_ids.to_vec(),
            2,
            200,
        ))
    })
    .map_err(|e| {
        ErrorData::internal_error(format!("Failed to load neighborhood edges: {e}"), None)
    })?;
    let refs = edges
        .iter()
        .filter_map(|edge| {
            let parsed_type = match edge
                .edge_type_parsed
                .clone()
                .or_else(|| serde_json::from_str(&edge.edge_type).ok())
            {
                Some(t) => t,
                None => {
                    eprintln!("WARNING: Skipping edge with unparseable type between {} and {} (edge_type={})", edge.source, edge.target, edge.edge_type);
                    return None;
                }
            };
            let type_str = match parsed_type {
                semantic_memory::GraphEdgeType::Semantic { .. } => "semantic",
                semantic_memory::GraphEdgeType::Temporal { .. } => "temporal",
                semantic_memory::GraphEdgeType::Causal { .. } => "causal",
                semantic_memory::GraphEdgeType::Entity { .. } => "entity",
            };
            Some(semantic_memory::discord::GraphEdgeRef {
                source: edge.source.clone(),
                target: edge.target.clone(),
                edge_type: type_str.to_string(),
                weight: edge.weight,
            })
        })
        .collect();
    Ok(refs)
}

type FactorEdgeTuple = (
    String,
    String,
    semantic_memory::GraphEdgeType,
    f64,
    Option<String>,
);

/// Helper: load graph edges for a neighborhood as factor graph tuples.
fn load_neighborhood_factor_edges(
    store: &semantic_memory::MemoryStore,
    seed_ids: &[String],
) -> Result<Vec<FactorEdgeTuple>, ErrorData> {
    if seed_ids.is_empty() {
        return load_stored_factor_edges(store);
    }
    let edges = tokio::task::block_in_place(|| {
        Handle::current().block_on(store.list_graph_edges_for_neighborhood(
            seed_ids.to_vec(),
            2,
            200,
        ))
    })
    .map_err(|e| {
        ErrorData::internal_error(format!("Failed to load neighborhood edges: {e}"), None)
    })?;
    let raw = edges
        .iter()
        .map(|edge| {
            let parsed_type = edge
                .edge_type_parsed
                .clone()
                .or_else(|| serde_json::from_str(&edge.edge_type).ok())
                .unwrap_or(semantic_memory::GraphEdgeType::Entity {
                    relation: "unknown".to_string(),
                });
            (
                edge.source.clone(),
                edge.target.clone(),
                parsed_type,
                edge.weight,
                edge.metadata.clone(),
            )
        })
        .collect();
    Ok(raw)
}

/// Load fact ids targeted by entity relation="supersedes" graph edges.
fn load_superseded_targets(
    store: &semantic_memory::MemoryStore,
) -> Result<HashSet<String>, ErrorData> {
    let edges =
        tokio::task::block_in_place(|| Handle::current().block_on(store.list_all_graph_edges()))
            .map_err(|e| {
                ErrorData::internal_error(format!("Failed to load graph edges: {e}"), None)
            })?;
    let mut targets = HashSet::new();
    for edge in edges {
        let parsed_type = edge
            .edge_type_parsed
            .clone()
            .or_else(|| serde_json::from_str(&edge.edge_type).ok());
        if let Some(semantic_memory::GraphEdgeType::Entity { relation }) = parsed_type {
            if relation == "supersedes" {
                targets.insert(edge.target);
            }
        }
    }
    Ok(targets)
}

fn query_allows_superseded(query: &str) -> bool {
    let q = query.to_lowercase();
    q.contains("supersed")
        || q.contains("stale")
        || q.contains("obsolete")
        || q.contains("histor")
        || q.contains("old fact")
        || q.contains("previous fact")
}

/// Serialize a JSON value to a pretty string, mapping serialization errors
/// to protocol-level errors instead of success strings.
/// Build a `ProjectionQuery` from the MCP-facing `ProjectionQueryParams`.
///
/// Maps the flat parameter struct into the library's `ProjectionQuery` with
/// a fully-resolved `ScopeKey` and typed ID filters.
fn build_projection_query(params: ProjectionQueryParams) -> semantic_memory::ProjectionQuery {
    use stack_ids::{ClaimId, ClaimVersionId, EntityId, ScopeKey};

    let scope = ScopeKey {
        namespace: params.namespace,
        domain: params.domain,
        workspace_id: params.workspace_id,
        repo_id: params.repo_id,
    };

    let limit = params.limit.unwrap_or(10) as usize;

    semantic_memory::ProjectionQuery {
        scope,
        text_query: params.text_query,
        valid_at: params.valid_at,
        recorded_at_or_before: params.recorded_at_or_before,
        subject_entity_id: params.subject_entity_id.map(EntityId::new),
        canonical_entity_id: params.canonical_entity_id.map(EntityId::new),
        claim_state: params.claim_state,
        claim_id: params.claim_id.map(ClaimId::new),
        claim_version_id: params.claim_version_id.map(ClaimVersionId::new),
        limit,
    }
}

fn json_to_string(value: &serde_json::Value) -> Result<String, ErrorData> {
    serde_json::to_string_pretty(value)
        .map_err(|e| ErrorData::internal_error(format!("Serialization error: {e}"), None))
}

/// Generate a receipt ID for MCP mutation operations.
/// Format: `mcp-receipt:<tool_name>:<uuid>` — traceable to the tool that produced it.
fn mcp_receipt_id(tool_name: &str) -> String {
    format!("mcp-receipt:{tool_name}:{}", uuid::Uuid::new_v4())
}

/// Current UTC timestamp in ISO 8601 format for receipt recording.
fn mcp_now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Build a receipt envelope for a mutation tool response.
/// Every material MCP mutation must include this in its JSON output.
fn mcp_receipt(tool_name: &str) -> serde_json::Value {
    serde_json::json!({
        "receipt_id": mcp_receipt_id(tool_name),
        "recorded_at": mcp_now_iso(),
        "tool": tool_name,
    })
}

/// Convert only a persisted fact into an autonomous-injection-safe witnessed hit.
///
/// The search index's source is not sufficient provenance: hydrate the fact row
/// so namespace and source are the values actually persisted with the memory.
/// Other result families currently lack this complete provenance surface, so are
/// deliberately omitted instead of receiving invented fields.
fn witnessed_injectible_fact(
    store: &semantic_memory::MemoryStore,
    result: semantic_memory::SearchResult,
    receipt_ref: &str,
) -> Result<Option<serde_json::Value>, ErrorData> {
    let semantic_memory::SearchSource::Fact { fact_id, .. } = &result.source else {
        return Ok(None);
    };
    let fact = tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(fact_id)))
        .map_err(|e| {
            ErrorData::internal_error(
                format!("witnessed fact provenance hydration failed: {e}"),
                None,
            )
        })?;
    let Some(fact) = fact else {
        return Ok(None);
    };
    let Some(source) = fact.source.filter(|source| !source.trim().is_empty()) else {
        return Ok(None);
    };
    let memory_id = format!("fact:{}", fact.id);
    Ok(Some(serde_json::json!({
        "memory_id": memory_id,
        "result_id": result.source.result_id(),
        "content": fact.content,
        "namespace": fact.namespace,
        "source": source,
        "trust": "persisted_unjudged",
        "state": "current",
        "retrieval_receipt_ref": receipt_ref,
        "score": result.score,
        "bm25_rank": result.bm25_rank,
        "vector_rank": result.vector_rank,
        "cosine_similarity": result.cosine_similarity,
    })))
}

#[tool_router]
impl SemanticMemoryServer {
    // ── Core search tools ────────────────────────────────────────────

    #[tool(
        description = "Semantic hybrid search (BM25 + vector + RRF). Returns ranked results with content, scores, and stable result IDs.",
        annotations(read_only_hint = true)
    )]
    fn sm_search(
        &self,
        Parameters(SearchParams {
            query,
            top_k,
            namespaces,
        }): Parameters<SearchParams>,
    ) -> Result<String, ErrorData> {
        let requested_k = top_k.map(|v| v as usize).unwrap_or(5);
        let allow_superseded = false;
        let search_k = if allow_superseded {
            requested_k
        } else {
            (requested_k * 4).max(20)
        };
        let ns: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(search_k), ns.as_deref(), None))
        });

        match result {
            Ok(results) => {
                let superseded_targets = if allow_superseded {
                    HashSet::new()
                } else {
                    load_superseded_targets(store)?
                };
                let fresh_results: Vec<_> = results
                    .iter()
                    .filter(|r| !superseded_targets.contains(&r.source.result_id()))
                    .collect();
                let result_refs: Vec<_> =
                    if superseded_targets.is_empty() {
                        results.iter().collect()
                    } else {
                        fresh_results
                    };
                let superseded_filtered_count = results.len().saturating_sub(result_refs.len());
                let json_results: Vec<serde_json::Value> = result_refs
                    .iter()
                    .take(requested_k)
                    .map(|r| {
                        serde_json::json!({
                            "result_id": r.source.result_id(),
                            "content": r.content,
                            "source": format!("{:?}", r.source),
                            "score": r.score,
                            "bm25_rank": r.bm25_rank,
                            "vector_rank": r.vector_rank,
                            "cosine_similarity": r.cosine_similarity,
                        })
                    })
                    .collect();
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "results": json_results,
                    "count": json_results.len(),
                    "superseded_filtered_count": superseded_filtered_count,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Search error: {e}"),
                None,
            )),
        }
    }

    /// Best-effort claim-ledger trust lookup for a bare fact id. Falls back to
    /// "persisted_unjudged" whenever claim-integration is disabled, no claim
    /// exists for the fact, or no support judgment has been recorded yet.
    #[cfg(feature = "claim-integration")]
    fn trust_for_fact(&self, bare_fact_id: &str) -> String {
        self.claim_trust
            .lock()
            .unwrap()
            .trust_for_fact(bare_fact_id)
    }

    #[cfg(not(feature = "claim-integration"))]
    fn trust_for_fact(&self, _bare_fact_id: &str) -> String {
        "persisted_unjudged".to_string()
    }

    /// Best-effort: link `bare_fact_id` to an existing claim whose normalized
    /// content matches `content`, if one exists and the fact isn't already
    /// linked. Never fails the caller — this is a convenience wiring, not a
    /// truth-store operation.
    #[cfg(feature = "claim-integration")]
    fn auto_link_fact_to_claims(&self, bare_fact_id: &str, content: &str) {
        self.claim_trust
            .lock()
            .unwrap()
            .auto_link_content(bare_fact_id, content);
    }

    #[cfg(not(feature = "claim-integration"))]
    fn auto_link_fact_to_claims(&self, _bare_fact_id: &str, _content: &str) {}

    /// Overwrites the "trust" field of each search result (keyed off its
    /// "memory_id" of the form "fact:<id>") with the claim-ledger support
    /// state, when one has been recorded. Also attempts to auto-link any
    /// still-unjudged result whose content now matches a claim created since
    /// the fact was added. Never fails the search.
    fn enrich_results_with_trust(&self, results: &mut [serde_json::Value]) {
        for result in results.iter_mut() {
            let bare_fact_id = result
                .get("memory_id")
                .and_then(|v| v.as_str())
                .map(|s| s.strip_prefix("fact:").unwrap_or(s).to_string());
            let Some(bare_fact_id) = bare_fact_id else {
                continue;
            };
            if let Some(content) = result.get("content").and_then(|v| v.as_str()) {
                self.auto_link_fact_to_claims(&bare_fact_id, content);
            }
            if let Some(obj) = result.as_object_mut() {
                obj.insert(
                    "trust".to_string(),
                    serde_json::Value::String(self.trust_for_fact(&bare_fact_id)),
                );
            }
        }
    }

    #[tool(
        description = "Mandatory witnessed retrieval. Bypasses cache, verifies durable receipt persistence, defaults to Current state, and supports privacy-preserving opt-in storage for complete replay.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_witnessed(
        &self,
        Parameters(SearchWitnessedParams {
            query,
            top_k,
            namespaces,
            request_id,
            retrieval_mode,
            replay_mode,
        }): Parameters<SearchWitnessedParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::{ExactnessProfile, ReceiptMode, ReplayMode, SearchContext};
        let k = top_k.map(|v| v as usize).unwrap_or(5);
        let request_id = request_id.unwrap_or_else(|| {
            format!(
                "mcp-witness-{}-{}",
                chrono::Utc::now().timestamp_micros(),
                WITNESS_REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            )
        });
        let digest = |s: &str| format!("blake3:{}", blake3::hash(s.as_bytes()).to_hex());
        let filters = serde_json::json!({"namespaces": namespaces});
        let query_digest = digest(&query);
        let input_digest = digest(
            &serde_json::json!({"query": query, "top_k": k, "filters": filters}).to_string(),
        );
        let filter_digest = digest(&filters.to_string());
        let retrieval_mode = retrieval_mode.unwrap_or(RetrievalModeParam::Hybrid);
        let retrieval_mode_name = match retrieval_mode {
            RetrievalModeParam::Hybrid => "hybrid",
            RetrievalModeParam::FtsOnly => "fts_only",
            RetrievalModeParam::VectorOnly => "vector_only",
        };
        let config_digest = digest(&format!(
            "retrieval_mode={retrieval_mode_name};top_k={k};state=current;cache=bypass;exactness=prefer_exact"
        ));
        let ns: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        let mut context = SearchContext::default_now();
        context.receipt_mode = ReceiptMode::ReturnReceipt;
        context.replay_mode = match replay_mode.unwrap_or(ReplayModeParam::NoReplay) {
            ReplayModeParam::NoReplay => ReplayMode::NoReplay,
            ReplayModeParam::StoreInputs => ReplayMode::StoreInputs,
        };
        context.exactness_profile = ExactnessProfile::PreferExact;
        context.request_id = Some(request_id.clone());
        context.query_text_digest = Some(query_digest.clone());
        context.query_input_digest = Some(input_digest.clone());
        context.filter_digest = Some(filter_digest.clone());
        // ReturnReceipt bypasses semantic-memory's cache and propagates persistence failure.
        let response = tokio::task::block_in_place(|| {
            Handle::current().block_on(async {
                match retrieval_mode {
                    RetrievalModeParam::Hybrid => {
                        self.bridge
                            .store
                            .search_with_context(&query, Some(k), ns.as_deref(), None, context)
                            .await
                    }
                    RetrievalModeParam::FtsOnly => {
                        self.bridge
                            .store
                            .search_fts_only_with_context(
                                &query,
                                Some(k),
                                ns.as_deref(),
                                None,
                                context,
                            )
                            .await
                    }
                    RetrievalModeParam::VectorOnly => {
                        self.bridge
                            .store
                            .search_vector_only_with_context(
                                &query,
                                Some(k),
                                ns.as_deref(),
                                None,
                                context,
                            )
                            .await
                    }
                }
            })
        })
        .map_err(|e| {
            ErrorData::internal_error(
                format!("witnessed search/receipt persistence failed: {e}"),
                None,
            )
        })?;
        let receipt = response.receipt.ok_or_else(|| {
            ErrorData::internal_error("witness missing; operation contained".to_string(), None)
        })?;
        let authority_state = tokio::task::block_in_place(|| {
            Handle::current().block_on(self.bridge.store.authority().current_state())
        })
        .map_err(|error| {
            ErrorData::internal_error(format!("authority state lookup failed: {error}"), None)
        })?;
        let durable = tokio::task::block_in_place(|| {
            Handle::current().block_on(self.bridge.store.get_search_receipt(&receipt.receipt_id))
        })
        .map_err(|e| {
            ErrorData::internal_error(format!("receipt verification failed: {e}"), None)
        })?;
        if durable.is_none() {
            return Err(ErrorData::internal_error(
                "receipt not durable; operation contained".to_string(),
                None,
            ));
        }
        let complete_replay_available = tokio::task::block_in_place(|| {
            Handle::current().block_on(
                self.bridge
                    .store
                    .search_replay_inputs_available(&receipt.receipt_id),
            )
        })
        .map_err(|e| {
            ErrorData::internal_error(format!("replay input verification failed: {e}"), None)
        })?;
        let stats =
            tokio::task::block_in_place(|| Handle::current().block_on(self.bridge.store.stats()))
                .map_err(|e| {
                ErrorData::internal_error(format!("model identity unavailable: {e}"), None)
            })?;
        let model_digest = digest(&serde_json::json!({"model": stats.embedding_model, "dimensions": stats.embedding_dimensions}).to_string());
        let receipt_ref = format!("receipt:{}", receipt.receipt_id);
        let mut results = Vec::new();
        for result in response.results {
            if let Some(hit) = witnessed_injectible_fact(&self.bridge.store, result, &receipt_ref)?
            {
                results.push(hit);
            }
        }
        // T2.6: Enrich search results with claim-ledger support state.
        // Best-effort: falls back to "persisted_unjudged" when no claim exists.
        self.enrich_results_with_trust(&mut results);

        // P1.3: Factor graph reranking (opt-in via integration feature).
        // When graph edges exist in the store, build a factor graph with
        // search scores as initial beliefs, run belief propagation, and
        // rerank results by refined beliefs. Items connected by multiple
        // relationship types get compounded confidence.
        #[cfg(feature = "integration")]
        {
            use semantic_memory::factor_graph::{
                factors_from_edges, FactorGraph, FactorGraphConfig,
            };
            let result_nodes: Vec<(String, f64)> = results
                .iter()
                .filter_map(|result| {
                    let id = result
                        .get("memory_id")
                        .and_then(|value| value.as_str())?
                        .to_string();
                    let score = result
                        .get("score")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.5);
                    Some((id, score))
                })
                .collect();
            if !result_nodes.is_empty() {
                let seed_ids: Vec<String> = result_nodes
                    .iter()
                    .map(|(item_id, _)| item_id.clone())
                    .collect();
                let edge_tuples = load_neighborhood_factor_edges(&self.bridge.store, &seed_ids)?;
                if !edge_tuples.is_empty() {
                    let factors = factors_from_edges(&edge_tuples);
                    let factor_graph =
                        FactorGraph::new(&result_nodes, factors, FactorGraphConfig::default());
                    let result_beliefs = factor_graph.propagate();
                    let reranked = result_beliefs.top_k(result_nodes.len());
                    // Reorder results by factor graph beliefs (higher = better).
                    results.sort_by(|a, b| {
                        let a_id = a.get("memory_id").and_then(|v| v.as_str()).unwrap_or("");
                        let b_id = b.get("memory_id").and_then(|v| v.as_str()).unwrap_or("");
                        let a_belief = reranked
                            .iter()
                            .find(|(id, _)| id == a_id)
                            .map(|(_, b)| *b)
                            .unwrap_or(0.0);
                        let b_belief = reranked
                            .iter()
                            .find(|(id, _)| id == b_id)
                            .map(|(_, b)| *b)
                            .unwrap_or(0.0);
                        b_belief
                            .partial_cmp(&a_belief)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        let ordered_results: Vec<_> = results.iter().map(|r| serde_json::json!({"result_id": r["result_id"], "result_digest": digest(&r.to_string())})).collect();
        let exactness = if receipt.approximate {
            "approximate_candidates"
        } else if receipt.exact_rerank {
            "exact_f32_rerank"
        } else {
            "backend_reported_non_approximate"
        };
        json_to_string(&serde_json::json!({
            "schema_version": "retrieval_response_v1", "ok": true, "request_id": request_id, "receipt_id": receipt.receipt_id, "retrieval_mode": retrieval_mode_name,
            "state_view": {"kind": "Current"}, "current_snapshot_id": authority_state.snapshot_id.0,
            "retrieval_epoch": authority_state.retrieval_epoch.0,
            "evaluation_time": receipt.evaluation_time,
            "authority": {
                "snapshot_id": authority_state.snapshot_id.0,
                "retrieval_epoch": authority_state.retrieval_epoch.0,
                "status": "Applied",
                "degradation": null
            },
            "digests": {"query_text": query_digest, "input": input_digest, "filter": filter_digest, "config": config_digest, "model": model_digest},
            "execution": {"cache": "bypassed", "candidate_backend": receipt.candidate_backend, "exactness": exactness, "artifact_generation_id": receipt.artifact_generation_id},
            "ordered_results": ordered_results, "results": results,
            "stage_outcomes": {
                "authority_snapshot": {"outcome": "Applied", "degradation": null},
                "hybrid_retrieval": {"outcome": if matches!(retrieval_mode, RetrievalModeParam::Hybrid) { "Applied" } else { "Skipped" }, "degradation": null},
                "selected_retrieval": {"outcome": "Applied", "degradation": null, "mode": retrieval_mode_name},
                "receipt_persistence": {"outcome": "Applied", "degradation": null},
                "cache": {"outcome": "Skipped", "degradation": "witnessed retrieval bypasses cache"},
                "replay": if complete_replay_available {
                    serde_json::json!({"outcome": "Applied", "degradation": null})
                } else {
                    serde_json::json!({"outcome": "AnalysisOnly", "degradation": "complete replay inputs are not available"})
                }
            },
            "degradations": receipt.degradations,
            "complete_replay_available": complete_replay_available
        }))
    }

    #[tool(
        description = "Search with proof-debt analysis. Runs a standard search, then for each result checks the claim-ledger trust index for support state. Returns results plus a proof_debt summary with total debt weight, unsupported count, and gate decision.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_proof_debt(
        &self,
        Parameters(SearchProofDebtParams {
            query,
            top_k,
            namespaces,
            budget_micros,
        }): Parameters<SearchProofDebtParams>,
    ) -> Result<String, ErrorData> {
        let k = top_k.map(|v| v as usize).unwrap_or(5);
        let store = &self.bridge.store;
        let ns: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());

        let results = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(k), ns.as_deref(), None))
        })
        .map_err(|e| ErrorData::internal_error(format!("search failed: {e}"), None))?;

        // T2.5/D3: Evaluate proof debt for search results using the real
        // claim-ledger budget machinery. Results with no linked claim carry
        // no debt (there is nothing to owe proof for). Results whose claim
        // is unjudged, heuristic-only, unsupported, or contradicted incur
        // the claim-ledger's real ProofDebt weights, and the aggregate is
        // run through the real gate evaluator — no hardcoded weights.
        #[cfg(feature = "claim-integration")]
        {
            use claim_ledger::{
                budget_for_claim, evaluate_proof_debt_gate_with_config, total_proof_debt_weight,
                ProofDebtBudgetConfig,
            };
            let budget = budget_micros.unwrap_or(500_000);
            let idx = self.claim_trust.lock().unwrap();
            let mut all_debts = Vec::new();
            let mut unsupported_count = 0usize;
            let mut per_result = Vec::new();

            for r in &results {
                let fact_id = r.source.result_id();
                let bare_id = fact_id.strip_prefix("fact:").unwrap_or(&fact_id);
                let trust = idx.trust_for_fact(bare_id);
                let (claim_id, debts) = match idx.claim_and_state_for_fact(bare_id) {
                    Some((claim_id, state)) => {
                        let debts = proof_debts_for_support_state(
                            state.unwrap_or(claim_ledger::SupportState::Unknown),
                        );
                        (Some(claim_id), debts)
                    }
                    None => (None, Vec::new()),
                };
                if !debts.is_empty() {
                    unsupported_count += 1;
                }
                let debt_micros = total_proof_debt_weight(&debts);
                all_debts.extend(debts);
                per_result.push(serde_json::json!({
                    "fact_id": fact_id,
                    "claim_id": claim_id,
                    "trust": trust,
                    "debt_micros": debt_micros,
                }));
            }

            let config = ProofDebtBudgetConfig::default();
            let total_debt = total_proof_debt_weight(&all_debts);
            let (proof_budget, debits) =
                budget_for_claim("sm_search_proof_debt", &all_debts, budget);
            let gate = evaluate_proof_debt_gate_with_config(&proof_budget, &config);

            json_to_string(&serde_json::json!({
                "ok": true,
                "query": query,
                "results": per_result,
                "count": results.len(),
                "proof_debt": {
                    "total_debt_micros": total_debt,
                    "unsupported_count": unsupported_count,
                    "budget_micros": budget,
                    "consumed_pct": gate.consumed_pct,
                    "exhausted": gate.exhausted,
                    "gate_decision": gate.decision,
                    "gate_summary": gate.summary,
                    "debit_count": debits.len(),
                },
                "receipt": mcp_receipt("sm_search_proof_debt"),
            }))
        }
        #[cfg(not(feature = "claim-integration"))]
        {
            json_to_string(&serde_json::json!({
                "ok": true,
                "query": query,
                "results": results.iter().map(|r| serde_json::json!({
                    "fact_id": r.source.result_id(),
                    "content": r.content,
                    "trust": "persisted_unjudged",
                })).collect::<Vec<_>>(),
                "count": results.len(),
                "proof_debt": {
                    "total_debt_micros": 0,
                    "unsupported_count": 0,
                    "budget_micros": budget_micros.unwrap_or(500_000),
                    "gate_decision": "Pass",
                    "note": "claim-integration feature not enabled",
                },
                "receipt": mcp_receipt("sm_search_proof_debt"),
            }))
        }
    }

    #[tool(
        description = "Benchmark trust quality across search results. Runs multiple queries and measures the trust distribution (supported/partially_supported/unsupported/contradicted/heuristic_only/persisted_unjudged) of returned results. Shows what fraction of retrieved facts have claim-ledger backing vs unjudged.",
        annotations(read_only_hint = true)
    )]
    fn sm_benchmark_trust(
        &self,
        Parameters(BenchmarkTrustParams {
            query_count,
            top_k,
            namespaces,
        }): Parameters<BenchmarkTrustParams>,
    ) -> Result<String, ErrorData> {
        let n = query_count.map(|v| v as usize).unwrap_or(10);
        let k = top_k.map(|v| v as usize).unwrap_or(5);
        let store = &self.bridge.store;
        let ns: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());

        // Use recent facts as benchmark queries (search for their own content).
        let facts =
            tokio::task::block_in_place(|| Handle::current().block_on(store.list_facts("", n, 0)))
                .map_err(|e| ErrorData::internal_error(format!("list_facts failed: {e}"), None))?;

        #[cfg(feature = "claim-integration")]
        {
            let idx = self.claim_trust.lock().unwrap();
            let mut trust_counts: HashMap<String, usize> = HashMap::new();
            for label in &[
                "supported",
                "partially_supported",
                "unsupported",
                "contradicted",
                "heuristic_only",
                "persisted_unjudged",
            ] {
                trust_counts.insert(label.to_string(), 0);
            }

            let mut total_results = 0usize;
            let mut per_query = Vec::new();

            for fact in &facts {
                let query = &fact.content;
                let results = tokio::task::block_in_place(|| {
                    Handle::current().block_on(store.search(query, Some(k), ns.as_deref(), None))
                })
                .unwrap_or_default();

                let mut query_trust: HashMap<String, usize> = HashMap::new();
                for r in &results {
                    let bare_id = r.source.result_id();
                    let bare = bare_id.strip_prefix("fact:").unwrap_or(&bare_id);
                    let trust = idx.trust_for_fact(bare);
                    *trust_counts.entry(trust.clone()).or_insert(0) += 1;
                    *query_trust.entry(trust.clone()).or_insert(0) += 1;
                    total_results += 1;
                }
                per_query.push(serde_json::json!({
                    "query": query,
                    "result_count": results.len(),
                    "trust_distribution": query_trust,
                }));
            }

            json_to_string(&serde_json::json!({
                "ok": true,
                "queries_run": facts.len(),
                "top_k": k,
                "total_results": total_results,
                "trust_distribution": trust_counts,
                "judged_pct": if total_results > 0 {
                    let judged = trust_counts.get("supported").copied().unwrap_or(0)
                        + trust_counts.get("partially_supported").copied().unwrap_or(0)
                        + trust_counts.get("contradicted").copied().unwrap_or(0)
                        + trust_counts.get("unsupported").copied().unwrap_or(0);
                    (judged as f64 / total_results as f64) * 100.0
                } else { 0.0 },
                "per_query": per_query,
                "receipt": mcp_receipt("sm_benchmark_trust"),
            }))
        }
        #[cfg(not(feature = "claim-integration"))]
        {
            json_to_string(&serde_json::json!({
                "ok": true,
                "queries_run": facts.len(),
                "top_k": k,
                "trust_distribution": {"persisted_unjudged": facts.len()},
                "judged_pct": 0.0,
                "note": "claim-integration feature not enabled",
                "receipt": mcp_receipt("sm_benchmark_trust"),
            }))
        }
    }

    #[tool(
        description = "Return the canonical typed origin-authority decision receipt for asserting a fact. The purpose is fixed to assertion; recall authority is not reused. This read-only decision surface never returns memory content.",
        annotations(read_only_hint = true)
    )]
    fn sm_decide_assertion_authority(
        &self,
        Parameters(params): Parameters<GovernedDecisionParams>,
    ) -> Result<String, ErrorData> {
        self.decide_governed_authority(params, semantic_memory::GovernedAccessPurposeV1::Assertion)
    }

    #[tool(
        description = "Return the canonical typed origin-authority decision receipt for acting on a fact. The purpose is fixed to action; recall or assertion authority is not reused. This read-only decision surface never returns memory content or performs the action.",
        annotations(read_only_hint = true)
    )]
    fn sm_decide_action_authority(
        &self,
        Parameters(params): Parameters<GovernedDecisionParams>,
    ) -> Result<String, ErrorData> {
        self.decide_governed_authority(params, semantic_memory::GovernedAccessPurposeV1::Action)
    }

    // DEPRECATED #[tool(
    // description = "Search with full score breakdown showing how BM25 and vector scores combine. Useful for debugging retrieval quality.",
    // annotations(read_only_hint = true)
    // )]
    #[allow(dead_code)]
    fn sm_search_explained(
        &self,
        Parameters(SearchExplainedParams { query, top_k }): Parameters<SearchExplainedParams>,
    ) -> Result<String, ErrorData> {
        let requested_k = top_k.map(|v| v as usize).unwrap_or(5);
        let allow_superseded = query_allows_superseded(&query);
        let search_k = if allow_superseded {
            requested_k
        } else {
            (requested_k * 4).max(20)
        };
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search_explained(&query, Some(search_k), None, None))
        });

        match result {
            Ok(results) => {
                let superseded_targets = if allow_superseded {
                    HashSet::new()
                } else {
                    load_superseded_targets(store)?
                };
                let fresh_results: Vec<_> = results
                    .iter()
                    .filter(|r| !superseded_targets.contains(&r.result.source.result_id()))
                    .collect();
                let result_refs: Vec<_> =
                    if superseded_targets.is_empty() {
                        results.iter().collect()
                    } else {
                        fresh_results
                    };
                let superseded_filtered_count = results.len().saturating_sub(result_refs.len());
                let json_results: Vec<serde_json::Value> = result_refs
                    .iter()
                    .take(requested_k)
                    .map(|r| {
                        serde_json::json!({
                            "result_id": r.result.source.result_id(),
                            "content": r.result.content,
                            "source": format!("{:?}", r.result.source),
                            "score": r.result.score,
                            "bm25_rank": r.result.bm25_rank,
                            "vector_rank": r.result.vector_rank,
                            "cosine_similarity": r.result.cosine_similarity,
                            "breakdown": {
                                "rrf_score": r.breakdown.rrf_score,
                                "bm25_score": r.breakdown.bm25_score,
                                "vector_score": r.breakdown.vector_score,
                                "recency_score": r.breakdown.recency_score,
                                "bm25_rank": r.breakdown.bm25_rank,
                                "vector_rank": r.breakdown.vector_rank,
                                "vector_source_rank": r.breakdown.vector_source_rank,
                                "vector_source_score": r.breakdown.vector_source_score,
                                "bm25_contribution": r.breakdown.bm25_contribution,
                                "vector_contribution": r.breakdown.vector_contribution,
                                "vector_reranked_from_f32": r.breakdown.vector_reranked_from_f32,
                                "bm25_weight": r.breakdown.bm25_weight,
                                "vector_weight": r.breakdown.vector_weight,
                                "recency_weight": r.breakdown.recency_weight,
                                "rrf_k": r.breakdown.rrf_k,
                            },
                        })
                    })
                    .collect();
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "results": json_results,
                    "count": json_results.len(),
                    "superseded_filtered_count": superseded_filtered_count,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Search error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Add a fact to the knowledge base. Embedded and indexed for semantic search. Returns fact ID and content digest."
    )]
    fn sm_add_fact(
        &self,
        Parameters(AddFactParams {
            content,
            namespace,
            source,
            extract_entities,
            memory_kind,
            sensitivity,
            evidence_refs,
            idempotency_key,
        }): Parameters<AddFactParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;

        // Admission gate: classify sensitivity
        let sens = sensitivity.unwrap_or_else(|| "internal".to_string());
        let kind = memory_kind.unwrap_or_else(|| "durable_fact".to_string());

        // Block confidential/restricted content from autocapture
        if sens == "confidential" || sens == "restricted" {
            return Err(ErrorData::invalid_params(
                format!("Admission gate BLOCKED: sensitivity='{sens}' content cannot be stored without explicit user request"),
                None,
            ));
        }

        // Block ephemeral_inference from becoming durable without evidence
        let explicit_evidence: Vec<String> = evidence_refs
            .as_deref()
            .unwrap_or_default()
            .iter()
            .filter(|reference| !reference.trim().is_empty())
            .cloned()
            .collect();
        if kind == "ephemeral_inference" && explicit_evidence.is_empty() {
            return Err(ErrorData::invalid_params(
                "Admission gate BLOCKED: ephemeral_inference requires evidence_refs to promote to durable".to_string(),
                None,
            ));
        }

        let mut authority_evidence = explicit_evidence;
        if let Some(source_ref) = source.as_ref().filter(|value| !value.trim().is_empty()) {
            if !authority_evidence.contains(source_ref) {
                authority_evidence.push(source_ref.clone());
            }
        }

        // TODO: Pass typed metadata (memory_kind, sensitivity, evidence_refs) through the authority append API once it supports a metadata parameter.

        let caller_idempotency_key = match idempotency_key {
            Some(key) if !key.trim().is_empty() => key,
            Some(_) => {
                return Err(ErrorData::invalid_params(
                    "idempotency_key must not be blank".to_string(),
                    None,
                ))
            }
            None => format!("mcp-sm-add-fact:{}", uuid::Uuid::new_v4()),
        };
        let origin = if authority_evidence.is_empty() {
            semantic_memory::OriginAuthorityLabelV1::operator_system(
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
            )
        } else {
            semantic_memory::OriginAuthorityLabelV1::new(
                semantic_memory::OriginClassV1::ExternalEvidence,
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
                format!(
                    "blake3:{}",
                    blake3::hash(authority_evidence.join("\n").as_bytes()).to_hex()
                ),
                semantic_memory::OriginRiskV1::Medium,
                semantic_memory::AuthorityScopesV1 {
                    recall: semantic_memory::AuthorityScopeV1::Universal,
                    assertion: semantic_memory::AuthorityScopeV1::Denied,
                    action: semantic_memory::AuthorityScopeV1::Denied,
                },
                semantic_memory::ElevationRequirementV1::ExplicitOperatorApproval,
                None,
                semantic_memory::RevocationStatusV1::Active,
                vec!["principal:semantic-memory-mcp".into()],
            )
            .map_err(|error| {
                ErrorData::internal_error(format!("invalid origin label: {error}"), None)
            })?
        };
        let permit = if authority_evidence.is_empty() {
            semantic_memory::AuthorityPermit::operator_system(
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
                semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
            )
        } else {
            semantic_memory::AuthorityPermit::with_evidence(
                "principal:semantic-memory-mcp",
                "caller:sm_add_fact",
                semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
                authority_evidence,
            )
        }
        .with_origin(origin);

        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.authority().append(
                permit,
                caller_idempotency_key,
                namespace.clone(),
                content.clone(),
                source.clone(),
            ))
        });

        match result {
            Ok(receipt) => {
                let id = receipt.affected_ids.first().cloned().ok_or_else(|| {
                    ErrorData::internal_error(
                        "authority append returned no affected fact id".to_string(),
                        None,
                    )
                })?;
                // D4: best-effort auto-link to an existing claim with matching
                // normalized content. Never fails the whole operation.
                self.auto_link_fact_to_claims(&id, &content);
                // Optional entity extraction — best-effort, never fails the whole operation.
                if extract_entities == Some(true) {
                    let prompt = format!(
                        "Extract entities from this text as JSON. Format: {{\"entities\": [{{\"name\": \"...\", \"type\": \"person|project|concept|tool|version|path\"}}]}}\nText: {content}\nJSON:"
                    );
                    let body = serde_json::json!({
                        "model": "granite4.1:3b",
                        "prompt": prompt,
                        "stream": false,
                        "options": {"temperature": 0, "num_predict": 200}
                    });
                    if let Ok(resp) = reqwest::blocking::Client::new()
                        .post("http://127.0.0.1:11434/api/generate")
                        .json(&body)
                        .send()
                    {
                        if let Ok(v) = resp.json::<serde_json::Value>() {
                            if let Some(response_str) = v.get("response").and_then(|r| r.as_str()) {
                                // Use boundary compiler for robust JSON parsing with duplicate-key rejection
                                let parsed_result =
                                    boundary_compiler::parse_with_dup_check(response_str.trim());
                                if let Ok(parsed) = parsed_result {
                                    if let Some(entities) =
                                        parsed.get("entities").and_then(|e| e.as_array())
                                    {
                                        let fact_node = format!("fact:{id}");
                                        for entity in entities {
                                            if let Some(name) =
                                                entity.get("name").and_then(|n| n.as_str())
                                            {
                                                let entity_node = format!("entity:{name}");
                                                let _ = tokio::task::block_in_place(|| {
                                                    Handle::current()
                                                        .block_on(store.add_graph_edge(
                                                        &fact_node,
                                                        &entity_node,
                                                        semantic_memory::GraphEdgeType::Entity {
                                                            relation: "mentions".to_string(),
                                                        },
                                                        1.0,
                                                        None,
                                                    ))
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                json_to_string(&serde_json::json!({
                    "ok": true,
                    "fact_id": id,
                    "namespace": namespace,
                    "receipt": mcp_receipt("sm_add_fact"),
                    "message": "Fact added successfully",
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Error adding fact: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Ingest a document with automatic chunking. Splits into chunks, each embedded and indexed. Returns document ID and chunk count.",
        annotations(idempotent_hint = true)
    )]
    fn sm_ingest_document(
        &self,
        Parameters(IngestDocumentParams {
            content,
            title,
            namespace,
        }): Parameters<IngestDocumentParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current()
                .block_on(store.ingest_document(&title, &content, &namespace, None, None))
        });

        match result {
            Ok(doc_id) => {
                let chunk_count = tokio::task::block_in_place(|| {
                    Handle::current().block_on(store.count_chunks_for_document(&doc_id))
                })
                .unwrap_or(0);
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "receipt": mcp_receipt("sm_ingest_document"),
                    "document_id": doc_id,
                    "title": title,
                    "chunk_count": chunk_count,
                    "message": "Document ingested successfully",
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Error ingesting document: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Get knowledge base statistics: fact/chunk/document/session counts, DB size, embedding model, and graph edge count.",
        annotations(read_only_hint = true)
    )]
    fn sm_stats(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let core = tokio::task::block_in_place(|| Handle::current().block_on(store.stats()));
        let graph = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_all_graph_edges())
        });
        let core_health = match &core {
            Ok(_) => serde_json::json!({"health": "healthy", "error": null}),
            Err(e) => serde_json::json!({"health": "error", "error": e.to_string()}),
        };
        let graph_health = match &graph {
            Ok(_) => serde_json::json!({"health": "healthy", "error": null}),
            Err(e) => serde_json::json!({"health": "error", "error": e.to_string()}),
        };
        let core_value = core.ok();
        let graph_count = graph.ok().map(|edges| edges.len());
        json_to_string(&serde_json::json!({
            "ok": core_value.is_some() && graph_count.is_some(),
            "components": {"core": core_health, "graph": graph_health},
            "facts": core_value.as_ref().map(|s| s.total_facts),
            "chunks": core_value.as_ref().map(|s| s.total_chunks),
            "documents": core_value.as_ref().map(|s| s.total_documents),
            "sessions": core_value.as_ref().map(|s| s.total_sessions),
            "messages": core_value.as_ref().map(|s| s.total_messages),
            "graph_edges": graph_count,
            "db_size_bytes": core_value.as_ref().map(|s| s.database_size_bytes),
            "db_size_mb": core_value.as_ref().map(|s| (s.database_size_bytes as f64 / 1_048_576.0 * 100.0).round() / 100.0),
            "embedding_model": core_value.as_ref().and_then(|s| s.embedding_model.clone()),
            "embedding_dimensions": core_value.as_ref().and_then(|s| s.embedding_dimensions),
        }))
    }

    #[tool(
        description = "Find shortest path between two items in the knowledge graph. Traverses all edge types. Returns node IDs with edge evidence per hop.",
        annotations(read_only_hint = true)
    )]
    fn sm_graph_path(
        &self,
        Parameters(GraphPathParams {
            from_id,
            to_id,
            max_depth,
        }): Parameters<GraphPathParams>,
    ) -> Result<String, ErrorData> {
        let depth = max_depth.map(|v| v as usize).unwrap_or(5);
        let store = &self.bridge.store;
        let g = store.graph_view();

        match typed_graph_path(g.as_ref(), &from_id, &to_id, depth) {
            Ok(GraphPathOutcome::Found(path)) => {
                // Build edge evidence for each hop by examining neighbors.
                let path_segments = build_path_segments(store, &path);
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "outcome": "Found",
                    "from": from_id,
                    "to": to_id,
                    "path": path,
                    "path_length": path.len(),
                    "segments": path_segments,
                }))
            }
            Ok(GraphPathOutcome::NoPathWithinCompleteSearch) => {
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "outcome": "NoPathWithinCompleteSearch",
                    "from": from_id,
                    "to": to_id,
                    "path": null,
                    "message": format!("No path found from {from_id} to {to_id} within depth {depth}"),
                }))
            }
            Ok(GraphPathOutcome::BudgetExceeded) => json_to_string(&serde_json::json!({
                "ok": false, "outcome": "BudgetExceeded", "from": from_id, "to": to_id,
                "path": null, "budget": {"max_depth": depth}
            })),
            Ok(GraphPathOutcome::InvalidEndpoint(endpoint)) => json_to_string(&serde_json::json!({
                "ok": false, "outcome": "InvalidEndpoint", "invalid_endpoint": endpoint,
                "from": from_id, "to": to_id, "path": null
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Graph view error: {e}"),
                None,
            )),
        }
    }

    // ── Direct read and supersession tools (v0.3.1) ──────────────────

    #[tool(
        description = "Fetch one fact by id (bare UUID or prefixed 'fact:<uuid>'). Returns full content, namespace, source, timestamps, and metadata.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_fact(
        &self,
        Parameters(GetFactParams { fact_id }): Parameters<GetFactParams>,
    ) -> Result<String, ErrorData> {
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&bare)));
        match result {
            Ok(Some(f)) => json_to_string(&serde_json::json!({
                "ok": true,
                "found": true,
                "fact": {
                    "result_id": format!("fact:{}", f.id),
                    "id": f.id,
                    "namespace": f.namespace,
                    "content": f.content,
                    "source": f.source,
                    "created_at": f.created_at,
                    "updated_at": f.updated_at,
                    "metadata": f.metadata,
                },
            })),
            Ok(None) => json_to_string(&serde_json::json!({
                "ok": true,
                "found": false,
                "message": format!("No fact with id '{fact_id}'"),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("get_fact error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Enumerate facts in a namespace (newest first) with pagination. Exhaustive, not similarity-ranked — for browsing, auditing, or deduping.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_facts(
        &self,
        Parameters(ListFactsParams {
            namespace,
            limit,
            offset,
        }): Parameters<ListFactsParams>,
    ) -> Result<String, ErrorData> {
        let lim = limit.map(|v| v as usize).unwrap_or(50);
        let off = offset.map(|v| v as usize).unwrap_or(0);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_facts(&namespace, lim, off))
        });
        match result {
            Ok(facts) => {
                let arr: Vec<serde_json::Value> = facts
                    .iter()
                    .map(|f| {
                        serde_json::json!({
                            "result_id": format!("fact:{}", f.id),
                            "id": f.id,
                            "namespace": f.namespace,
                            "content": f.content,
                            "source": f.source,
                            "updated_at": f.updated_at,
                        })
                    })
                    .collect();
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "namespace": namespace,
                    "count": arr.len(),
                    "limit": lim,
                    "offset": off,
                    "facts": arr,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("list_facts error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "List namespaces that currently contain facts. Use before sm_list_facts to discover what is stored.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_namespaces(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_fact_namespaces())
        });
        match result {
            Ok(ns) => json_to_string(&serde_json::json!({
                "ok": true,
                "count": ns.len(),
                "namespaces": ns,
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("list_namespaces error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Fetch a fact plus its graph neighbors WITH their content in one call. Hydrates neighbor facts for ids returned by graph tools.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_fact_neighbors(
        &self,
        Parameters(GetFactNeighborsParams { item_id }): Parameters<GetFactNeighborsParams>,
    ) -> Result<String, ErrorData> {
        let node_id = if item_id.contains(':') {
            item_id.clone()
        } else {
            format!("fact:{item_id}")
        };
        let bare = node_id
            .strip_prefix("fact:")
            .unwrap_or(&node_id)
            .to_string();
        let store = &self.bridge.store;

        let center =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&bare)))
                .map_err(|e| ErrorData::internal_error(format!("get_fact error: {e}"), None))?;
        let edges = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_graph_edges_for_node(&node_id))
        })
        .map_err(|e| ErrorData::internal_error(format!("list edges error: {e}"), None))?;

        let mut neighbors: Vec<serde_json::Value> = Vec::new();
        for e in &edges {
            let outgoing = e.source == node_id;
            let other = if outgoing { &e.target } else { &e.source };
            let other_bare = other.strip_prefix("fact:").unwrap_or(other).to_string();
            let content = tokio::task::block_in_place(|| {
                Handle::current().block_on(store.get_fact(&other_bare))
            })
            .ok()
            .flatten()
            .map(|f| f.content);
            neighbors.push(serde_json::json!({
                "neighbor_id": other,
                "direction": if outgoing { "out" } else { "in" },
                "edge_type": e.edge_type,
                "weight": e.weight,
                "content": content,
            }));
        }
        json_to_string(&serde_json::json!({
            "ok": true,
            "item_id": node_id,
            "center_content": center.map(|f| f.content),
            "neighbor_count": neighbors.len(),
            "neighbors": neighbors,
        }))
    }

    #[tool(
        description = "Create a replacement fact and link it to a stale fact via 'supersedes' edge. Use instead of deleting outdated facts. Returns new fact id and edge id.",
        annotations(idempotent_hint = true)
    )]
    fn sm_supersede_fact(
        &self,
        Parameters(SupersedeFactParams {
            old_fact_id,
            content,
            namespace,
            source,
            reason,
        }): Parameters<SupersedeFactParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::GraphEdgeType;

        let old_bare = old_fact_id
            .strip_prefix("fact:")
            .unwrap_or(&old_fact_id)
            .to_string();
        let old_node = format!("fact:{old_bare}");
        let store = &self.bridge.store;
        let old =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&old_bare)))
                .map_err(|e| ErrorData::internal_error(format!("get old fact error: {e}"), None))?;
        let Some(old_fact) = old else {
            return Err(ErrorData::invalid_params(
                format!("No fact with id '{old_fact_id}'"),
                None,
            ));
        };

        let ns = namespace.unwrap_or_else(|| old_fact.namespace.clone());
        let new_id = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_fact(&ns, &content, source.as_deref(), None))
        })
        .map_err(|e| ErrorData::internal_error(format!("add replacement fact error: {e}"), None))?;
        let new_node = format!("fact:{new_id}");
        let metadata = serde_json::json!({
            "reason": reason.unwrap_or_else(|| "replacement fact supersedes stale fact".to_string()),
            "old_fact_id": old_bare,
        });
        let edge = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_graph_edge(
                &new_node,
                &old_node,
                GraphEdgeType::Entity {
                    relation: "supersedes".to_string(),
                },
                1.0,
                Some(metadata),
            ))
        })
        .map_err(|e| ErrorData::internal_error(format!("add supersedes edge error: {e}"), None))?;

        json_to_string(&serde_json::json!({
            "ok": true,
            "receipt": mcp_receipt("sm_supersede_fact"),
            "new_fact_id": new_id,
            "new_result_id": new_node,
            "old_fact_id": old_bare,
            "old_result_id": old_node,
            "namespace": ns,
            "edge_id": edge.id,
            "relation": "supersedes",
        }))
    }

    // ── Conversation / session tools (v0.3.0) ────────────────────────

    // DEPRECATED #[tool(
    // description = "Create a conversation session (container for messages). Returns session id. Use to persist history recallable via sm_search_conversations.",
    // annotations(idempotent_hint = true)
    // )]
    #[allow(dead_code)]
    fn sm_create_session(
        &self,
        Parameters(CreateSessionParams { channel, metadata }): Parameters<CreateSessionParams>,
    ) -> Result<String, ErrorData> {
        let meta: Option<serde_json::Value> = metadata
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok());
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.create_session_with_metadata(&channel, meta))
        });
        match result {
            Ok(id) => json_to_string(
                &serde_json::json!({"ok": true, "session_id": id, "channel": channel, "receipt": mcp_receipt("sm_create_session")}),
            ),
            Err(e) => Err(ErrorData::internal_error(
                format!("create_session error: {e}"),
                None,
            )),
        }
    }

    // DEPRECATED #[tool(
    // description = "Append a message to a session. role: user|assistant|system|tool. Message is embedded and FTS-indexed. Returns message id."
    // )]
    #[allow(dead_code)]
    fn sm_add_message(
        &self,
        Parameters(AddMessageParams {
            session_id,
            role,
            content,
        }): Parameters<AddMessageParams>,
    ) -> Result<String, ErrorData> {
        let parsed_role = match role.to_lowercase().as_str() {
            "user" => semantic_memory::types::Role::User,
            "assistant" => semantic_memory::types::Role::Assistant,
            "system" => semantic_memory::types::Role::System,
            "tool" => semantic_memory::types::Role::Tool,
            other => {
                return Err(ErrorData::invalid_params(
                    format!("invalid role '{other}' (use user|assistant|system|tool)"),
                    None,
                ))
            }
        };
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_message_embedded(
                &session_id,
                parsed_role,
                &content,
                None,
                None,
            ))
        });
        match result {
            Ok(id) => json_to_string(
                &serde_json::json!({"ok": true, "message_id": id, "session_id": session_id, "receipt": mcp_receipt("sm_add_message")}),
            ),
            Err(e) => Err(ErrorData::internal_error(
                format!("add_message error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "List recent conversation sessions (newest first) with message counts.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_sessions(
        &self,
        Parameters(ListSessionsParams { limit, offset }): Parameters<ListSessionsParams>,
    ) -> Result<String, ErrorData> {
        let lim = limit.map(|v| v as usize).unwrap_or(20);
        let off = offset.map(|v| v as usize).unwrap_or(0);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_sessions(lim, off))
        });
        match result {
            Ok(sessions) => json_to_string(&serde_json::json!({
                "ok": true,
                "count": sessions.len(),
                "sessions": sessions.iter().map(|s| serde_json::json!({
                    "session_id": s.id,
                    "channel": s.channel,
                    "message_count": s.message_count,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("list_sessions error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Get most recent messages from a session within a token budget (default 4000), chronological order. Returns role, content, timestamps.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_messages(
        &self,
        Parameters(GetMessagesParams {
            session_id,
            max_tokens,
        }): Parameters<GetMessagesParams>,
    ) -> Result<String, ErrorData> {
        let budget = max_tokens.unwrap_or(4000);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.get_messages_within_budget(&session_id, budget))
        });
        match result {
            Ok(msgs) => json_to_string(&serde_json::json!({
                "ok": true,
                "session_id": session_id,
                "count": msgs.len(),
                "messages": msgs.iter().map(|m| serde_json::json!({
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "token_count": m.token_count,
                    "created_at": m.created_at,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("get_messages error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Hybrid semantic search over stored conversation MESSAGES (not facts). Recall what was discussed in past sessions. Returns ranked messages.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_conversations(
        &self,
        Parameters(SearchConversationsParams { query, top_k }): Parameters<
            SearchConversationsParams,
        >,
    ) -> Result<String, ErrorData> {
        let k = top_k.map(|v| v as usize);
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search_conversations(&query, k, None))
        });
        match result {
            Ok(results) => json_to_string(&serde_json::json!({
                "ok": true,
                "count": results.len(),
                "results": results.iter().map(|r| serde_json::json!({
                    "result_id": r.source.result_id(),
                    "content": r.content,
                    "score": r.score,
                    "cosine_similarity": r.cosine_similarity,
                })).collect::<Vec<_>>(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("search_conversations error: {e}"),
                None,
            )),
        }
    }

    // ── Feature-gated tools ──────────────────────────────────────────
    // Note: cfg gates are removed from individual tool methods because
    // rmcp's #[tool_router] macro needs all tools visible at expansion
    // time. The `full` feature in Cargo.toml already enables the
    // semantic-memory sub-features these tools depend on.

    #[tool(
        description = "Profile a query and get an adaptive routing decision. Determines which retrieval stages (BM25, vector, rerank, graph, decoder, discord) to activate.",
        annotations(read_only_hint = true)
    )]
    fn sm_route_query(
        &self,
        Parameters(RouteQueryParams { query }): Parameters<RouteQueryParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::rl_routing::{is_trained, route_with_policy};
        use semantic_memory::routing::{QueryProfile, RetrievalRouter};

        let router = RetrievalRouter {
            decoder_enabled: true,
            discord_enabled: true,
            corpus_density: 0.5,
            ..Default::default()
        };

        let profile = QueryProfile::from_query(&query);
        let policy = tokio::task::block_in_place(|| {
            Handle::current().block_on(self.bridge.store.load_routing_policy())
        })
        .map_err(|e| ErrorData::internal_error(format!("load routing policy error: {e}"), None))?;
        let (decision, routing_source) = match policy.as_ref().filter(|p| is_trained(p)) {
            Some(policy) => (route_with_policy(policy, &profile), "trained_policy"),
            None => (router.route(&profile), "heuristic"),
        };
        json_to_string(&serde_json::json!({
            "ok": true,
            "routing_source": routing_source,
            "bm25_coarse": decision.bm25_coarse,
            "vector_medium": decision.vector_medium,
            "rerank_fine": decision.rerank_fine,
            "graph_expansion": decision.graph_expansion,
            "decoder": decision.decoder,
            "discord": decision.discord,
            "no_retrieval": decision.no_retrieval,
            "reasoning": decision.reasoning,
        }))
    }

    #[tool(
        description = "Adaptive search: profiles query, routes to appropriate stages, applies factor graph belief propagation if decoder is activated. Returns results with stable IDs.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_with_routing(
        &self,
        Parameters(SearchWithRoutingParams {
            query,
            top_k,
            contradictions,
            group_by_community,
        }): Parameters<SearchWithRoutingParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::integration::plan_execution;
        use semantic_memory::rl_routing::{is_trained, route_with_policy};
        use semantic_memory::routing::{QueryProfile, RetrievalRouter};

        let k = top_k.map(|v| v as usize).unwrap_or(5);
        let allow_superseded = query_allows_superseded(&query);
        let search_k = if allow_superseded { k } else { (k * 4).max(20) };

        let router = RetrievalRouter {
            decoder_enabled: true,
            discord_enabled: true,
            corpus_density: 0.5,
            ..Default::default()
        };

        // Select learned routing only after enough examples have been durably
        // persisted. A missing or still-untrained policy uses heuristics.
        let store = &self.bridge.store;
        let policy =
            tokio::task::block_in_place(|| Handle::current().block_on(store.load_routing_policy()))
                .map_err(|e| {
                    ErrorData::internal_error(format!("load routing policy error: {e}"), None)
                })?;
        let profile = QueryProfile::from_query(&query);
        let (decision, routing_source) = match policy.as_ref().filter(|p| is_trained(p)) {
            Some(policy) => (route_with_policy(policy, &profile), "trained_policy"),
            None => (router.route(&profile), "heuristic"),
        };
        let contras = contradictions.unwrap_or_default();
        let plan = plan_execution(&decision, contras.clone());

        let store = &self.bridge.store;
        let search_result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(search_k), None, None))
        });

        match search_result {
            Ok(results) => {
                let superseded_targets = if allow_superseded {
                    HashSet::new()
                } else {
                    load_superseded_targets(store)?
                };
                let fresh_results: Vec<_> = results
                    .iter()
                    .filter(|r| !superseded_targets.contains(&r.source.result_id()))
                    .collect();
                let result_refs: Vec<_> =
                    if superseded_targets.is_empty() {
                        results.iter().collect()
                    } else {
                        fresh_results
                    };
                let superseded_filtered_count = results.len().saturating_sub(result_refs.len());
                let json_results: Vec<serde_json::Value> = result_refs
                    .iter()
                    .take(k)
                    .map(|r| {
                        serde_json::json!({
                            "result_id": r.source.result_id(),
                            "content": r.content,
                            "score": r.score,
                        })
                    })
                    .collect();

                let mut factor_graph_payload = serde_json::json!({
                    "enabled": false,
                });

                let mut decoder_executed = false;
                let mut discord_executed = false;
                let mut discord_results_payload: Vec<serde_json::Value> = Vec::new();

                if decision.decoder {
                    #[cfg(feature = "full")]
                    {
                        use semantic_memory::factor_graph::{
                            factors_from_edges, FactorGraph, FactorGraphConfig,
                        };

                        let graph_edges = tokio::task::block_in_place(|| {
                            Handle::current().block_on(store.list_all_graph_edges())
                        });

                        match graph_edges {
                            Ok(edges) => {
                                let raw_edges: Vec<(
                                    String,
                                    String,
                                    semantic_memory::GraphEdgeType,
                                    f64,
                                    Option<String>,
                                )> = edges
                                    .iter()
                                    .map(|edge| {
                                        let parsed_type = edge
                                            .edge_type_parsed
                                            .clone()
                                            .or_else(|| serde_json::from_str(&edge.edge_type).ok())
                                            .unwrap_or(semantic_memory::GraphEdgeType::Entity {
                                                relation: "unknown".to_string(),
                                            });
                                        (
                                            edge.source.clone(),
                                            edge.target.clone(),
                                            parsed_type,
                                            edge.weight,
                                            edge.metadata.clone(),
                                        )
                                    })
                                    .collect();

                                let nodes: Vec<(String, f64)> = result_refs
                                    .iter()
                                    .map(|r| (r.source.result_id(), r.score))
                                    .collect();
                                let factors = factors_from_edges(&raw_edges);
                                let graph =
                                    FactorGraph::new(&nodes, factors, FactorGraphConfig::default());
                                let propagated = graph.propagate();
                                let top_beliefs = propagated.top_k(k);

                                factor_graph_payload = serde_json::json!({
                                    "enabled": true,
                                    "top_k_beliefs": top_beliefs
                                        .into_iter()
                                        .map(|(item_id, belief)| serde_json::json!({
                                            "item_id": item_id,
                                            "belief": belief,
                                        }))
                                        .collect::<Vec<_>>(),
                                    "iterations": propagated.iterations,
                                    "converged": propagated.converged,
                                    "elapsed_ms": propagated.elapsed_ms,
                                    "factor_counts": {
                                        "semantic": propagated.factor_counts.semantic,
                                        "temporal": propagated.factor_counts.temporal,
                                        "causal": propagated.factor_counts.causal,
                                        "entity": propagated.factor_counts.entity,
                                        "total": propagated.factor_counts.total(),
                                    },
                                });
                                decoder_executed = true;
                            }
                            Err(e) => {
                                factor_graph_payload = serde_json::json!({
                                    "enabled": false,
                                    "error": format!("factor graph analysis failed: {e}"),
                                });
                            }
                        }
                    }

                    #[cfg(not(feature = "full"))]
                    {
                        factor_graph_payload = serde_json::json!({
                            "enabled": false,
                            "reason": "factor graph analysis requires the `full` feature",
                        });
                    }

                    if !plan.contradictions.is_empty() {
                        use semantic_memory::decoder::{compute_correction, detect_syndromes};
                        let result_scores: Vec<(String, f64)> = result_refs
                            .iter()
                            .map(|r| (r.source.result_id(), r.score))
                            .collect();
                        let syndromes = detect_syndromes(&result_scores, &plan.contradictions);
                        let _ = compute_correction(&syndromes, 10.0);
                        decoder_executed = true;
                    }
                }

                if plan.use_discord {
                    use semantic_memory::discord::DiscordScorer;
                    let direct_ids: Vec<String> =
                        result_refs.iter().map(|r| r.source.result_id()).collect();
                    let existing_ids: std::collections::HashSet<String> =
                        direct_ids.iter().cloned().collect();
                    if let Ok(edges) = load_neighborhood_edge_refs(&self.bridge.store, &direct_ids)
                    {
                        let scorer = DiscordScorer::with_defaults();
                        let discord_hits = scorer.score(&direct_ids, &edges);
                        for hit in &discord_hits {
                            if !existing_ids.contains(&hit.item_id) {
                                discord_results_payload.push(serde_json::json!({
                                    "result_id": hit.item_id,
                                    "discord_score": hit.discord_score,
                                    "anchor_ids": hit.anchor_ids,
                                    "relationship_types": hit.relationship_types,
                                }));
                            }
                        }
                        discord_executed = true;
                    }
                }

                let mut matryoshka_payload = serde_json::json!({
                    "enabled": false,
                });
                if decision.vector_medium {
                    #[cfg(feature = "full")]
                    {
                        use semantic_memory::integration::multi_resolution_route;
                        use semantic_memory::matryoshka::MatryoshkaConfig;
                        use semantic_memory::routing::QueryProfile;

                        let route_profile = QueryProfile::from_query(&query);
                        let route_decision =
                            multi_resolution_route(&route_profile, &MatryoshkaConfig::default());
                        matryoshka_payload = serde_json::json!({
                            "enabled": true,
                            "candidate_dim": route_decision.candidate_dim,
                            "heuristic_recall_estimate": route_decision.estimated_recall,
                            "recall_basis": "heuristic_dimensional_model_not_corpus_measured",
                            "embedding_dim": route_decision.embedding_dim,
                            "reasoning": route_decision.reasoning,
                        });
                    }

                    #[cfg(not(feature = "full"))]
                    {
                        matryoshka_payload = serde_json::json!({
                            "enabled": false,
                            "reason": "matryoshka routing requires the `full` feature",
                        });
                    }
                }

                // Community grouping (opt-in).
                let grouped_results_payload: serde_json::Value = if group_by_community == Some(true)
                {
                    let seed_ids: Vec<String> = result_refs
                        .iter()
                        .take(k)
                        .map(|r| r.source.result_id())
                        .collect();
                    let edges = load_neighborhood_edge_pairs(store, &seed_ids).unwrap_or_default();
                    if !edges.is_empty() {
                        use semantic_memory::community::detect_communities;
                        let communities = detect_communities(&edges, 1.0, 42);
                        let mut member_to_comm: std::collections::HashMap<String, String> =
                            std::collections::HashMap::new();
                        for c in &communities {
                            for m in &c.members {
                                member_to_comm.insert(m.clone(), c.id.clone());
                            }
                        }
                        let mut groups: std::collections::HashMap<String, Vec<serde_json::Value>> =
                            std::collections::HashMap::new();
                        let mut ungrouped: Vec<serde_json::Value> = Vec::new();
                        for r in &json_results {
                            if let Some(rid) = r.get("result_id").and_then(|v| v.as_str()) {
                                match member_to_comm.get(rid).cloned() {
                                    Some(cid) => groups.entry(cid).or_default().push(r.clone()),
                                    None => ungrouped.push(r.clone()),
                                }
                            }
                        }
                        let mut map = serde_json::Map::new();
                        for (cid, items) in groups {
                            map.insert(format!("community_{cid}"), serde_json::json!(items));
                        }
                        if !ungrouped.is_empty() {
                            map.insert("ungrouped".to_string(), serde_json::json!(ungrouped));
                        }
                        serde_json::Value::Object(map)
                    } else {
                        serde_json::Value::Null
                    }
                } else {
                    serde_json::Value::Null
                };

                // Task 7: Auto-call topology when routing returns Class D (SYNTHESIS) and >10 results.
                let mut topology_payload = serde_json::json!({ "auto_called": false });
                {
                    use semantic_memory::routing::{QueryComplexityClass, QueryProfile};
                    let route_profile = QueryProfile::from_query(&query);
                    if route_profile.complexity_class == QueryComplexityClass::Synthesis
                        && result_refs.len() > 10
                    {
                        #[cfg(feature = "full")]
                        {
                            use semantic_memory::topology::{compute_betti_numbers, find_voids};
                            let edges = load_stored_edge_pairs(store).unwrap_or_default();
                            if !edges.is_empty() {
                                let mut adjacency: std::collections::HashMap<String, Vec<String>> =
                                    std::collections::HashMap::new();
                                for (src, tgt) in &edges {
                                    adjacency.entry(src.clone()).or_default().push(tgt.clone());
                                    adjacency.entry(tgt.clone()).or_default().push(src.clone());
                                }
                                let betti = compute_betti_numbers(&adjacency);
                                let voids = find_voids(&edges);
                                topology_payload = serde_json::json!({
                                    "auto_called": true,
                                    "trigger": "synthesis_class_with_10_plus_results",
                                    "betti_numbers": {
                                        "betti_0": betti.betti_0,
                                        "betti_1": betti.betti_1,
                                    },
                                    "void_count": voids.len(),
                                    "voids": voids.iter().map(|v| serde_json::json!({
                                        "description": v.description,
                                        "void_type": format!("{:?}", v.void_type),
                                        "nearby_items": v.nearby_items,
                                        "suggested_connections": v.suggested_connections,
                                    })).collect::<Vec<_>>(),
                                });
                            } else {
                                topology_payload = serde_json::json!({
                                    "auto_called": true,
                                    "trigger": "synthesis_class_with_10_plus_results",
                                    "note": "no graph edges in store",
                                });
                            }
                        }
                        #[cfg(not(feature = "full"))]
                        {
                            topology_payload = serde_json::json!({
                                "auto_called": true,
                                "trigger": "synthesis_class_with_10_plus_results",
                                "error": "topology requires the full feature",
                            });
                        }
                    }
                }

                json_to_string(&serde_json::json!({
                    "ok": true,
                    "routing_decision": {
                        "source": routing_source,
                        "bm25_coarse": decision.bm25_coarse,
                        "vector_medium": decision.vector_medium,
                        "rerank_fine": decision.rerank_fine,
                        "graph_expansion": decision.graph_expansion,
                        "decoder": decision.decoder,
                        "discord": decision.discord,
                        "no_retrieval": decision.no_retrieval,
                        "reasoning": decision.reasoning,
                    },
                    "results": json_results,
                    "count": json_results.len(),
                    "superseded_filtered_count": superseded_filtered_count,
                    "decoder_planned": plan.use_decoder,
                    "decoder_executed": decoder_executed,
                    "discord_planned": plan.use_discord,
                    "discord_executed": discord_executed,
                    "discord_results": discord_results_payload,
                    "factor_graph": factor_graph_payload,
                    "matryoshka": matryoshka_payload,
                    "grouped_results": grouped_results_payload,
                    "topology": topology_payload,
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("Search error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Detect contradictions in search results. Runs syndrome detection, computes corrections, and applies belief propagation to refine confidence scores.",
        annotations(read_only_hint = true)
    )]
    fn sm_decoder_analyze(
        &self,
        Parameters(DecoderAnalyzeParams {
            results,
            contradictions,
        }): Parameters<DecoderAnalyzeParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::decoder::{
            compute_correction, detect_syndromes, pass_messages, ConflictGraph,
        };

        let contras = contradictions.unwrap_or_default();
        let syndromes = detect_syndromes(&results, &contras);
        let corrections = compute_correction(&syndromes, 10.0);
        let graph = ConflictGraph::from_syndromes(&results, &syndromes);
        let mp = pass_messages(&graph, 50, 0.001);

        json_to_string(&serde_json::json!({
            "ok": true,
            "syndromes": syndromes.iter().map(|s| serde_json::json!({
                "id": s.id,
                "severity": format!("{:?}", s.severity),
                "items": s.items,
                "description": s.description,
                "type": format!("{:?}", s.syndrome_type),
            })).collect::<Vec<_>>(),
            "syndrome_count": syndromes.len(),
            "corrections": corrections.iter().map(|c| serde_json::json!({
                "id": c.id,
                "confidence": c.confidence,
                "cost": c.cost,
                "operations": c.operations.len(),
            })).collect::<Vec<_>>(),
            "correction_count": corrections.len(),
            "message_passing": {
                "iterations": mp.iterations,
                "converged": mp.converged,
                "elapsed_ms": mp.elapsed_ms,
            },
        }))
    }

    #[tool(
        description = "Detect contradictions among the top results for a query from their CONTENT (numeric, value, negation, or antonym disagreement) — no pre-asserted edges required. Returns candidate conflicting pairs, each with the signals that fired and a human-readable reason. Persist a confirmed pair with sm_add_graph_edge(edge_type=\"contradicts\") so the decoder/community/factor-graph tools pick it up.",
        annotations(read_only_hint = true)
    )]
    fn sm_detect_contradictions(
        &self,
        Parameters(DetectContradictionsParams {
            query,
            top_k,
            record_to_ledger,
        }): Parameters<DetectContradictionsParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::contradiction_detect::{detect_contradictions, DetectorConfig};

        let k = top_k.map(|v| v as usize).unwrap_or(10);
        let store = &self.bridge.store;
        let results = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search(&query, Some(k), None, None))
        })
        .map_err(|e| ErrorData::internal_error(format!("search failed: {e}"), None))?;

        let items: Vec<(String, String)> = results
            .iter()
            .map(|r| (r.source.result_id(), r.content.clone()))
            .collect();

        let pairs = detect_contradictions(&items, &DetectorConfig::default());

        // T2.4/D2: Optionally record detected contradictions as real,
        // hash-chained claim-ledger entries (LedgerEvent::ContradictionCandidate)
        // and update the in-process ClaimTrustIndex so search-time trust
        // lookups reflect the conflict. The trust index is a lookup cache
        // only — the ledger entries below are the durable record.
        let (ledger_recorded, ledger_entries) = if record_to_ledger.unwrap_or(false) {
            #[cfg(feature = "claim-integration")]
            {
                use claim_ledger::{LedgerEntryBuilder, SupportState};
                let mut count = 0usize;
                let mut entries = Vec::new();
                for p in &pairs {
                    let fact_a = p.a.strip_prefix("fact:").unwrap_or(&p.a).to_string();
                    let fact_b = p.b.strip_prefix("fact:").unwrap_or(&p.b).to_string();
                    {
                        let mut idx = self.claim_trust.lock().unwrap();
                        idx.record_judgment(fact_a.clone(), SupportState::Contradicted);
                        idx.record_judgment(fact_b.clone(), SupportState::Contradicted);
                    }

                    let pattern = p
                        .signals
                        .iter()
                        .map(|s| format!("{s:?}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    let contradiction_id =
                        claim_ledger::ids::contradiction_id(&fact_a, &fact_b, &pattern);

                    let mut ledger = self.claim_ledger_store.lock().unwrap();
                    let sequence = ledger.next_sequence();
                    let previous_digest = ledger.last_digest();
                    let append_result = LedgerEntryBuilder::new(sequence, previous_digest)
                        .add_contradiction_candidate(
                            &contradiction_id,
                            vec![fact_a.clone(), fact_b.clone()],
                            &pattern,
                            &p.reason,
                        )
                        .map_err(|e| e.to_string())
                        .and_then(|entry| ledger.append(entry));
                    drop(ledger);

                    match append_result {
                        Ok(entry_hash) => {
                            count += 1;
                            entries.push(serde_json::json!({
                                "contradiction_id": contradiction_id,
                                "claim_refs": [fact_a, fact_b],
                                "sequence": sequence,
                                "entry_hash": entry_hash,
                            }));
                        }
                        Err(e) => {
                            return Err(ErrorData::internal_error(
                                format!("failed to record contradiction to ledger: {e}"),
                                None,
                            ));
                        }
                    }
                }
                (count, entries)
            }
            #[cfg(not(feature = "claim-integration"))]
            {
                (0, Vec::new())
            }
        } else {
            (0, Vec::new())
        };

        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "items_scanned": items.len(),
            "contradictions": pairs.iter().map(|p| serde_json::json!({
                "a": p.a,
                "b": p.b,
                "score": p.score,
                "signals": p.signals.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>(),
                "reason": p.reason,
            })).collect::<Vec<_>>(),
            "count": pairs.len(),
            "ledger_recorded": ledger_recorded,
            "ledger_entries": ledger_entries,
            "receipt": mcp_receipt("sm_detect_contradictions"),
        }))
    }

    #[tool(
        description = "Second-order retrieval: find items related to your search results through the graph, but NOT themselves direct hits. Loads edges from store automatically.",
        annotations(read_only_hint = true)
    )]
    fn sm_discord_search(
        &self,
        Parameters(DiscordSearchParams { direct_result_ids }): Parameters<DiscordSearchParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::discord::DiscordScorer;

        // Use neighborhood loading: only load edges within 2 hops of the
        // direct result IDs instead of the entire graph.
        let edges = load_neighborhood_edge_refs(&self.bridge.store, &direct_result_ids)?;
        let scorer = DiscordScorer::with_defaults();
        let results = scorer.score(&direct_result_ids, &edges);

        json_to_string(&serde_json::json!({
            "ok": true,
            "discord_results": results.iter().map(|r| serde_json::json!({
                "item_id": r.item_id,
                "discord_score": r.discord_score,
                "anchor_ids": r.anchor_ids,
                "relationship_types": r.relationship_types,
            })).collect::<Vec<_>>(),
            "count": results.len(),
            "edges_loaded": edges.len(),
            "edges_scope": "neighborhood",
        }))
    }

    #[tool(
        description = "Set provenance (evidence confidence) for an item. Confidence in [0.0, 1.0] with support count. Returns a provenance receipt.",
        annotations(idempotent_hint = true)
    )]
    fn sm_set_provenance(
        &self,
        Parameters(SetProvenanceParams {
            item_id,
            confidence,
            support_count,
        }): Parameters<SetProvenanceParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::provenance::{
            ConfidenceSemiring, ConfidenceValue, ProvenanceItemType,
        };

        // SM-AUD-015: Validate confidence is finite and in [0, 1].
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(ErrorData::invalid_params(
                format!("confidence must be a finite value in [0.0, 1.0], got {confidence}"),
                None,
            ));
        }

        let value = ConfidenceValue::new(confidence, support_count);
        let store = &self.bridge.store;

        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.set_provenance::<ConfidenceSemiring>(
                &ProvenanceItemType::Fact,
                &item_id,
                &value,
                &[],
                None,
            ))
        });

        match result {
            Ok(receipt) => json_to_string(&serde_json::json!({
                "ok": true,
                "provenance_id": receipt.provenance_id,
                "item_id": receipt.item_id,
                "semiring_type": receipt.semiring_type,
                "recorded_at": receipt.recorded_at,
                "message": "Provenance set successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Provenance error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Run a memory lifecycle pass: analyze items for syndromes, compute corrections, identify subtraction candidates, and check compression needs.",
        annotations(read_only_hint = true)
    )]
    fn sm_run_lifecycle(
        &self,
        Parameters(RunLifecycleParams { item_ids }): Parameters<RunLifecycleParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::decoder::{compute_correction, detect_syndromes};
        use semantic_memory::integration::{
            corrections_to_subtraction_candidates, should_trigger_recompression,
        };

        let results: Vec<(String, f64)> = item_ids.iter().map(|id| (id.clone(), 0.5)).collect();
        let syndromes = detect_syndromes(&results, &[]);
        let corrections = compute_correction(&syndromes, 10.0);

        let sub_candidates = corrections_to_subtraction_candidates(&corrections);

        let subtracted_count = sub_candidates.len();
        let remaining_count = item_ids.len().saturating_sub(subtracted_count);
        let recompression = should_trigger_recompression(subtracted_count, remaining_count, false);

        let store = &self.bridge.store;
        let graph_edges = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_all_graph_edges())
        });
        let stored_edges: Vec<(String, String)> = graph_edges
            .as_ref()
            .map(|edges| {
                edges
                    .iter()
                    .map(|edge| (edge.source.clone(), edge.target.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let mut topology_voids: Vec<serde_json::Value> = Vec::new();
        let mut betti = serde_json::json!({
            "betti_0": 0usize,
            "betti_1": 0usize,
        });
        #[allow(unused_mut)]
        let mut topology_error: Option<String> = None;

        let mut communities: Vec<serde_json::Value> = Vec::new();
        let mut community_contradictions: Vec<serde_json::Value> = Vec::new();
        #[allow(unused_mut)]
        let mut community_error: Option<String> = None;

        let mut subgraph_assessment = serde_json::json!({
            "subgraphs_identified": 0usize,
            "subgraphs_pruned": 0usize,
        });
        #[allow(unused_mut)]
        let mut subgraph_error: Option<String> = None;

        #[cfg(feature = "full")]
        {
            use std::collections::HashMap;

            if !stored_edges.is_empty() {
                let analysis_edges = stored_edges.clone();

                use semantic_memory::topology::{compute_betti_numbers, find_voids};

                let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
                for (left, right) in &analysis_edges {
                    adjacency
                        .entry(left.clone())
                        .or_default()
                        .push(right.clone());
                    adjacency
                        .entry(right.clone())
                        .or_default()
                        .push(left.clone());
                }

                let betti_numbers = compute_betti_numbers(&adjacency);
                betti = serde_json::json!({
                    "betti_0": betti_numbers.betti_0,
                    "betti_1": betti_numbers.betti_1,
                });

                topology_voids = find_voids(&analysis_edges)
                    .into_iter()
                    .map(|v| {
                        serde_json::json!({
                            "description": v.description,
                            "void_type": format!("{:?}", v.void_type),
                            "nearby_items": v.nearby_items,
                            "suggested_connections": v.suggested_connections,
                        })
                    })
                    .collect();

                use semantic_memory::community::{
                    community_contradiction_scan, detect_communities,
                };

                let detected = detect_communities(&analysis_edges, 1.0, 42);
                communities = detected
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "id": c.id,
                            "members": c.members,
                            "level": c.level,
                            "parent": c.parent,
                            "member_count": c.members.len(),
                        })
                    })
                    .collect();

                community_contradictions = community_contradiction_scan(&detected, &[])
                    .into_iter()
                    .map(|cc| {
                        serde_json::json!({
                            "community_id": cc.community_id,
                            "item_a": cc.item_a,
                            "item_b": cc.item_b,
                            "description": cc.description,
                        })
                    })
                    .collect();

                use semantic_memory::integration::autonomous_subgraph_maintenance;
                use semantic_memory::subgraph_pruning::AccessLog;
                use std::collections::HashSet;

                let mut access_items: HashSet<String> = HashSet::new();
                for (left, right) in &analysis_edges {
                    access_items.insert(left.clone());
                    access_items.insert(right.clone());
                }

                let access_logs = access_items
                    .into_iter()
                    .map(|item| AccessLog {
                        item_id: item,
                        access_count: 1,
                        last_accessed: "1970-01-01T00:00:00Z".to_string(),
                    })
                    .collect::<Vec<_>>();

                let report = autonomous_subgraph_maintenance(&analysis_edges, &access_logs, &[], 0);
                subgraph_assessment = serde_json::json!({
                    "subgraphs_identified": report.subgraphs_identified,
                    "subgraphs_pruned": report.subgraphs_pruned,
                    "summary": report.summary,
                });
            }
        }

        #[cfg(not(feature = "full"))]
        {
            if !stored_edges.is_empty() {
                topology_error = Some(
                    "topology/community/subgraph phases require the `full` feature".to_string(),
                );
                community_error = Some(
                    "topology/community/subgraph phases require the `full` feature".to_string(),
                );
                subgraph_error = Some(
                    "topology/community/subgraph phases require the `full` feature".to_string(),
                );
            }
        }

        #[cfg(feature = "full")]
        let (f32_count, compressed_count) =
            item_ids
                .iter()
                .fold((0usize, 0usize), |(f32_count, compressed_count), _| {
                    use semantic_memory::compression_governor::{
                        decide_quantization, QuantizationLevel,
                    };

                    match decide_quantization(0.5) {
                        QuantizationLevel::F32 => (f32_count + 1, compressed_count),
                        _ => (f32_count, compressed_count + 1),
                    }
                });
        #[cfg(not(feature = "full"))]
        let (f32_count, compressed_count) = (0usize, 0usize);

        json_to_string(&serde_json::json!({
            "ok": true,
            "items_analyzed": item_ids.len(),
            "syndromes_detected": syndromes.len(),
            "corrections_computed": corrections.len(),
            "subtraction_candidates": sub_candidates.iter().map(|c| serde_json::json!({
                "item_id": c.item_id,
                "structuring_score": c.structuring_score,
                "operation_type": c.operation_type,
                "reason": c.reason,
            })).collect::<Vec<_>>(),
            "recompression_triggered": recompression.triggered,
            "recompression_reason": recompression.reason,
            "topology": {
                "enabled": !stored_edges.is_empty(),
                "voids": topology_voids,
                "void_count": topology_voids.len(),
                "betti_numbers": betti,
                "error": topology_error,
            },
            "community_detection": {
                "enabled": !stored_edges.is_empty(),
                "communities": communities,
                "community_count": communities.len(),
                "contradictions": community_contradictions,
                "contradiction_count": community_contradictions.len(),
                "error": community_error,
            },
            "subgraph_pruning_assessment": {
                "enabled": !stored_edges.is_empty(),
                "subgraph_count": subgraph_assessment["subgraphs_identified"].as_u64().unwrap_or(0),
                "pruned_count": subgraph_assessment["subgraphs_pruned"].as_u64().unwrap_or(0),
                "summary": subgraph_assessment["summary"].as_str().unwrap_or(""),
                "error": subgraph_error,
            },
            "turbo_quantization_assessment": {
                "items_assessed": item_ids.len(),
                "would_retain_f32": f32_count,
                "would_compress": compressed_count,
            },
            "summary": format!(
                "Analyzed {} items: {} syndromes, {} corrections, {} subtraction candidates, recompression: {}",
                item_ids.len(), syndromes.len(), corrections.len(), sub_candidates.len(),
                if recompression.triggered { "needed" } else { "not needed" }
            ),
        }))
    }

    // ── First-class graph edge tools ───────────────────────────────

    #[tool(
        description = "Add a durable, typed graph edge between two nodes. Edge types: semantic, temporal, causal, entity. Idempotent — same edge returns existing ID.",
        annotations(idempotent_hint = true)
    )]
    fn sm_add_graph_edge(
        &self,
        Parameters(params): Parameters<AddGraphEdgeParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::GraphEdgeType;

        // SM-AUD-015: Validate numeric params are finite and in range.
        if let Some(cs) = params.cosine_similarity {
            if !cs.is_finite() || !(0.0..=1.0).contains(&cs) {
                return Err(ErrorData::invalid_params(
                    format!("cosine_similarity must be finite and in [0.0, 1.0], got {cs}"),
                    None,
                ));
            }
        }
        if let Some(conf) = params.confidence {
            if !conf.is_finite() || !(0.0..=1.0).contains(&conf) {
                return Err(ErrorData::invalid_params(
                    format!("confidence must be finite and in [0.0, 1.0], got {conf}"),
                    None,
                ));
            }
        }

        let edge_type = match params.edge_type {
            EdgeType::Semantic => GraphEdgeType::Semantic {
                cosine_similarity: params.cosine_similarity.unwrap_or(0.5),
            },
            EdgeType::Temporal => GraphEdgeType::Temporal {
                delta_secs: params.delta_secs.unwrap_or(0),
            },
            EdgeType::Causal => GraphEdgeType::Causal {
                confidence: params.confidence.unwrap_or(0.5),
                evidence_ids: params.evidence_ids.unwrap_or_default(),
            },
            EdgeType::Entity => GraphEdgeType::Entity {
                relation: params.relation.unwrap_or_else(|| "related".to_string()),
            },
        };

        // MCP-004: Reject malformed metadata JSON instead of silently dropping it.
        let metadata = match params.metadata.as_deref() {
            None => None,
            Some(s) => match serde_json::from_str::<serde_json::Value>(s) {
                Ok(v) => Some(v),
                Err(e) => {
                    return Err(ErrorData::invalid_params(
                        format!("metadata is not valid JSON: {e}"),
                        None,
                    ))
                }
            },
        };

        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_graph_edge(
                &params.source,
                &params.target,
                edge_type,
                params.weight,
                metadata,
            ))
        });

        match result {
            Ok(edge) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": mcp_receipt("sm_add_graph_edge"),
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "content_digest": edge.content_digest,
                "recorded_at": edge.recorded_at,
                "message": "Graph edge added successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Error adding graph edge: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "List graph edges for a specific node (as source or target), or all edges if no node_id. Returns non-invalidated edges only.",
        annotations(read_only_hint = true)
    )]
    fn sm_list_graph_edges(
        &self,
        Parameters(ListGraphEdgesParams { node_id }): Parameters<ListGraphEdgesParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = match node_id {
            Some(id) => tokio::task::block_in_place(|| {
                Handle::current().block_on(store.list_graph_edges_for_node(&id))
            }),
            None => tokio::task::block_in_place(|| {
                Handle::current().block_on(store.list_all_graph_edges())
            }),
        };

        match result {
            Ok(edges) => json_to_string(&serde_json::json!({
                "ok": true,
                "edges": edges.iter().map(|e| serde_json::json!({
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "metadata": e.metadata,
                    "recorded_at": e.recorded_at,
                })).collect::<Vec<_>>(),
                "count": edges.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Error listing graph edges: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Invalidate a stored graph edge by ID. Append-only — edge is never deleted, only marked invalidated with a reason.",
        annotations(idempotent_hint = true)
    )]
    fn sm_invalidate_graph_edge(
        &self,
        Parameters(InvalidateGraphEdgeParams { edge_id, reason }): Parameters<
            InvalidateGraphEdgeParams,
        >,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.invalidate_graph_edge(&edge_id, &reason))
        });

        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": mcp_receipt("sm_invalidate_graph_edge"),
                "edge_id": edge_id,
                "message": "Edge invalidated successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("Error invalidating edge: {e}"),
                None,
            )),
        }
    }

    // ── Factor graph, topology, and community tools ─────────────────

    #[tool(
        description = "Run factor graph belief propagation on stored graph edges. Models all 4 edge types as factors. Returns unified confidence scores after convergence.",
        annotations(read_only_hint = true)
    )]
    fn sm_factor_graph(
        &self,
        Parameters(params): Parameters<FactorGraphParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::factor_graph::{factors_from_edges, FactorGraph, FactorGraphConfig};

        let defaults = FactorGraphConfig::default();
        let config = FactorGraphConfig {
            semantic_weight: params.semantic_weight.unwrap_or(defaults.semantic_weight),
            temporal_weight: params.temporal_weight.unwrap_or(defaults.temporal_weight),
            causal_weight: params.causal_weight.unwrap_or(defaults.causal_weight),
            entity_weight: params.entity_weight.unwrap_or(defaults.entity_weight),
            self_influence: params.self_influence.unwrap_or(defaults.self_influence),
            max_iterations: params
                .max_iterations
                .map(|v| v as usize)
                .unwrap_or(defaults.max_iterations),
            convergence_threshold: params
                .convergence_threshold
                .unwrap_or(defaults.convergence_threshold),
        };

        // Use neighborhood loading: only load edges within 2 hops of the
        // node seeds instead of the entire graph.
        let seed_ids: Vec<String> = params.nodes.iter().map(|n| n.item_id.clone()).collect();
        let raw_edges = load_neighborhood_factor_edges(&self.bridge.store, &seed_ids)?;
        let factors = factors_from_edges(&raw_edges);

        let nodes: Vec<(String, f64)> = params
            .nodes
            .iter()
            .map(|n| (n.item_id.clone(), n.initial_belief))
            .collect();

        let graph = FactorGraph::new(&nodes, factors, config);
        let result = graph.propagate();

        json_to_string(&serde_json::json!({
            "ok": true,
            "node_beliefs": result.node_beliefs,
            "iterations": result.iterations,
            "converged": result.converged,
            "elapsed_ms": result.elapsed_ms,
            "edges_loaded": raw_edges.len(),
            "edges_scope": "neighborhood",
            "factor_counts": {
                "semantic": result.factor_counts.semantic,
                "temporal": result.factor_counts.temporal,
                "causal": result.factor_counts.causal,
                "entity": result.factor_counts.entity,
                "total": result.factor_counts.total(),
            },
            "config": {
                "semantic_weight": result.config.semantic_weight,
                "temporal_weight": result.config.temporal_weight,
                "causal_weight": result.config.causal_weight,
                "entity_weight": result.config.entity_weight,
                "self_influence": result.config.self_influence,
                "max_iterations": result.config.max_iterations,
                "convergence_threshold": result.config.convergence_threshold,
            },
        }))
    }

    #[tool(
        description = "Find topological voids in the knowledge graph. Computes Betti numbers (components and cycles) and detects structural gaps. Loads edges from store.",
        annotations(read_only_hint = true)
    )]
    fn sm_topology(
        &self,
        Parameters(_params): Parameters<TopologyParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::topology::{compute_betti_numbers, find_voids, gap_report};

        // MCP-001: Load edges from the store, not from caller-supplied params.
        let edges = load_stored_edge_pairs(&self.bridge.store)?;

        let mut adjacency: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for (src, tgt) in &edges {
            adjacency.entry(src.clone()).or_default().push(tgt.clone());
            adjacency.entry(tgt.clone()).or_default().push(src.clone());
        }

        let betti = compute_betti_numbers(&adjacency);
        let voids = find_voids(&edges);
        let report = gap_report(&voids);

        json_to_string(&serde_json::json!({
            "ok": true,
            "betti_numbers": {
                "betti_0": betti.betti_0,
                "betti_1": betti.betti_1,
            },
            "voids": voids.iter().map(|v| serde_json::json!({
                "description": v.description,
                "nearby_items": v.nearby_items,
                "suggested_connections": v.suggested_connections,
                "void_type": format!("{:?}", v.void_type),
            })).collect::<Vec<_>>(),
            "void_count": voids.len(),
            "edges_loaded_from_store": edges.len(),
            "report": report,
        }))
    }

    #[tool(
        description = "Detect communities in the knowledge graph (Leiden-inspired). Returns community assignments, optional contradiction scans, and compression recommendations.",
        annotations(read_only_hint = true)
    )]
    fn sm_community(
        &self,
        Parameters(params): Parameters<CommunityParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::community::{
            community_aware_compression, community_contradiction_scan, detect_communities,
        };

        // MCP-001: Load edges from the store, not from caller-supplied params.
        let edges = load_stored_edge_pairs(&self.bridge.store)?;

        let resolution = params.resolution.unwrap_or(1.0);
        let seed = params.seed.unwrap_or(42);

        let communities = detect_communities(&edges, resolution, seed);

        let contradictions = params.contradictions.unwrap_or_default();
        let community_contras = community_contradiction_scan(&communities, &contradictions);

        let importance_scores = params.importance_scores.unwrap_or_default();
        let compression = community_aware_compression(&communities, &importance_scores);

        let summarize = params.summarize.unwrap_or(false);
        let store = &self.bridge.store;
        let communities_json: Vec<serde_json::Value> = communities
            .iter()
            .map(|c| {
                let summary: Option<String> = if summarize && !c.members.is_empty() {
                    let member_texts: Vec<String> = c
                        .members
                        .iter()
                        .filter_map(|mid| {
                            let bare = mid.strip_prefix("fact:").unwrap_or(mid);
                            tokio::task::block_in_place(|| {
                                Handle::current().block_on(store.get_fact(bare))
                            })
                            .ok()
                            .flatten()
                            .map(|f| f.content)
                        })
                        .collect();
                    if !member_texts.is_empty() {
                        let combined = member_texts.join("\n---\n");
                        let prompt = format!(
                            "Summarize these related facts in 1-2 sentences:\n{combined}\nSummary:"
                        );
                        let body = serde_json::json!({
                            "model": "granite4.1:3b",
                            "prompt": prompt,
                            "stream": false,
                            "options": {"temperature": 0, "num_predict": 100}
                        });
                        reqwest::blocking::Client::new()
                            .post("http://127.0.0.1:11434/api/generate")
                            .json(&body)
                            .send()
                            .ok()
                            .and_then(|resp| resp.json::<serde_json::Value>().ok())
                            .and_then(|v| {
                                v.get("response")
                                    .and_then(|r| r.as_str())
                                    .map(|s| s.trim().to_string())
                            })
                    } else {
                        None
                    }
                } else {
                    None
                };
                serde_json::json!({
                    "id": c.id,
                    "members": c.members,
                    "level": c.level,
                    "parent": c.parent,
                    "member_count": c.members.len(),
                    "summary": summary,
                })
            })
            .collect();

        json_to_string(&serde_json::json!({
            "ok": true,
            "communities": communities_json,
            "community_count": communities.len(),
            "contradictions": community_contras.iter().map(|cc| serde_json::json!({
                "community_id": cc.community_id,
                "item_a": cc.item_a,
                "item_b": cc.item_b,
                "description": cc.description,
            })).collect::<Vec<_>>(),
            "contradiction_count": community_contras.len(),
            "compression_recommendations": compression.iter().map(|cr| serde_json::json!({
                "community_id": cr.community_id,
                "quantization_level": cr.quantization_level,
                "reason": cr.reason,
            })).collect::<Vec<_>>(),
            "compression_count": compression.len(),
            "edges_loaded_from_store": edges.len(),
        }))
    }

    // ── Delete / forget tools (admin-ops) ────────────────────────────
    // Governed forgetting. Corrected facts should still use supersession; this
    // tool closes the selected fact and all derived access paths while retaining
    // a content-free tombstone receipt.

    #[tool(
        description = "Identify and optionally prune reasoning subgraphs in the knowledge graph. Runs autonomous subgraph maintenance: identifies connected subgraphs, ranks by access frequency (least-accessed first), and optionally prunes. Dry-run by default. Returns identified subgraphs, pruning priority, and pruning receipts.",
        annotations(read_only_hint = true)
    )]
    fn sm_subgraph_prune(
        &self,
        Parameters(SubgraphPruneParams { dry_run, max_prune }): Parameters<SubgraphPruneParams>,
    ) -> Result<String, ErrorData> {
        #[cfg(feature = "subgraph-pruning")]
        {
            use semantic_memory::integration::autonomous_subgraph_maintenance;
            use semantic_memory::subgraph_pruning::AccessLog;

            let dry = dry_run.unwrap_or(true);
            let max = max_prune.map(|v| v as usize).unwrap_or(5);

            // Load edges from store
            let edges = load_stored_edge_pairs(&self.bridge.store)?;

            // No access logs available — derive empty (all subgraphs treated as equally stale)
            let access_logs: Vec<AccessLog> = Vec::new();

            // Load contradictions from contradiction graph edges
            let raw_edges = tokio::task::block_in_place(|| {
                Handle::current().block_on(self.bridge.store.list_all_graph_edges())
            })
            .map_err(|e| ErrorData::internal_error(format!("load edges failed: {e}"), None))?;
            let contradictions: Vec<(String, String)> = raw_edges
                .iter()
                .filter_map(|e| {
                    let parsed = e
                        .edge_type_parsed
                        .clone()
                        .or_else(|| serde_json::from_str(&e.edge_type).ok());
                    match parsed {
                        Some(semantic_memory::GraphEdgeType::Entity { relation })
                            if relation == "contradicts" =>
                        {
                            Some((e.source.clone(), e.target.clone()))
                        }
                        _ => None,
                    }
                })
                .collect();

            let prune_count = if dry { 0 } else { max };
            let report =
                autonomous_subgraph_maintenance(&edges, &access_logs, &contradictions, prune_count);

            json_to_string(&serde_json::json!({
                "ok": true,
                "dry_run": dry,
                "subgraphs_identified": report.subgraphs_identified,
                "subgraphs_pruned": report.subgraphs_pruned,
                "receipts": report.receipts.iter().map(|r| serde_json::json!({
                    "subgraph_root": r.subgraph_root,
                    "pruned_nodes": r.pruned_nodes,
                })).collect::<Vec<_>>(),
                "summary": report.summary,
                "receipt_field": mcp_receipt("sm_subgraph_prune"),
            }))
        }
        #[cfg(not(feature = "subgraph-pruning"))]
        {
            let _ = (dry_run, max_prune);
            json_to_string(&serde_json::json!({
                "ok": true,
                "note": "subgraph-pruning feature not enabled",
                "receipt": mcp_receipt("sm_subgraph_prune"),
            }))
        }
    }

    #[tool(
        description = "Forget a single fact by id through the governed dependency-closure path. Scrubs canonical content and derived FTS/vector/graph/cache/export/replay surfaces while retaining a content-free tombstone receipt. Prefer sm_supersede_fact for corrections.",
        annotations(destructive_hint = true)
    )]
    fn sm_delete_fact(
        &self,
        Parameters(DeleteFactParams { fact_id }): Parameters<DeleteFactParams>,
    ) -> Result<String, ErrorData> {
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;
        let fact = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.get_fact_raw_compat(&bare))
        })
        .map_err(|e| ErrorData::internal_error(format!("delete_fact lookup error: {e}"), None))?
        .ok_or_else(|| ErrorData::invalid_params(format!("fact not found: fact:{bare}"), None))?;
        let origin_record = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.authority().get_origin_authority(&bare))
        })
        .map_err(|e| {
            ErrorData::internal_error(format!("delete_fact origin lookup error: {e}"), None)
        })?
        .ok_or_else(|| {
            ErrorData::invalid_params(
                format!("fact has no governed origin authority: fact:{bare}"),
                None,
            )
        })?;
        let resource_principal = origin_record.label.origin_principal;
        let permit = semantic_memory::AuthorityPermit::operator_system(
            resource_principal.clone(),
            "caller:sm_delete_fact",
            semantic_memory::AuthorityPermit::FORGET_CAPABILITY,
        )
        .with_origin(semantic_memory::OriginAuthorityLabelV1::operator_system(
            &resource_principal,
            "caller:sm_delete_fact",
        ));
        let request = semantic_memory::ForgettingClosureRequestV1::new(
            vec![bare.clone()],
            fact.namespace,
            "explicit MCP fact-forgetting request",
            4096,
        );
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.authority().forget(
                permit,
                format!("mcp-sm-delete-fact:{}", uuid::Uuid::new_v4()),
                request,
            ))
        });
        match result {
            Ok(receipt) => json_to_string(&serde_json::json!({
                "ok": true,
                "deleted": false,
                "forgotten": true,
                "fact_id": format!("fact:{bare}"),
                "forgetting_receipt": receipt,
                "message": "Fact forgotten through governed dependency closure",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("delete_fact forgetting error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Permanently delete ALL memory in a namespace — facts, documents, chunks, sessions/messages. HARD delete, irreversible. Returns per-surface deletion count.",
        annotations(destructive_hint = true)
    )]
    fn sm_delete_namespace(
        &self,
        Parameters(DeleteNamespaceParams { namespace }): Parameters<DeleteNamespaceParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.delete_namespace(&namespace))
        });
        match result {
            Ok(r) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": mcp_receipt("sm_delete_namespace"),
                "namespace": namespace,
                "deleted": {
                    "facts": r.facts,
                    "documents": r.documents,
                    "chunks": r.chunks,
                    "messages": r.messages,
                    "sessions": r.sessions,
                    "episodes": r.episodes,
                    "projection_rows": r.projection_rows,
                },
                "message": "Namespace permanently deleted",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("delete_namespace error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Update a fact's content in-place. Re-embeds the fact and updates FTS index. Use this to correct outdated facts without deleting and re-adding.",
        annotations(idempotent_hint = true)
    )]
    fn sm_update_fact(
        &self,
        Parameters(UpdateFactParams { fact_id, content }): Parameters<UpdateFactParams>,
    ) -> Result<String, ErrorData> {
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.update_fact(&bare, &content))
        });
        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": mcp_receipt("sm_update_fact"),
                "fact_id": format!("fact:{bare}"),
                "message": "Fact content updated and re-embedded",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("update_fact error: {e}"),
                None,
            )),
        }
    }

    // NOTE: Consolidation requires atomic transaction support. Excluded from stable profile until transaction safety is implemented.
    #[tool(
        description = "Consolidate two near-duplicate facts into one. Merges their content, updates the kept fact, and supersedes the other with a 'consolidated with' edge. Use this to clean up duplicate knowledge."
    )]
    fn sm_consolidate_facts(
        &self,
        Parameters(ConsolidateFactsParams {
            keep_id,
            supersede_id,
            merged_content,
        }): Parameters<ConsolidateFactsParams>,
    ) -> Result<String, ErrorData> {
        let keep_bare = keep_id
            .strip_prefix("fact:")
            .unwrap_or(&keep_id)
            .to_string();
        let sup_bare = supersede_id
            .strip_prefix("fact:")
            .unwrap_or(&supersede_id)
            .to_string();
        let store = &self.bridge.store;

        // Get both facts to determine namespace and merge content
        let keep_fact =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&keep_bare)));
        let sup_fact =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&sup_bare)));

        let (namespace, final_content) = match (keep_fact, sup_fact) {
            (Ok(Some(k)), Ok(Some(s))) => {
                let ns = k.namespace.clone();
                let content = merged_content.unwrap_or_else(|| {
                    if k.content.len() >= s.content.len() {
                        if !k.content.contains(&s.content) {
                            format!("{}\n\nAdditional: {}", k.content, s.content)
                        } else {
                            k.content.clone()
                        }
                    } else if !s.content.contains(&k.content) {
                        format!("{}\n\nAdditional: {}", s.content, k.content)
                    } else {
                        s.content.clone()
                    }
                });
                (ns, content)
            }
            (Ok(Some(k)), _) => (
                k.namespace.clone(),
                merged_content.unwrap_or(k.content.clone()),
            ),
            (Err(_), _) | (Ok(None), _) => {
                return Err(ErrorData::internal_error(
                    "keep fact not found".to_string(),
                    None,
                ));
            }
        };

        // Update the kept fact with merged content
        let update_result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.update_fact(&keep_bare, &final_content))
        });
        if let Err(e) = update_result {
            return Err(ErrorData::internal_error(
                format!("update keep fact error: {e}"),
                None,
            ));
        }

        // Supersede the other fact: add a new fact with merged content and link with "supersedes" edge
        use semantic_memory::GraphEdgeType;
        let new_id = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.add_fact(&namespace, &final_content, None, None))
        });
        match new_id {
            Ok(nid) => {
                let new_node = format!("fact:{nid}");
                let old_node = format!("fact:{sup_bare}");
                let metadata = serde_json::json!({
                    "reason": "consolidated duplicate",
                    "consolidated_with": format!("fact:{}", keep_bare),
                });
                let _edge = tokio::task::block_in_place(|| {
                    Handle::current().block_on(store.add_graph_edge(
                        &new_node,
                        &old_node,
                        GraphEdgeType::Entity {
                            relation: "supersedes".to_string(),
                        },
                        1.0,
                        Some(metadata),
                    ))
                });
                json_to_string(&serde_json::json!({
                    "ok": true,
                    "receipt": mcp_receipt("sm_consolidate_facts"),
                    "kept_fact_id": format!("fact:{}", keep_bare),
                    "superseded_fact_id": format!("fact:{}", sup_bare),
                    "new_fact_id": format!("fact:{}", nid),
                    "message": "Facts consolidated: kept fact updated, duplicate superseded",
                }))
            }
            Err(e) => Err(ErrorData::internal_error(
                format!("supersede error: {e}"),
                None,
            )),
        }
    }

    // ── RL routing feedback ────────────────────────────────────────────

    #[tool(
        description = "Return the current persisted RL routing policy, including weights, training example count, and last update time.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_routing_policy(&self) -> Result<String, ErrorData> {
        use semantic_memory::rl_routing::is_trained;

        let policy = tokio::task::block_in_place(|| {
            Handle::current().block_on(self.bridge.store.load_routing_policy())
        })
        .map_err(|e| ErrorData::internal_error(format!("load routing policy error: {e}"), None))?;

        match policy {
            Some(policy) => json_to_string(&serde_json::json!({
                "ok": true,
                "policy": {
                    "weights": policy.weights,
                    "training_examples_count": policy.trained_examples,
                    "trained": is_trained(&policy),
                    "last_updated": policy.last_updated,
                }
            })),
            None => json_to_string(&serde_json::json!({
                "ok": true,
                "policy": null,
            })),
        }
    }

    #[tool(
        description = "MUTATING: record a caller-supplied proxy feedback label for retrieval routing. This is not a verified outcome. Persists the updated tabular routing policy every 10 outcomes.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = false
        )
    )]
    fn sm_record_outcome(
        &self,
        Parameters(RecordOutcomeParams { query, outcome }): Parameters<RecordOutcomeParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::rl_routing::{
            is_trained, record_routing_outcome, route_with_policy, RoutingOutcome,
        };
        use semantic_memory::routing::{QueryProfile, RetrievalRouter};

        let outcome_enum = match outcome.to_lowercase().as_str() {
            "good" => RoutingOutcome::Good,
            "bad" => RoutingOutcome::Bad,
            "neutral" => RoutingOutcome::Neutral,
            _ => {
                return Err(ErrorData::invalid_params(
                    format!("outcome must be 'good', 'bad', or 'neutral', got '{outcome}'"),
                    None,
                ));
            }
        };

        let profile = QueryProfile::from_query(&query);
        // TODO: Accept decision_id from the original routing receipt, not recompute.
        let router = RetrievalRouter::default();
        let store = &self.bridge.store;
        let mut batch = self
            .routing_policy_batch
            .lock()
            .map_err(|_| ErrorData::internal_error("routing policy batch lock poisoned", None))?;
        let mut policy = match batch.policy.take() {
            Some(policy) => policy,
            None => tokio::task::block_in_place(|| {
                Handle::current().block_on(store.load_routing_policy())
            })
            .map_err(|e| {
                ErrorData::internal_error(format!("load routing policy error: {e}"), None)
            })?
            .unwrap_or_default(),
        };
        let decision = if is_trained(&policy) {
            route_with_policy(&policy, &profile)
        } else {
            router.route(&profile)
        };
        record_routing_outcome(&mut policy, &profile, &decision, outcome_enum);
        batch.pending_outcomes += 1;
        let persisted = batch.pending_outcomes >= ROUTING_POLICY_PERSIST_BATCH;
        if persisted {
            if let Err(e) = tokio::task::block_in_place(|| {
                Handle::current().block_on(store.save_routing_policy(&policy))
            }) {
                batch.policy = Some(policy);
                return Err(ErrorData::internal_error(
                    format!("persist routing policy error: {e}"),
                    None,
                ));
            }
            batch.pending_outcomes = 0;
        }
        let pending_outcomes = batch.pending_outcomes;
        batch.policy = Some(policy.clone());

        json_to_string(&serde_json::json!({
            "ok": true,
            "receipt": mcp_receipt("sm_record_outcome"),
            "mutating": true,
            "query": query,
            "feedback": {"kind": "ProxyLabel", "label": outcome},
            "routing_decision": {
                "bm25_coarse": decision.bm25_coarse,
                "vector_medium": decision.vector_medium,
                "rerank_fine": decision.rerank_fine,
                "graph_expansion": decision.graph_expansion,
                "decoder": decision.decoder,
                "discord": decision.discord,
                "no_retrieval": decision.no_retrieval,
                "reasoning": decision.reasoning,
            },
            "policy_state": {
                "trained_examples": policy.trained_examples,
                "baseline": policy.baseline,
                "weights": policy.weights,
                "last_updated": policy.last_updated,
                "persisted": persisted,
                "pending_outcomes": pending_outcomes,
            },
            "message": "Routing outcome recorded (best-effort). NOTE: outcome is associated with a recomputed decision, not the original served decision.",
        }))
    }

    // ─── Claim-ledger integration ──────────────────────────────────────

    #[cfg(feature = "claim-integration")]
    #[tool(
        description = "Verify and checkpoint the claim ledger when configured entry/byte thresholds are exceeded. Defaults to dry_run=true. A successful rotation atomically activates a digest-verified snapshot and retained tail, emits a compaction receipt bound to the prior ledger head, and keeps bounded backups.",
        annotations(read_only_hint = false, destructive_hint = true)
    )]
    fn sm_compact_claim_ledger(
        &self,
        Parameters(CompactClaimLedgerParams {
            dry_run,
            max_entries,
            max_bytes,
            retain_tail_entries,
            max_backups,
        }): Parameters<CompactClaimLedgerParams>,
    ) -> Result<String, ErrorData> {
        let mut ledger = self.claim_ledger_store.lock().unwrap();
        let result = ledger
            .compact(ClaimLedgerCompactionConfig {
                dry_run: dry_run.unwrap_or(true),
                max_entries: max_entries.unwrap_or(10_000),
                max_bytes: max_bytes.unwrap_or(16 * 1024 * 1024),
                retain_tail_entries: retain_tail_entries.unwrap_or(256),
                max_backups: max_backups.unwrap_or(3),
            })
            .map_err(|error| ErrorData::internal_error(error, None))?;
        if result["compacted"].as_bool().unwrap_or(false) {
            let mut index = self.claim_trust.lock().unwrap();
            index.disable();
            if ledger.trust_enabled {
                index.enabled = true;
                if let Some(snapshot) = &ledger.snapshot {
                    index.load_snapshot(snapshot);
                }
                index.rebuild_from_ledger_incremental(&ledger.entries);
            }
        }
        json_to_string(&result)
    }

        // PREVIEW: does not persist state; requires claim-integration feature and restart-roundtrip persistence implementation.
#[cfg(feature = "claim-integration")]
    #[tool(
        description = "Create a typed Claim from a semantic-memory fact. The claim gets a source-spanned provenance record from the fact's metadata. Returns the claim ID.",
        annotations(read_only_hint = false, idempotent_hint = true)
    )]
    fn sm_create_claim(
        &self,
        Parameters(CreateClaimParams {
            fact_id,
            source_span,
        }): Parameters<CreateClaimParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::Claim;
        let bare = fact_id
            .strip_prefix("fact:")
            .unwrap_or(&fact_id)
            .to_string();
        let store = &self.bridge.store;

        // Get the fact content
        let fact =
            tokio::task::block_in_place(|| Handle::current().block_on(store.get_fact(&bare)));
        let fact = match fact {
            Ok(Some(f)) => f,
            _ => {
                return Err(ErrorData::internal_error(
                    format!("fact not found: {fact_id}"),
                    None,
                ))
            }
        };

        // Create a claim from the fact
        let source_id = format!("semantic-memory:fact:{bare}");
        let span_id = source_span.unwrap_or_else(|| "full".to_string());
        let claim = Claim::new(&source_id, &span_id, &fact.content, "fact");

        let claim_id = claim.claim_id.clone();
        let normalized = &claim.normalized_claim;

        {
            let mut ledger = self.claim_ledger_store.lock().unwrap();
            let sequence = ledger.next_sequence();
            let previous_digest = ledger.last_digest();
            let entry = claim_ledger::LedgerEntryBuilder::new(sequence, previous_digest)
                .add_claim(&claim_id, &source_id, &span_id, normalized)
                .map_err(|e| {
                    ErrorData::internal_error(
                        format!("failed to build claim ledger entry: {e}"),
                        None,
                    )
                })?;
            ledger.append(entry).map_err(|e| {
                ErrorData::internal_error(format!("failed to record claim to ledger: {e}"), None)
            })?;
        }

        {
            let mut idx = self.claim_trust.lock().unwrap();
            idx.link_fact(bare.clone(), claim_id.clone());
            idx.register_claim(claim_id.clone(), normalized.clone());
        }

        json_to_string(&serde_json::json!({
            "ok": true,
            "receipt": mcp_receipt("sm_create_claim"),
            "claim_id": claim_id,
            "source_id": source_id,
            "span_id": span_id,
            "claim_text": fact.content,
            "normalized_claim": normalized,
            "claim_type": "fact",
            "message": "Claim created from semantic-memory fact with source-spanned provenance",
        }))
    }

        // PREVIEW: does not persist state; requires claim-integration feature and restart-roundtrip persistence implementation.
#[cfg(feature = "claim-integration")]
    #[tool(
        description = "Add evidence to a claim. Creates an EvidenceBundle linking the evidence text to the claim. Returns the evidence bundle ID.",
        annotations(read_only_hint = false)
    )]
    fn sm_add_evidence(
        &self,
        Parameters(AddEvidenceParams {
            claim_id,
            evidence_text,
            source_type,
        }): Parameters<AddEvidenceParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{EvidenceBundle, EvidenceLink, EvidenceRelation};
        let mut bundle = EvidenceBundle::new(&claim_id);
        let link = EvidenceLink {
            relation: EvidenceRelation::Supports,
            source_id: source_type.unwrap_or_else(|| "semantic-memory".to_string()),
            span_id: "full".to_string(),
            quote: evidence_text.clone(),
            digest: claim_ledger::ids::sha256_text(&evidence_text),
            support_role: "supporting".to_string(),
        };
        bundle.evidence_links.push(link);

        json_to_string(&serde_json::json!({
            "ok": true,
            "receipt": mcp_receipt("sm_add_evidence"),
            "evidence_bundle_id": bundle.evidence_bundle_id,
            "claim_id": claim_id,
            "evidence_count": bundle.evidence_links.len(),
            "message": "Evidence added to claim",
        }))
    }

        // PREVIEW: does not persist state; requires claim-integration feature and restart-roundtrip persistence implementation.
#[cfg(feature = "claim-integration")]
    #[tool(
        description = "Judge the support state of a claim. Creates a SupportJudgment (supported, unsupported, contested, or heuristic_only) with optional rationale.",
        annotations(read_only_hint = false)
    )]
    fn sm_judge_support(
        &self,
        Parameters(JudgeSupportParams {
            claim_id,
            judgment,
            rationale,
        }): Parameters<JudgeSupportParams>,
    ) -> Result<String, ErrorData> {
        use claim_ledger::{SupportJudgment, SupportState};
        let state = match judgment.to_lowercase().as_str() {
            "supported" => SupportState::Supported,
            "partially_supported" | "partial" => SupportState::PartiallySupported,
            "unsupported" => SupportState::Unsupported,
            "contradicted" | "contested" => SupportState::Contradicted,
            "heuristic_only" | "heuristic" => SupportState::HeuristicOnly,
            _ => return Err(ErrorData::invalid_params(
                format!("Invalid judgment '{judgment}'. Must be: supported, partially_supported, unsupported, contradicted, or heuristic_only"),
                None,
            )),
        };
        let j = SupportJudgment {
            support_judgment_id: claim_ledger::ids::ulid(),
            claim_id: claim_id.clone(),
            evidence_bundle_ref: claim_ledger::ids::evidence_bundle_id(&claim_id),
            support_state: state,
            method: "agent_judgment".to_string(),
            rationale: rationale.unwrap_or_default(),
            contradiction_refs: Vec::new(),
            proof_debt: Vec::new(),
            created_recorded_time: chrono::Utc::now(),
        };

        {
            let mut ledger = self.claim_ledger_store.lock().unwrap();
            let sequence = ledger.next_sequence();
            let previous_digest = ledger.last_digest();
            let entry = claim_ledger::LedgerEntryBuilder::new(sequence, previous_digest)
                .add_support_judgment(
                    &j.support_judgment_id,
                    &claim_id,
                    &j.evidence_bundle_ref,
                    state,
                    &j.method,
                )
                .map_err(|e| {
                    ErrorData::internal_error(
                        format!("failed to build support judgment ledger entry: {e}"),
                        None,
                    )
                })?;
            ledger.append(entry).map_err(|e| {
                ErrorData::internal_error(
                    format!("failed to record support judgment to ledger: {e}"),
                    None,
                )
            })?;
        }

        self.claim_trust
            .lock()
            .unwrap()
            .record_judgment(claim_id.clone(), state);

        json_to_string(&serde_json::json!({
            "ok": true,
            "receipt": mcp_receipt("sm_judge_support"),
            "support_judgment_id": j.support_judgment_id,
            "claim_id": claim_id,
            "state": judgment.to_lowercase(),
            "message": "Support judgment recorded",
        }))
    }

    // ─── Bitemporal search ─────────────────────────────────────────────

    #[tool(
        description = "PREVIEW: Temporal filtering is not yet implemented. Results are NOT filtered by the provided timestamp. This tool exists for API exploration only.",
        annotations(read_only_hint = true)
    )]
    fn sm_search_as_of_preview(
        &self,
        Parameters(SearchAsOfParams {
            query,
            as_of_date,
            top_k,
            namespace,
        }): Parameters<SearchAsOfParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let k = top_k.unwrap_or(5);
        let ns_slice: Option<Vec<&str>> = namespace.as_ref().map(|n| vec![n.as_str()]);

        // Parse the as-of date
        let _as_of = chrono::DateTime::parse_from_rfc3339(&as_of_date)
            .map_err(|e| ErrorData::invalid_params(
                format!("Invalid as_of_date '{as_of_date}': {e}. Use ISO 8601 format like 2026-01-15T00:00:00Z"),
                None,
            ))?
            .with_timezone(&chrono::Utc);

        // Search normally, then filter by date
        let results = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.search_with_view(
                &query,
                Some(k * 2),
                ns_slice.as_deref(),
                None,
                semantic_memory::StateView::HistoricalAt(as_of_date.clone()),
            ))
        })
        .map_err(|e| ErrorData::internal_error(format!("search error: {e}"), None))?;

        let filtered: Vec<_> = results.into_iter().take(k).collect();

        let result_json: Vec<serde_json::Value> = filtered
            .iter()
            .map(|r| {
                serde_json::json!({
                    "result_id": r.source.result_id(),
                    "content": r.content,
                    "score": r.score,
                })
            })
            .collect();

        json_to_string(&serde_json::json!({
            "ok": true,
            "query": query,
            "as_of_date": as_of_date,
            "results": result_json,
            "count": filtered.len(),
            "message": format!("Found {} facts valid as of {}", filtered.len(), as_of_date),
        }))
    }

    // ─── Verification gate ─────────────────────────────────────────────

        // PREVIEW: does not persist state; requires claim-integration feature and restart-roundtrip persistence implementation.
    #[cfg(feature = "claim-integration")]
#[tool(
        description = "Verify a claim against risk class requirements. Low/medium claims need cheap checks. High claims need falsification. Critical claims need replay AND falsification. Returns disposition: promote, reject, quarantine, or defer.",
        annotations(read_only_hint = true)
    )]
    fn sm_verify_claim(
        &self,
        Parameters(VerifyClaimParams {
            claim,
            risk_class,
            evidence_refs,
            refutation_attempted,
        }): Parameters<VerifyClaimParams>,
    ) -> Result<String, ErrorData> {
        let risk = risk_class.to_lowercase();
        let has_evidence = evidence_refs
            .as_ref()
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let refuted = refutation_attempted.unwrap_or(false);

        // Required checks by risk class
        let (needs_replay, needs_falsification, disposition, rationale) = match risk.as_str() {
            "low" => (
                false,
                false,
                "promote",
                "Low risk: cheap checks only, claim can be promoted",
            ),
            "medium" => (
                true,
                false,
                "promote",
                "Medium risk: replay check required, claim can be promoted",
            ),
            "high" => (
                true,
                true,
                if refuted {
                    "quarantine"
                } else if has_evidence {
                    "promote"
                } else {
                    "defer"
                },
                if refuted {
                    "High risk: refutation attempted, claim quarantined"
                } else if has_evidence {
                    "High risk: falsification passed with evidence, claim promoted"
                } else {
                    "High risk: no evidence provided, claim deferred"
                },
            ),
            "critical" => (
                true,
                true,
                if refuted {
                    "quarantine"
                } else if has_evidence {
                    "promote_pending_replay"
                } else {
                    "defer"
                },
                if refuted {
                    "Critical risk: refutation found, claim quarantined"
                } else if has_evidence {
                    "Critical risk: evidence provided, replay verification required before promotion"
                } else {
                    "Critical risk: requires evidence AND refutation, claim deferred"
                },
            ),
            _ => {
                return Err(ErrorData::invalid_params(
                    format!("Invalid risk_class '{risk}'. Must be: low, medium, high, or critical"),
                    None,
                ))
            }
        };

        json_to_string(&serde_json::json!({
            "ok": true,
            "claim": claim,
            "risk_class": risk,
            "required_checks": {
                "cheap_checks": true,
                "replay_checks": needs_replay,
                "falsification_checks": needs_falsification,
            },
            "has_evidence": has_evidence,
            "refutation_attempted": refuted,
            "disposition": disposition,
            "rationale": rationale,
            "can_promote": disposition == "promote",
        }))
    }

    // ─── Search receipt tools (GAP #6-7) ────────────────────────────

    #[tool(
        description = "Load a durable search receipt by receipt/request ID. Returns the stored receipt with evaluation time, retrieval family, result IDs, and digests.",
        annotations(read_only_hint = true)
    )]
    fn sm_get_search_receipt(
        &self,
        Parameters(GetSearchReceiptParams { receipt_id }): Parameters<GetSearchReceiptParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.get_search_receipt(&receipt_id))
        });
        match result {
            Ok(Some(receipt)) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": {
                    "receipt_id": receipt.receipt_id,
                    "trace_id": receipt.trace_id,
                    "search_profile": receipt.search_profile,
                    "evaluation_time": receipt.evaluation_time,
                    "result_ids": receipt.result_ids,
                    "query_embedding_digest": receipt.query_embedding_digest,
                    "query_text_digest": receipt.query_text_digest,
                    "query_input_digest": receipt.query_input_digest,
                    "filter_digest": receipt.filter_digest,
                    "redaction_state": receipt.redaction_state,
                    "approximate": receipt.approximate,
                    "attempt_family_id": receipt.attempt_family_id,
                    "budget_id": receipt.budget_id,
                },
            })),
            Ok(None) => json_to_string(&serde_json::json!({
                "ok": true,
                "found": false,
                "receipt_id": receipt_id,
                "message": "No receipt found with that ID",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("get_search_receipt error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Replay a durable search receipt with caller-supplied query text and filters. Compares original results to replay results, reporting matches, missing IDs, and added IDs.",
        annotations(read_only_hint = true)
    )]
    fn sm_replay_search_receipt(
        &self,
        Parameters(ReplaySearchReceiptParams {
            receipt_id,
            query,
            top_k,
            namespaces,
        }): Parameters<ReplaySearchReceiptParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let k = top_k.map(|v| v as usize);
        let ns_slice: Option<Vec<&str>> = namespaces
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.replay_search_receipt(
                &receipt_id,
                &query,
                k,
                ns_slice.as_deref(),
                None,
            ))
        });
        match result {
            Ok(report) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt_id": report.receipt_id,
                "replay_receipt_id": report.replay_receipt_id,
                "query_embedding_digest_matches": report.query_embedding_digest_matches,
                "result_ids_match": report.result_ids_match,
                "missing_result_ids": report.missing_result_ids,
                "added_result_ids": report.added_result_ids,
                "original_receipt": {
                    "receipt_id": report.original_receipt.receipt_id,
                    "result_ids": report.original_receipt.result_ids,
                    "search_profile": report.original_receipt.search_profile,
                    "evaluation_time": report.original_receipt.evaluation_time,
                },
                "replay_receipt": {
                    "receipt_id": report.replay_receipt.receipt_id,
                    "result_ids": report.replay_receipt.result_ids,
                    "search_profile": report.replay_receipt.search_profile,
                    "evaluation_time": report.replay_receipt.evaluation_time,
                },
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("replay_search_receipt error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Replay a durable search receipt using query text and filters retained by explicit opt-in. Returns the replay comparison report.",
        annotations(read_only_hint = true)
    )]
    fn sm_replay_search(
        &self,
        Parameters(ReplayStoredSearchParams { receipt_id }): Parameters<ReplayStoredSearchParams>,
    ) -> Result<String, ErrorData> {
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(
                self.bridge
                    .store
                    .replay_search_from_stored_inputs(&receipt_id),
            )
        });
        match result {
            Ok(report) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt_id": report.receipt_id,
                "replay_receipt_id": report.replay_receipt_id,
                "query_embedding_digest_matches": report.query_embedding_digest_matches,
                "result_ids_match": report.result_ids_match,
                "missing_result_ids": report.missing_result_ids,
                "added_result_ids": report.added_result_ids,
                "original_result_ids": report.original_receipt.result_ids,
                "replay_result_ids": report.replay_receipt.result_ids,
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("replay_search error: {e}"),
                None,
            )),
        }
    }

    // ─── Reconcile tool (GAP #8) ────────────────────────────────────

    #[tool(
        description = "Reconcile detected integrity issues. Actions: report_only (just check), rebuild_fts (rebuild FTS indexes), re_embed (re-embed all content). Returns an integrity report after the action.",
        annotations(idempotent_hint = true)
    )]
    fn sm_reconcile(
        &self,
        Parameters(ReconcileParams { action }): Parameters<ReconcileParams>,
    ) -> Result<String, ErrorData> {
        let action_enum = match action.to_lowercase().as_str() {
            "report_only" | "report-only" => semantic_memory::ReconcileAction::ReportOnly,
            "rebuild_fts" | "rebuild-fts" => semantic_memory::ReconcileAction::RebuildFts,
            "re_embed" | "re-embed" | "reembed" => semantic_memory::ReconcileAction::ReEmbed,
            _ => {
                return Err(ErrorData::invalid_params(
                    format!("action must be 'report_only', 'rebuild_fts', or 're_embed', got '{action}'"),
                    None,
                ));
            }
        };
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.reconcile(action_enum))
        });
        match result {
            Ok(report) => json_to_string(&serde_json::json!({
                "ok": report.ok,
                "schema_version": report.schema_version,
                "fact_count": report.fact_count,
                "chunk_count": report.chunk_count,
                "message_count": report.message_count,
                "facts_missing_embeddings": report.facts_missing_embeddings,
                "chunks_missing_embeddings": report.chunks_missing_embeddings,
                "issues": report.issues,
                "issue_count": report.issues.len(),
                "action": action,
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("reconcile error: {e}"),
                None,
            )),
        }
    }

    // ─── Maintenance tools (GAP #9) ─────────────────────────────────

    #[tool(
        description = "Vacuum the database to reclaim space after deletions. This is a maintenance operation that may take a moment.",
        annotations(idempotent_hint = true)
    )]
    fn sm_vacuum(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| Handle::current().block_on(store.vacuum()));
        match result {
            Ok(()) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": mcp_receipt("sm_vacuum"),
                "message": "Database vacuumed successfully",
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("vacuum error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Re-embed all facts, chunks, messages, and episodes. Call after changing embedding models. Returns the count of items re-embedded.",
        annotations(idempotent_hint = true)
    )]
    fn sm_reembed_all(&self) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.reembed_all()));
        match result {
            Ok(count) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipt": mcp_receipt("sm_reembed_all"),
                "reembedded_count": count,
                "message": format!("Re-embedded {count} items"),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("reembed_all error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Check if embeddings need re-generation after a model change. Returns true if the embedding model or dimensions have changed since the last embedding was stored.",
        annotations(read_only_hint = true)
    )]
    fn sm_embeddings_are_dirty(
        &self,
        Parameters(_params): Parameters<EmbeddingsAreDirtyParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.embeddings_are_dirty())
        });
        match result {
            Ok(dirty) => json_to_string(&serde_json::json!({
                "ok": true,
                "dirty": dirty,
                "message": if dirty { "Embeddings are dirty and need re-generation. Call sm_reembed_all." } else { "Embeddings are up to date" },
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("embeddings_are_dirty error: {e}"),
                None,
            )),
        }
    }

    // ─── Projection query tools (GAP #10) ───────────────────────────

    #[tool(
        description = "Query imported claim projection rows. Filters by scope, text, valid-time, and claim state. Returns claim version rows with full provenance.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_claim_versions(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_claim_versions(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or_else(|_| serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_claim_versions error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported relation projection rows. Filters by scope, text, valid-time, and subject entity. Returns relation version rows with full provenance.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_relation_versions(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_relation_versions(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_relation_versions error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported episode projection rows. Filters by scope and text. Returns episode rows with cause/effect and outcome data.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_episodes(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result =
            tokio::task::block_in_place(|| Handle::current().block_on(store.query_episodes(query)));
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_episodes error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported entity-alias rows. Filters by scope, canonical entity, and text. Returns alias rows with merge and review state.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_entity_aliases(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_entity_aliases(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_entity_aliases error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Query imported evidence-reference rows. Filters by scope, claim, and claim version. Returns evidence reference rows with fetch handles and source authority.",
        annotations(read_only_hint = true)
    )]
    fn sm_query_evidence_refs(
        &self,
        Parameters(params): Parameters<ProjectionQueryParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let query = build_projection_query(params);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.query_evidence_refs(query))
        });
        match result {
            Ok(rows) => json_to_string(&serde_json::json!({
                "ok": true,
                "results": serde_json::to_value(&rows).unwrap_or(serde_json::json!([])),
                "count": rows.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("query_evidence_refs error: {e}"),
                None,
            )),
        }
    }

    // ─── Import tools (GAP #11) ─────────────────────────────────────

    #[tool(
        description = "Import a projection envelope atomically. All records are committed in a single transaction or the entire import is rolled back. Pass the envelope as a JSON string.",
        annotations(idempotent_hint = true)
    )]
    #[allow(deprecated)]
    fn sm_import_envelope(
        &self,
        Parameters(ImportEnvelopeParams { envelope_json }): Parameters<ImportEnvelopeParams>,
    ) -> Result<String, ErrorData> {
        let envelope: semantic_memory::projection_import::ImportEnvelope =
            serde_json::from_str(&envelope_json).map_err(|e| {
                ErrorData::invalid_params(format!("Failed to parse envelope JSON: {e}"), None)
            })?;
        envelope.validate().map_err(|e| {
            ErrorData::invalid_params(format!("Envelope validation failed: {e}"), None)
        })?;
        let store = &self.bridge.store;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.import_envelope(&envelope))
        });
        match result {
            Ok(receipt) => json_to_string(&serde_json::json!({
                "ok": true,
                "envelope_id": receipt.envelope_id,
                "was_duplicate": receipt.was_duplicate,
                "imported_count": receipt.record_count,
                "receipt_id": receipt.envelope_id,
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("import_envelope error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "Check whether an envelope has already been imported. Returns import receipts for the given envelope ID.",
        annotations(read_only_hint = true)
    )]
    #[allow(deprecated)]
    fn sm_import_status(
        &self,
        Parameters(ImportStatusParams { envelope_id }): Parameters<ImportStatusParams>,
    ) -> Result<String, ErrorData> {
        use semantic_memory::projection_import::EnvelopeId;
        let store = &self.bridge.store;
        let env_id = EnvelopeId::new(&envelope_id);
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.import_status(&env_id))
        });
        match result {
            Ok(receipts) => json_to_string(&serde_json::json!({
                "ok": true,
                "envelope_id": envelope_id,
                "receipts": serde_json::to_value(&receipts).unwrap_or(serde_json::json!([])),
                "count": receipts.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("import_status error: {e}"),
                None,
            )),
        }
    }

    #[tool(
        description = "List recent imports, optionally filtered by namespace. Returns import receipt records.",
        annotations(read_only_hint = true)
    )]
    #[allow(deprecated)]
    fn sm_list_imports(
        &self,
        Parameters(ListImportsParams { namespace, limit }): Parameters<ListImportsParams>,
    ) -> Result<String, ErrorData> {
        let store = &self.bridge.store;
        let lim = limit.unwrap_or(20) as usize;
        let result = tokio::task::block_in_place(|| {
            Handle::current().block_on(store.list_imports(namespace.as_deref(), lim))
        });
        match result {
            Ok(receipts) => json_to_string(&serde_json::json!({
                "ok": true,
                "receipts": serde_json::to_value(&receipts).unwrap_or(serde_json::json!([])),
                "count": receipts.len(),
            })),
            Err(e) => Err(ErrorData::internal_error(
                format!("list_imports error: {e}"),
                None,
            )),
        }
    }
    // ── LLM output parser tools ─────────────────────────────────────────

    #[cfg(feature = "llm-parser")]
    #[tool(
        description = "Parse JSON from raw LLM output. Handles think blocks, markdown fences, trailing text, and common JSON errors without an additional LLM call. Returns the extracted JSON as a string.",
        annotations(read_only_hint = true)
    )]
    fn sm_parse_json(
        &self,
        Parameters(ParseJsonParams { raw_output }): Parameters<ParseJsonParams>,
    ) -> Result<String, ErrorData> {
        match llm_output_parser::parse_json::<serde_json::Value>(&raw_output) {
            Ok(value) => {
                Ok(serde_json::to_string_pretty(&value).unwrap_or_else(|_| "null".to_string()))
            }
            Err(e) => Ok(json_to_string(&serde_json::json!({
                "ok": false,
                "error": e.to_string(),
                "input_preview": &raw_output.chars().take(200).collect::<String>(),
            }))?),
        }
    }

    #[cfg(feature = "llm-parser")]
    #[tool(
        description = "Parse JSON from raw LLM output as an untyped serde_json::Value. Useful when the expected schema is unknown.",
        annotations(read_only_hint = true)
    )]
    fn sm_parse_json_value(
        &self,
        Parameters(ParseJsonValueParams { raw_output }): Parameters<ParseJsonValueParams>,
    ) -> Result<String, ErrorData> {
        match llm_output_parser::parse_json_value(&raw_output) {
            Ok(value) => {
                Ok(serde_json::to_string_pretty(&value).unwrap_or_else(|_| "null".to_string()))
            }
            Err(e) => Ok(json_to_string(&serde_json::json!({
                "ok": false,
                "error": e.to_string(),
            }))?),
        }
    }

    #[cfg(feature = "llm-parser")]
    #[tool(
        description = "Strip </think> blocks from text. Removes chain-of-thought reasoning that some models emit. Returns cleaned text.",
        annotations(read_only_hint = true)
    )]
    fn sm_strip_think_tags(
        &self,
        Parameters(StripThinkTagsParams { text }): Parameters<StripThinkTagsParams>,
    ) -> Result<String, ErrorData> {
        Ok(llm_output_parser::strip_think_tags(&text))
    }

    #[cfg(feature = "llm-parser")]
    #[tool(
        description = "Attempt to repair common LLM JSON errors: trailing commas, unquoted keys, single quotes, missing brackets. Returns the repaired JSON string or an error.",
        annotations(read_only_hint = true)
    )]
    fn sm_repair_json(
        &self,
        Parameters(RepairJsonParams { json_string }): Parameters<RepairJsonParams>,
    ) -> Result<String, ErrorData> {
        match llm_output_parser::try_repair_json(&json_string) {
            Some(repaired) => Ok(repaired),
            None => Ok(json_to_string(&serde_json::json!({
                "ok": false,
                "error": "Could not repair JSON. The input may not be valid JSON even after common fixes.",
            }))?),
        }
    }

    #[cfg(feature = "llm-parser")]
    #[tool(
        description = "Parse a string list from raw LLM output. Handles markdown bullet lists, numbered lists, comma-separated values, and JSON arrays. Returns a JSON array of cleaned strings.",
        annotations(read_only_hint = true)
    )]
    fn sm_parse_string_list(
        &self,
        Parameters(ParseStringListParams { raw_output }): Parameters<ParseStringListParams>,
    ) -> Result<String, ErrorData> {
        match llm_output_parser::parse_string_list(&raw_output) {
            Ok(list) => {
                Ok(serde_json::to_string_pretty(&list).unwrap_or_else(|_| "[]".to_string()))
            }
            Err(e) => Ok(json_to_string(&serde_json::json!({
                "ok": false,
                "error": e.to_string(),
            }))?),
        }
    }

    #[cfg(feature = "llm-parser")]
    #[tool(
        description = "Parse a choice from raw LLM output given a list of valid options. Handles extra text, casing differences, and partial matches. Returns the matched option or an error.",
        annotations(read_only_hint = true)
    )]
    fn sm_parse_choice(
        &self,
        Parameters(ParseChoiceParams {
            raw_output,
            options,
        }): Parameters<ParseChoiceParams>,
    ) -> Result<String, ErrorData> {
        let opt_refs: Vec<&str> = options.iter().map(|s| s.as_str()).collect();
        match llm_output_parser::parse_choice(&raw_output, &opt_refs) {
            Ok(choice) => Ok(choice.to_string()),
            Err(e) => Ok(json_to_string(&serde_json::json!({
                "ok": false,
                "error": e.to_string(),
                "options": options,
            }))?),
        }
    }

    #[cfg(feature = "llm-parser")]
    #[tool(
        description = "Parse a number from raw LLM output. Handles text like 'The answer is 42' or 'Score: 0.85'. Returns the number as a string.",
        annotations(read_only_hint = true)
    )]
    fn sm_parse_number(
        &self,
        Parameters(ParseNumberParams { raw_output }): Parameters<ParseNumberParams>,
    ) -> Result<String, ErrorData> {
        match llm_output_parser::parse_number::<f64>(&raw_output) {
            Ok(n) => Ok(n.to_string()),
            Err(e) => Ok(json_to_string(&serde_json::json!({
                "ok": false,
                "error": e.to_string(),
            }))?),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphPathOutcome {
    Found(Vec<String>),
    NoPathWithinCompleteSearch,
    BudgetExceeded,
    InvalidEndpoint(String),
}

/// Adapter-owned BFS whose terminal states are explicit. The underlying graph
/// API returns `None` for both exhausted and bounded searches, so callers must
/// not consume it directly when correctness depends on that distinction.
fn typed_graph_path(
    graph: &dyn semantic_memory::GraphView,
    from: &str,
    to: &str,
    max_depth: usize,
) -> Result<GraphPathOutcome, semantic_memory::MemoryError> {
    use semantic_memory::GraphDirection;
    let from_edges = graph.neighbors(from, GraphDirection::Both, 1)?;
    if from_edges.is_empty() {
        return Ok(GraphPathOutcome::InvalidEndpoint(from.to_string()));
    }
    let to_edges = graph.neighbors(to, GraphDirection::Both, 1)?;
    if to_edges.is_empty() {
        return Ok(GraphPathOutcome::InvalidEndpoint(to.to_string()));
    }
    if from == to {
        return Ok(GraphPathOutcome::Found(vec![from.to_string()]));
    }

    let mut visited = HashSet::from([from.to_string()]);
    let mut parents = HashMap::<String, String>::new();
    let mut queue = VecDeque::from([(from.to_string(), 0usize)]);
    let mut hit_depth_budget = false;
    while let Some((node, depth)) = queue.pop_front() {
        let edges = graph.neighbors(&node, GraphDirection::Both, 1)?;
        for edge in edges {
            let next = if edge.source == node {
                edge.target
            } else {
                edge.source
            };
            if visited.contains(&next) {
                continue;
            }
            if depth >= max_depth {
                hit_depth_budget = true;
                continue;
            }
            visited.insert(next.clone());
            parents.insert(next.clone(), node.clone());
            if next == to {
                let mut path = vec![to.to_string()];
                let mut cursor = to.to_string();
                while let Some(parent) = parents.get(&cursor) {
                    path.push(parent.clone());
                    if parent == from {
                        break;
                    }
                    cursor = parent.clone();
                }
                path.reverse();
                return Ok(GraphPathOutcome::Found(path));
            }
            if visited.len() >= 500 {
                return Ok(GraphPathOutcome::BudgetExceeded);
            }
            queue.push_back((next, depth + 1));
        }
    }
    Ok(if hit_depth_budget {
        GraphPathOutcome::BudgetExceeded
    } else {
        GraphPathOutcome::NoPathWithinCompleteSearch
    })
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod correctness_contract_tests {
    use super::*;
    use crate::bridge::{BridgeConfig, EmbedderBackend};
    use semantic_memory::{GraphDirection, GraphEdge, GraphEdgeType, GraphView, MemoryError};

    struct TestGraph(Vec<(&'static str, &'static str)>);

    fn add_fact_params(content: &str, idempotency_key: Option<&str>) -> AddFactParams {
        AddFactParams {
            content: content.into(),
            namespace: "authority-test".into(),
            source: Some("tests/authority-source.md".into()),
            extract_entities: Some(false),
            memory_kind: Some("durable_fact".into()),
            sensitivity: Some("internal".into()),
            evidence_refs: Some(vec!["evidence:authority-test".into()]),
            idempotency_key: idempotency_key.map(str::to_string),
        }
    }

    fn invoke_add_fact(
        runtime: &tokio::runtime::Runtime,
        server: &SemanticMemoryServer,
        params: AddFactParams,
    ) -> Result<String, ErrorData> {
        runtime.block_on(async {
            tokio::task::block_in_place(|| server.sm_add_fact(Parameters(params)))
        })
    }

    fn invoke_record_outcome(
        runtime: &tokio::runtime::Runtime,
        server: &SemanticMemoryServer,
        query: &str,
        outcome: &str,
    ) -> serde_json::Value {
        let body = runtime
            .block_on(async {
                tokio::task::block_in_place(|| {
                    server.sm_record_outcome(Parameters(RecordOutcomeParams {
                        query: query.to_string(),
                        outcome: outcome.to_string(),
                    }))
                })
            })
            .expect("record routing outcome");
        serde_json::from_str(&body).expect("routing outcome JSON")
    }

    #[test]
    fn routing_feedback_batch_persists_and_survives_restart() {
        use semantic_memory::rl_routing::{is_trained, route_with_policy};
        use semantic_memory::routing::{QueryProfile, RetrievalRouter};

        let dir = tempfile::tempdir().unwrap();
        let memory_dir = dir.path().to_path_buf();
        let make_config = || BridgeConfig {
            memory_dir: memory_dir.clone(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = SemanticMemoryServer::new(MemoryBridge::open(make_config()).unwrap(), "full");
        let query = "compare rust vs python performance";

        for index in 0..ROUTING_POLICY_PERSIST_BATCH {
            let receipt = invoke_record_outcome(&runtime, &server, query, "bad");
            assert_eq!(
                receipt["policy_state"]["persisted"],
                index + 1 == ROUTING_POLICY_PERSIST_BATCH
            );
        }

        let persisted = runtime
            .block_on(server.bridge.store.load_routing_policy())
            .unwrap()
            .expect("batched routing policy persisted");
        assert_eq!(persisted.trained_examples, ROUTING_POLICY_PERSIST_BATCH);
        assert!(persisted.last_updated.is_some());
        assert!(is_trained(&persisted));

        let profile = QueryProfile::from_query(query);
        let heuristic = RetrievalRouter::default().route(&profile);
        let before_restart = route_with_policy(&persisted, &profile);
        assert_ne!(before_restart.rerank_fine, heuristic.rerank_fine);

        let policy_json = runtime
            .block_on(async { tokio::task::block_in_place(|| server.sm_get_routing_policy()) })
            .unwrap();
        let policy_json: serde_json::Value = serde_json::from_str(&policy_json).unwrap();
        assert_eq!(
            policy_json["policy"]["training_examples_count"],
            ROUTING_POLICY_PERSIST_BATCH
        );
        assert!(policy_json["policy"]["last_updated"].is_string());

        drop(server);
        let restarted =
            SemanticMemoryServer::new(MemoryBridge::open(make_config()).unwrap(), "full");
        let loaded = runtime
            .block_on(restarted.bridge.store.load_routing_policy())
            .unwrap()
            .expect("routing policy loaded after restart");
        let after_restart = route_with_policy(&loaded, &profile);
        assert_eq!(after_restart.bm25_coarse, before_restart.bm25_coarse);
        assert_eq!(after_restart.vector_medium, before_restart.vector_medium);
        assert_eq!(after_restart.rerank_fine, before_restart.rerank_fine);
        assert_eq!(after_restart.decoder, before_restart.decoder);
    }

    fn governed_decision_server(
        scopes: semantic_memory::AuthorityScopesV1,
    ) -> (SemanticMemoryServer, tokio::runtime::Runtime, String) {
        use semantic_memory::{
            AuthorityPermit, ElevationRequirementV1, NamespaceScopeV1, OriginAuthorityLabelV1,
            OriginClassV1, OriginRiskV1, RevocationStatusV1, SubjectPrincipalV1,
        };

        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let origin = OriginAuthorityLabelV1::new(
            OriginClassV1::UserStatement,
            "principal:writer",
            "governed-decision-test",
            "blake3:governed-decision-source",
            OriginRiskV1::Low,
            scopes,
            ElevationRequirementV1::ExplicitOperatorApproval,
            None,
            RevocationStatusV1::Active,
            vec!["principal:alice".into(), "team:medical".into()],
        )
        .unwrap()
        .with_subject_principal(SubjectPrincipalV1::new("principal:patient").unwrap())
        .with_resource_scope(NamespaceScopeV1::exact("medical"));
        let fact_id = runtime
            .block_on(
                bridge.store.authority().append(
                    AuthorityPermit::with_evidence(
                        "principal:writer",
                        "governed-decision-test",
                        AuthorityPermit::APPEND_CAPABILITY,
                        vec!["evidence:test".into()],
                    )
                    .with_origin(origin),
                    "governed-decision".into(),
                    "medical".into(),
                    "CONFIDENTIAL_MEMORY_SENTINEL".into(),
                    None,
                ),
            )
            .unwrap()
            .affected_ids[0]
            .clone();
        (SemanticMemoryServer::new(bridge, "full"), runtime, fact_id)
    }

    fn decision_params(fact_id: &str) -> GovernedDecisionParams {
        GovernedDecisionParams {
            fact_id: fact_id.into(),
            caller: "principal:alice".into(),
            subject: "principal:patient".into(),
            audiences: vec!["principal:alice".into(), "team:medical".into()],
            scope: GovernedNamespaceScopeParams {
                namespace: "medical".into(),
                domain: None,
                workspace_id: None,
                repo_id: None,
            },
            delegation_or_elevation: None,
        }
    }

    fn invoke_decision(
        runtime: &tokio::runtime::Runtime,
        server: &SemanticMemoryServer,
        purpose: semantic_memory::GovernedAccessPurposeV1,
        params: GovernedDecisionParams,
    ) -> serde_json::Value {
        let body = runtime
            .block_on(async {
                tokio::task::block_in_place(|| match purpose {
                    semantic_memory::GovernedAccessPurposeV1::Assertion => {
                        server.sm_decide_assertion_authority(Parameters(params))
                    }
                    semantic_memory::GovernedAccessPurposeV1::Action => {
                        server.sm_decide_action_authority(Parameters(params))
                    }
                    _ => unreachable!(),
                })
            })
            .unwrap();
        serde_json::from_str(&body).unwrap()
    }

    #[test]
    fn recall_authority_does_not_imply_assertion_or_action_authority_and_denials_omit_content() {
        use semantic_memory::{AuthorityScopeV1, AuthorityScopesV1, GovernedAccessPurposeV1};
        let (server, runtime, fact_id) = governed_decision_server(AuthorityScopesV1 {
            recall: AuthorityScopeV1::Audience,
            assertion: AuthorityScopeV1::Denied,
            action: AuthorityScopeV1::Denied,
        });

        for purpose in [
            GovernedAccessPurposeV1::Assertion,
            GovernedAccessPurposeV1::Action,
        ] {
            let receipt = invoke_decision(&runtime, &server, purpose, decision_params(&fact_id));
            assert_eq!(receipt["schema_version"], "origin_authority_decision_v1");
            assert_eq!(receipt["allowed"], false);
            assert_eq!(
                receipt["purpose"],
                match purpose {
                    GovernedAccessPurposeV1::Assertion => "assertion",
                    GovernedAccessPurposeV1::Action => "action",
                    _ => unreachable!(),
                }
            );
            assert!(receipt["reasons"]
                .as_array()
                .unwrap()
                .iter()
                .any(|reason| { reason == "scope_or_principal_denied" }));
            let serialized = receipt.to_string();
            assert!(!serialized.contains("CONFIDENTIAL_MEMORY_SENTINEL"));
            for forbidden in ["fact", "content", "origin", "memory"] {
                assert!(receipt.get(forbidden).is_none(), "leaked field {forbidden}");
            }
        }
    }

    #[test]
    fn assertion_decision_honors_delegation_expiry_audience_and_namespace() {
        use semantic_memory::{AuthorityScopeV1, AuthorityScopesV1, GovernedAccessPurposeV1};
        let (server, runtime, fact_id) = governed_decision_server(AuthorityScopesV1 {
            recall: AuthorityScopeV1::Audience,
            assertion: AuthorityScopeV1::Audience,
            action: AuthorityScopeV1::Audience,
        });
        let mut params = decision_params(&fact_id);
        params.delegation_or_elevation = Some(GovernedLeaseParams {
            lease_id: "lease:assertion".into(),
            delegator: "principal:patient".into(),
            delegatee: "principal:alice".into(),
            purposes: vec![GovernedAccessPurposeParam::Assertion],
            scope: params.scope.clone(),
            audiences: vec!["team:medical".into()],
            expires_at: "2999-01-01T00:00:00Z".into(),
            revoked: false,
            elevation: false,
        });
        assert_eq!(
            invoke_decision(
                &runtime,
                &server,
                GovernedAccessPurposeV1::Assertion,
                params.clone()
            )["allowed"],
            true
        );

        let mut expired = params.clone();
        expired.delegation_or_elevation.as_mut().unwrap().expires_at =
            "2000-01-01T00:00:00Z".into();
        let receipt = invoke_decision(
            &runtime,
            &server,
            GovernedAccessPurposeV1::Assertion,
            expired,
        );
        assert_eq!(receipt["allowed"], false);
        assert!(receipt["reasons"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("delegation_expired")));

        let mut wrong_audience = params.clone();
        wrong_audience.audiences = vec!["team:other".into()];
        let receipt = invoke_decision(
            &runtime,
            &server,
            GovernedAccessPurposeV1::Assertion,
            wrong_audience,
        );
        assert_eq!(receipt["allowed"], false);
        assert!(receipt["reasons"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("audience_intersection_empty")));

        let mut wrong_namespace = params;
        wrong_namespace.scope.namespace = "other".into();
        let receipt = invoke_decision(
            &runtime,
            &server,
            GovernedAccessPurposeV1::Assertion,
            wrong_namespace,
        );
        assert_eq!(receipt["allowed"], false);
        assert!(receipt["reasons"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("namespace_scope_mismatch")));
    }

    #[test]
    fn action_decision_honors_live_elevation_without_converting_it_to_admin_access() {
        use semantic_memory::{AuthorityScopeV1, AuthorityScopesV1, GovernedAccessPurposeV1};
        let (server, runtime, fact_id) = governed_decision_server(AuthorityScopesV1 {
            recall: AuthorityScopeV1::Audience,
            assertion: AuthorityScopeV1::Denied,
            action: AuthorityScopeV1::Audience,
        });
        let mut params = decision_params(&fact_id);
        params.delegation_or_elevation = Some(GovernedLeaseParams {
            lease_id: "lease:elevation".into(),
            delegator: "principal:patient".into(),
            delegatee: "principal:alice".into(),
            purposes: vec![GovernedAccessPurposeParam::Admin],
            scope: params.scope.clone(),
            audiences: vec![],
            expires_at: "2999-01-01T00:00:00Z".into(),
            revoked: false,
            elevation: true,
        });
        let receipt = invoke_decision(&runtime, &server, GovernedAccessPurposeV1::Action, params);
        assert_eq!(receipt["allowed"], true);
        assert_eq!(receipt["purpose"], "action");
        assert_eq!(receipt["lease_id"], "lease:elevation");
    }
    impl GraphView for TestGraph {
        fn neighbors(
            &self,
            node: &str,
            _: GraphDirection,
            _: usize,
        ) -> Result<Vec<GraphEdge>, MemoryError> {
            Ok(self
                .0
                .iter()
                .filter(|(a, b)| *a == node || *b == node)
                .map(|(a, b)| GraphEdge {
                    source: (*a).into(),
                    target: (*b).into(),
                    edge_type: GraphEdgeType::Entity {
                        relation: "test".into(),
                    },
                    weight: 1.0,
                    metadata: None,
                })
                .collect())
        }
        fn path(&self, _: &str, _: &str, _: usize) -> Result<Option<Vec<String>>, MemoryError> {
            unreachable!()
        }
    }

    #[test]
    fn graph_path_outcomes_do_not_conflate_exhaustion_and_budget() {
        let graph = TestGraph(vec![("a", "b"), ("b", "c"), ("x", "y")]);
        assert_eq!(
            typed_graph_path(&graph, "a", "c", 2).unwrap(),
            GraphPathOutcome::Found(vec!["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(
            typed_graph_path(&graph, "a", "c", 1).unwrap(),
            GraphPathOutcome::BudgetExceeded
        );
        assert_eq!(
            typed_graph_path(&graph, "a", "x", 5).unwrap(),
            GraphPathOutcome::NoPathWithinCompleteSearch
        );
        assert_eq!(
            typed_graph_path(&graph, "missing", "x", 5).unwrap(),
            GraphPathOutcome::InvalidEndpoint("missing".into())
        );
    }

    #[test]
    fn sm_add_fact_uses_authority_append_and_preserves_output_contract() {
        let dir = tempfile::tempdir().unwrap();
        let server = SemanticMemoryServer::new(
            MemoryBridge::open(BridgeConfig {
                memory_dir: dir.path().to_path_buf(),
                embedder_backend: EmbedderBackend::Mock,
                embedding_url: String::new(),
                embedding_model: "mock".into(),
                embedding_dims: 768,
                turbo_quant_enabled: false,
                turbo_quant_bits: None,
                turbo_quant_projections: None,
            })
            .unwrap(),
            "full",
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        server
            .bridge
            .store
            .authority()
            .set_fault(Some(semantic_memory::AuthorityFaultStage::BeforeAppend));

        let error = invoke_add_fact(
            &runtime,
            &server,
            add_fact_params("must pass through authority", Some("mcp-authority-fault")),
        )
        .unwrap_err();
        assert!(error.message.contains("authority fault injected"));
        assert_eq!(
            runtime
                .block_on(server.bridge.store.stats())
                .unwrap()
                .total_facts,
            0
        );

        let body = invoke_add_fact(
            &runtime,
            &server,
            add_fact_params("must pass through authority", Some("mcp-authority-success")),
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["namespace"], "authority-test");
        assert_eq!(json["message"], "Fact added successfully");
        assert!(json["fact_id"].as_str().is_some());
        assert_eq!(json.as_object().unwrap().len(), 5);
        assert!(json["receipt"]["receipt_id"].as_str().is_some());
        assert!(json["receipt"]["recorded_at"].as_str().is_some());
        assert_eq!(json["receipt"]["tool"], "sm_add_fact");
    }

    #[test]
    fn sm_add_fact_replays_explicit_key_but_unkeyed_identical_writes_remain_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let server = SemanticMemoryServer::new(
            MemoryBridge::open(BridgeConfig {
                memory_dir: dir.path().to_path_buf(),
                embedder_backend: EmbedderBackend::Mock,
                embedding_url: String::new(),
                embedding_model: "mock".into(),
                embedding_dims: 768,
                turbo_quant_enabled: false,
                turbo_quant_bits: None,
                turbo_quant_projections: None,
            })
            .unwrap(),
            "full",
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let first = invoke_add_fact(
            &runtime,
            &server,
            add_fact_params("retry-safe", Some("caller-retry-key")),
        )
        .unwrap();
        let retry = invoke_add_fact(
            &runtime,
            &server,
            add_fact_params("retry-safe", Some("caller-retry-key")),
        )
        .unwrap();
        // Compare fact_id rather than full JSON: receipt_id and recorded_at
        // are unique per call by design, so full-string equality would fail.
        let first_fact = serde_json::from_str::<serde_json::Value>(&first).unwrap()["fact_id"]
            .as_str()
            .unwrap()
            .to_string();
        let retry_fact = serde_json::from_str::<serde_json::Value>(&retry).unwrap()["fact_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(first_fact, retry_fact);

        let unkeyed_a = invoke_add_fact(
            &runtime,
            &server,
            add_fact_params("legitimate duplicate", None),
        )
        .unwrap();
        let unkeyed_b = invoke_add_fact(
            &runtime,
            &server,
            add_fact_params("legitimate duplicate", None),
        )
        .unwrap();
        let fact_id = |body: &str| {
            serde_json::from_str::<serde_json::Value>(body).unwrap()["fact_id"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_ne!(fact_id(&unkeyed_a), fact_id(&unkeyed_b));
        assert_eq!(
            runtime
                .block_on(server.bridge.store.stats())
                .unwrap()
                .total_facts,
            3
        );
    }

    #[test]
    fn sm_delete_fact_uses_governed_forgetting_closure() {
        let dir = tempfile::tempdir().unwrap();
        let server = SemanticMemoryServer::new(
            MemoryBridge::open(BridgeConfig {
                memory_dir: dir.path().to_path_buf(),
                embedder_backend: EmbedderBackend::Mock,
                embedding_url: String::new(),
                embedding_model: "mock".into(),
                embedding_dims: 768,
                turbo_quant_enabled: false,
                turbo_quant_bits: None,
                turbo_quant_projections: None,
            })
            .unwrap(),
            "full",
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let added = invoke_add_fact(
            &runtime,
            &server,
            add_fact_params("forget through MCP", Some("mcp-forget-source")),
        )
        .unwrap();
        let fact_id = serde_json::from_str::<serde_json::Value>(&added).unwrap()["fact_id"]
            .as_str()
            .unwrap()
            .to_string();
        let response = runtime
            .block_on(async {
                tokio::task::block_in_place(|| {
                    server.sm_delete_fact(Parameters(DeleteFactParams {
                        fact_id: fact_id.clone(),
                    }))
                })
            })
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(json["forgotten"], true);
        assert_eq!(json["deleted"], false);
        assert_eq!(
            json["forgetting_receipt"]["schema_version"],
            "forgetting_closure_receipt_v1"
        );
        let raw = runtime
            .block_on(server.bridge.store.get_fact_raw_compat(&fact_id))
            .unwrap()
            .unwrap();
        assert_eq!(raw.content, "[FORGOTTEN]");

        let http_origin = semantic_memory::OriginAuthorityLabelV1::operator_system(
            "principal:semantic-memory-http",
            "caller:http-add",
        );
        let http_fact = runtime
            .block_on(
                server.bridge.store.authority().append(
                    semantic_memory::AuthorityPermit::operator_system(
                        "principal:semantic-memory-http",
                        "caller:http-add",
                        semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
                    )
                    .with_origin(http_origin),
                    "cross-adapter-forget".into(),
                    "authority-test".into(),
                    "cross adapter fact".into(),
                    None,
                ),
            )
            .unwrap()
            .affected_ids[0]
            .clone();
        let response = runtime
            .block_on(async {
                tokio::task::block_in_place(|| {
                    server.sm_delete_fact(Parameters(DeleteFactParams {
                        fact_id: http_fact.clone(),
                    }))
                })
            })
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(json["forgotten"], true);
        assert_eq!(
            runtime
                .block_on(server.bridge.store.get_fact_raw_compat(&http_fact))
                .unwrap()
                .unwrap()
                .content,
            "[FORGOTTEN]"
        );
    }

    #[test]
    fn sm_add_fact_preserves_ephemeral_evidence_refs_admission_rule() {
        let dir = tempfile::tempdir().unwrap();
        let server = SemanticMemoryServer::new(
            MemoryBridge::open(BridgeConfig {
                memory_dir: dir.path().to_path_buf(),
                embedder_backend: EmbedderBackend::Mock,
                embedding_url: String::new(),
                embedding_model: "mock".into(),
                embedding_dims: 768,
                turbo_quant_enabled: false,
                turbo_quant_bits: None,
                turbo_quant_projections: None,
            })
            .unwrap(),
            "full",
        );
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut params = add_fact_params("unsupported inference", Some("ephemeral-source-only"));
        params.memory_kind = Some("ephemeral_inference".into());
        params.evidence_refs = None;

        let error = invoke_add_fact(&runtime, &server, params).unwrap_err();
        assert!(error.message.contains("requires evidence_refs"));
        assert_eq!(
            runtime
                .block_on(server.bridge.store.stats())
                .unwrap()
                .total_facts,
            0
        );
    }

    #[test]
    fn sm_search_tool_returns_only_current_supersession_head() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let old = runtime
            .block_on(
                bridge
                    .store
                    .add_fact("state", "runtime channel is violet", None, None),
            )
            .unwrap();
        let new = runtime
            .block_on(
                bridge
                    .store
                    .add_fact("state", "runtime channel is saffron", None, None),
            )
            .unwrap();
        runtime
            .block_on(bridge.store.add_graph_edge(
                &format!("fact:{new}"),
                &format!("fact:{old}"),
                GraphEdgeType::Entity {
                    relation: "supersedes".into(),
                },
                1.0,
                None,
            ))
            .unwrap();
        let server = SemanticMemoryServer::new(bridge, "full");
        let body = runtime
            .block_on(async {
                tokio::task::block_in_place(|| {
                    server.sm_search(Parameters(SearchParams {
                        query: "runtime channel".into(),
                        top_k: Some(10),
                        namespaces: Some(vec!["state".into()]),
                    }))
                })
            })
            .unwrap();
        assert!(body.contains("saffron"));
        assert!(!body.contains("violet"));
    }

    #[test]
    fn witnessed_search_hydrates_complete_honest_fact_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let fact_id = runtime
            .block_on(bridge.store.add_fact(
                "provenance-test",
                "witnessed facts must retain their source",
                Some("tests/witnessed-source.md"),
                None,
            ))
            .unwrap();
        let server = SemanticMemoryServer::new(bridge, "full");
        let body = runtime
            .block_on(async {
                tokio::task::block_in_place(|| {
                    server.sm_search_witnessed(Parameters(SearchWitnessedParams {
                        query: "witnessed facts retain source".into(),
                        top_k: Some(5),
                        namespaces: Some(vec!["provenance-test".into()]),
                        request_id: Some("witnessed-provenance-test".into()),
                        retrieval_mode: None,
                        replay_mode: None,
                    }))
                })
            })
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        let hit = json["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|hit| hit["memory_id"] == format!("fact:{fact_id}"))
            .expect("the stored fact is an injectible witnessed result");

        assert_eq!(hit["namespace"], "provenance-test");
        assert_eq!(hit["source"], "tests/witnessed-source.md");
        assert_eq!(hit["trust"], "persisted_unjudged");
        assert_eq!(hit["state"], "current");
        assert_eq!(
            hit["retrieval_receipt_ref"],
            format!("receipt:{}", json["receipt_id"].as_str().unwrap())
        );
    }

    #[test]
    fn witnessed_search_omits_noninjectible_fact_without_source() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let _fact_id = runtime
            .block_on(bridge.store.add_fact(
                "provenance-test",
                "source-less facts cannot be injected autonomously",
                None,
                None,
            ))
            .unwrap();
        let server = SemanticMemoryServer::new(bridge, "full");
        let body = runtime
            .block_on(async {
                tokio::task::block_in_place(|| {
                    server.sm_search_witnessed(Parameters(SearchWitnessedParams {
                        query: "source-less facts autonomous injection".into(),
                        top_k: Some(5),
                        namespaces: Some(vec!["provenance-test".into()]),
                        request_id: Some("witnessed-source-less-test".into()),
                        retrieval_mode: None,
                        replay_mode: None,
                    }))
                })
            })
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(json["results"].as_array().unwrap().is_empty());
    }

    #[test]
    fn sm_search_witnessed_exposes_authority_state_and_vector_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let permit = semantic_memory::AuthorityPermit::operator_system(
            "witness-principal",
            "witness-caller",
            semantic_memory::AuthorityPermit::APPEND_CAPABILITY,
        );
        let receipt = runtime
            .block_on(bridge.store.authority().append(
                permit,
                "witnessed-state-1".into(),
                "stateful".into(),
                "governed witnessed state should expose".into(),
                Some("tests/witnessed-state.md".into()),
            ))
            .unwrap();

        let server = SemanticMemoryServer::new(bridge, "lean");
        let body = runtime
            .block_on(async {
                tokio::task::block_in_place(|| {
                    server.sm_search_witnessed(Parameters(SearchWitnessedParams {
                        query: "governed witnessed state should expose".into(),
                        top_k: Some(5),
                        namespaces: Some(vec!["stateful".into()]),
                        request_id: Some("witnessed-state-request".into()),
                        retrieval_mode: None,
                        replay_mode: None,
                    }))
                })
            })
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();

        assert_eq!(json["retrieval_mode"], "hybrid");
        assert!(json["authority"]["snapshot_id"].as_str().is_some());
        assert!(
            json["authority"]["retrieval_epoch"].as_u64().is_some()
                || json["authority"]["retrieval_epoch"].as_str().is_some(),
            "authoritative retrieval epoch should be surfaced"
        );
        assert!(json["current_snapshot_id"].as_str().is_some());
        assert!(
            json["retrieval_epoch"].as_u64().is_some()
                || json["retrieval_epoch"].as_str().is_some(),
            "top-level retrieval epoch should be surfaced"
        );

        assert_eq!(json["authority"]["status"], "Applied");
        assert_eq!(
            json["stage_outcomes"]["authority_snapshot"]["outcome"],
            "Applied"
        );

        let hit = json["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|hit| {
                hit.get("content")
                    .and_then(|value| value.as_str())
                    .is_some_and(|content| content.contains("governed witnessed state"))
            })
            .expect("governed witnessed result should be returned");

        assert!(hit["cosine_similarity"].as_f64().is_some());
        assert_eq!(
            hit["source"],
            serde_json::json!("tests/witnessed-state.md"),
            "witnessed path should retain source"
        );
        assert_eq!(
            json["ordered_results"][0]["result_id"],
            format!("fact:{}", receipt.affected_ids[0])
        );
    }

    #[test]
    fn sm_search_witnessed_accepts_general_retrieval_modes() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(
            bridge
                .store
                .add_fact("modes", "alpha beta gamma", Some("test"), None),
        )
        .unwrap();
        let server = SemanticMemoryServer::new(bridge, "full");
        for retrieval_mode in [
            RetrievalModeParam::Hybrid,
            RetrievalModeParam::FtsOnly,
            RetrievalModeParam::VectorOnly,
        ] {
            let body = rt
                .block_on(async {
                    server.sm_search_witnessed(Parameters(SearchWitnessedParams {
                        query: "alpha beta".into(),
                        top_k: Some(5),
                        namespaces: Some(vec!["modes".into()]),
                        request_id: None,
                        retrieval_mode: Some(retrieval_mode),
                        replay_mode: None,
                    }))
                })
                .unwrap();
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["ok"], true);
            assert!(json["receipt_id"].is_string());
            assert_eq!(
                json["retrieval_mode"],
                serde_json::to_value(retrieval_mode).unwrap()
            );
        }
    }

    #[test]
    fn witnessed_search_opt_in_enables_complete_replay() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime
            .block_on(bridge.store.add_fact(
                "stored-replay",
                "complete replay uses explicitly retained inputs",
                Some("tests/stored-replay.md"),
                None,
            ))
            .unwrap();
        let server = SemanticMemoryServer::new(bridge, "full");
        let body = runtime
            .block_on(async {
                server.sm_search_witnessed(Parameters(SearchWitnessedParams {
                    query: "complete replay explicitly retained inputs".into(),
                    top_k: Some(1),
                    namespaces: Some(vec!["stored-replay".into()]),
                    request_id: Some("mcp-stored-replay".into()),
                    retrieval_mode: None,
                    replay_mode: Some(ReplayModeParam::StoreInputs),
                }))
            })
            .unwrap();
        let witnessed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(witnessed["complete_replay_available"], true);
        assert_eq!(witnessed["stage_outcomes"]["replay"]["outcome"], "Applied");

        let replay = runtime
            .block_on(async {
                server.sm_replay_search(Parameters(ReplayStoredSearchParams {
                    receipt_id: "mcp-stored-replay".into(),
                }))
            })
            .unwrap();
        let replay: serde_json::Value = serde_json::from_str(&replay).unwrap();
        assert_eq!(replay["result_ids_match"], true);
        assert_eq!(replay["query_embedding_digest_matches"], true);
    }

    #[cfg(feature = "claim-integration")]
    fn claim_test_entries() -> Vec<claim_ledger::LedgerEntry> {
        let first = claim_ledger::LedgerEntryBuilder::new(1, None)
            .add_claim(
                "claim-1",
                "semantic-memory:fact:fact-1",
                "full",
                "verified claim",
            )
            .unwrap();
        let second = claim_ledger::LedgerEntryBuilder::new(2, Some(first.entry_digest.clone()))
            .add_support_judgment(
                "judgment-1",
                "claim-1",
                "bundle-1",
                claim_ledger::SupportState::Supported,
                "test",
            )
            .unwrap();
        vec![first, second]
    }

    #[cfg(feature = "claim-integration")]
    fn write_claim_jsonl(path: &std::path::Path, entries: &[claim_ledger::LedgerEntry]) {
        let mut contents = String::new();
        for entry in entries {
            contents.push_str(&claim_ledger::serialize_entry(entry).unwrap());
            contents.push('\n');
        }
        std::fs::write(path, contents).unwrap();
    }

    #[cfg(feature = "claim-integration")]
    #[test]
    fn claim_ledger_startup_rebuilds_from_verified_snapshot_and_tail() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("claim_ledger.jsonl");
        let entries = claim_test_entries();
        write_claim_jsonl(&legacy, &entries);
        let mut store = ClaimLedgerStore::open(legacy.clone());
        let result = store
            .compact(ClaimLedgerCompactionConfig {
                dry_run: false,
                max_entries: 0,
                max_bytes: 0,
                retain_tail_entries: 1,
                max_backups: 2,
            })
            .unwrap();
        assert_eq!(result["compacted"], true);

        let reopened = ClaimLedgerStore::open(legacy);
        assert!(reopened.trust_enabled);
        assert!(reopened.snapshot.is_some());
        assert_eq!(reopened.entries.len(), 1);
        let mut index = ClaimTrustIndex::default();
        index.load_snapshot(reopened.snapshot.as_ref().unwrap());
        index.rebuild_from_ledger_incremental(&reopened.entries);
        assert_eq!(index.trust_for_fact("fact-1"), "supported");
        assert_eq!(index.last_processed_sequence, 2);
    }

    #[cfg(feature = "claim-integration")]
    #[test]
    fn interrupted_compaction_files_leave_old_ledger_usable() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("claim_ledger.jsonl");
        let entries = claim_test_entries();
        write_claim_jsonl(&legacy, &entries);
        let interrupted = dir.path().join("claim_ledger_generations/.tmp-interrupted");
        std::fs::create_dir_all(&interrupted).unwrap();
        std::fs::write(interrupted.join("snapshot.json"), b"incomplete").unwrap();

        let reopened = ClaimLedgerStore::open(legacy);
        assert!(reopened.trust_enabled);
        assert!(reopened.snapshot.is_none());
        assert_eq!(reopened.entries.len(), entries.len());
    }

    #[cfg(feature = "claim-integration")]
    #[test]
    fn tampered_active_snapshot_disables_claim_trust() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("claim_ledger.jsonl");
        write_claim_jsonl(&legacy, &claim_test_entries());
        let mut store = ClaimLedgerStore::open(legacy.clone());
        store
            .compact(ClaimLedgerCompactionConfig {
                dry_run: false,
                max_entries: 0,
                max_bytes: 0,
                retain_tail_entries: 1,
                max_backups: 2,
            })
            .unwrap();
        let snapshot_path = store.path.parent().unwrap().join("snapshot.json");
        let mut snapshot: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&snapshot_path).unwrap()).unwrap();
        snapshot["claims"][0]["normalized_claim"] = serde_json::json!("tampered");
        std::fs::write(&snapshot_path, serde_json::to_vec(&snapshot).unwrap()).unwrap();

        let reopened = ClaimLedgerStore::open(legacy);
        assert!(!reopened.trust_enabled);
        assert!(reopened.entries.is_empty());
    }

    #[cfg(feature = "claim-integration")]
    #[test]
    fn compact_claim_ledger_defaults_to_dry_run_without_writes() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = MemoryBridge::open(BridgeConfig {
            memory_dir: dir.path().to_path_buf(),
            embedder_backend: EmbedderBackend::Mock,
            embedding_url: String::new(),
            embedding_model: "mock".into(),
            embedding_dims: 768,
            turbo_quant_enabled: false,
            turbo_quant_bits: None,
            turbo_quant_projections: None,
        })
        .unwrap();
        let server = SemanticMemoryServer::new(bridge, "full");
        assert!(server.exposes_tool("sm_compact_claim_ledger"));
        {
            let mut store = server.claim_ledger_store.lock().unwrap();
            for entry in claim_test_entries() {
                store.append(entry).unwrap();
            }
        }
        let before = std::fs::read(dir.path().join("claim_ledger.jsonl")).unwrap();
        let response = server
            .sm_compact_claim_ledger(Parameters(CompactClaimLedgerParams {
                dry_run: None,
                max_entries: Some(0),
                max_bytes: Some(0),
                retain_tail_entries: Some(1),
                max_backups: Some(2),
            }))
            .unwrap();
        let response: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["dry_run"], true);
        assert_eq!(response["compacted"], false);
        assert_eq!(
            before,
            std::fs::read(dir.path().join("claim_ledger.jsonl")).unwrap()
        );
        assert!(!dir
            .path()
            .join("claim_ledger.active_compaction.json")
            .exists());
        assert!(!dir.path().join("claim_ledger_generations").exists());
    }
}

/// Build path segments with edge evidence for each hop in a path.
/// SM-AUD-011: Include edge type, weight, and metadata for each hop.
fn build_path_segments(
    store: &semantic_memory::MemoryStore,
    path: &[String],
) -> Vec<serde_json::Value> {
    let mut segments = Vec::new();
    if path.len() < 2 {
        return segments;
    }

    for i in 0..path.len() - 1 {
        let from = &path[i];
        let to = &path[i + 1];

        // Get neighbors of the current node to find the edge to the next node.
        let g = store.graph_view();
        match g.neighbors(from, semantic_memory::GraphDirection::Both, 1) {
            Ok(edges) => {
                // Find the edge that connects from -> to.
                let connecting = edges.iter().find(|e| {
                    (e.source == *from && e.target == *to) || (e.source == *to && e.target == *from)
                });

                if let Some(edge) = connecting {
                    let edge_type_str = match &edge.edge_type {
                        semantic_memory::GraphEdgeType::Semantic { cosine_similarity } => {
                            serde_json::json!({
                                "type": "semantic",
                                "cosine_similarity": cosine_similarity,
                            })
                        }
                        semantic_memory::GraphEdgeType::Temporal { delta_secs } => {
                            serde_json::json!({
                                "type": "temporal",
                                "delta_secs": delta_secs,
                            })
                        }
                        semantic_memory::GraphEdgeType::Causal {
                            confidence,
                            evidence_ids,
                        } => {
                            serde_json::json!({
                                "type": "causal",
                                "confidence": confidence,
                                "evidence_ids": evidence_ids,
                            })
                        }
                        semantic_memory::GraphEdgeType::Entity { relation } => {
                            serde_json::json!({
                                "type": "entity",
                                "relation": relation,
                            })
                        }
                    };

                    segments.push(serde_json::json!({
                        "source": from,
                        "target": to,
                        "edge_type": edge_type_str,
                        "weight": edge.weight,
                        "metadata": edge.metadata,
                    }));
                } else {
                    // No edge found between consecutive path nodes — shouldn't
                    // happen but handle gracefully.
                    segments.push(serde_json::json!({
                        "source": from,
                        "target": to,
                        "edge_type": null,
                        "weight": null,
                        "metadata": null,
                    }));
                }
            }
            Err(_) => {
                segments.push(serde_json::json!({
                    "source": from,
                    "target": to,
                    "edge_type": null,
                    "weight": null,
                    "metadata": null,
                }));
            }
        }
    }

    segments
}

#[tool_handler(
    router = self.tool_router,
    name = "semantic-memory-mcp",
    version = "0.3.1",
    instructions = "Persistent local semantic memory with hybrid search, graph reasoning, and conversation persistence. ALWAYS search first before asking the user for context. Use sm_decide_assertion_authority or sm_decide_action_authority for content-free, fixed-purpose authority decisions; recall authority never implies either purpose. In the full operator profile, use sm_search_with_routing for complex/multi-hop queries, sm_get_fact to hydrate IDs returned by graph tools, sm_supersede_fact (not delete) for stale corrections, and sm_add_graph_edge after adding facts to connect them. Read tools are safe; write tools (add/delete/supersede) should be user-approved. Search auto-filters superseded facts unless querying for history."
)]
impl ServerHandler for SemanticMemoryServer {}
