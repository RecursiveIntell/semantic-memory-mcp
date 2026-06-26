//! Integration tests for knowledge-runtime orchestration tools.
//!
//! Tests the 6 new orchestration MCP tools using the mock embedder
//! (no model download, no Ollama, no network).

#![cfg(feature = "orchestration")]

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
fn server_constructs_with_orchestration() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    // This should succeed — runtime is constructed behind orchestration feature
    let _server = SemanticMemoryServer::new(bridge, "full", String::new(), String::new());
}

#[test]
fn server_constructs_with_orchestration_lean() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let _server = SemanticMemoryServer::new(bridge, "lean", String::new(), String::new());
}

#[test]
fn server_constructs_with_orchestration_standard() {
    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let _server = SemanticMemoryServer::new(bridge, "standard", String::new(), String::new());
}

// Test that the knowledge-runtime can be constructed from the adapter
#[test]
fn knowledge_runtime_constructs_from_store() {
    use knowledge_runtime::adapters::semantic_memory::SemanticMemoryAdapter;
    use knowledge_runtime::config::{EntityConfig, ProjectionConfig, QueryConfig, RuntimeConfig};
    use knowledge_runtime::{KnowledgeRuntime, Scope};

    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let adapter = SemanticMemoryAdapter::new(bridge.store.clone());
    let config = RuntimeConfig {
        default_scope: Scope::new("general"),
        query: QueryConfig::default(),
        entity: EntityConfig::default(),
        projection: ProjectionConfig::default(),
        strict_temporal: false,
        strict_scope: false,
    };
    let runtime = KnowledgeRuntime::new(config, adapter);
    assert!(runtime.is_ok(), "KnowledgeRuntime should construct");
}

// Test classification directly
#[test]
fn classify_query_returns_mode() {
    use knowledge_runtime::adapters::semantic_memory::SemanticMemoryAdapter;
    use knowledge_runtime::config::{EntityConfig, ProjectionConfig, QueryConfig, RuntimeConfig};
    use knowledge_runtime::{KnowledgeRuntime, QueryMode, Scope};

    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let adapter = SemanticMemoryAdapter::new(bridge.store.clone());
    let config = RuntimeConfig {
        default_scope: Scope::new("general"),
        query: QueryConfig::default(),
        entity: EntityConfig::default(),
        projection: ProjectionConfig::default(),
        strict_temporal: false,
        strict_scope: false,
    };
    let runtime = KnowledgeRuntime::new(config, adapter).expect("runtime");
    let result = runtime.classify("what is the architecture of the system?");
    // Should classify as semantic
    assert!(
        result.mode == QueryMode::SemanticLookup || matches!(result.mode, QueryMode::Mixed { .. })
    );
}

// Test entity lookup returns unresolved for empty registry
#[test]
fn entity_lookup_returns_unresolved_for_empty_registry() {
    use knowledge_runtime::adapters::semantic_memory::SemanticMemoryAdapter;
    use knowledge_runtime::config::{EntityConfig, ProjectionConfig, QueryConfig, RuntimeConfig};
    use knowledge_runtime::{KnowledgeRuntime, MatchQuality, Scope, ScopeKey};

    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let adapter = SemanticMemoryAdapter::new(bridge.store.clone());
    let config = RuntimeConfig {
        default_scope: Scope::new("general"),
        query: QueryConfig::default(),
        entity: EntityConfig::default(),
        projection: ProjectionConfig::default(),
        strict_temporal: false,
        strict_scope: false,
    };
    let runtime = KnowledgeRuntime::new(config, adapter).expect("runtime");
    let scope = ScopeKey::namespace_only("general");
    let result = runtime
        .entity_registry()
        .resolve("nonexistent_entity", &scope);
    assert_eq!(result.quality, MatchQuality::Unresolved);
    assert!(result.entity.is_none());
}

// Test projection health returns missing for unknown projection
#[test]
fn projection_health_returns_missing_for_unknown() {
    use knowledge_runtime::adapters::semantic_memory::SemanticMemoryAdapter;
    use knowledge_runtime::config::{EntityConfig, ProjectionConfig, QueryConfig, RuntimeConfig};
    use knowledge_runtime::{
        KnowledgeRuntime, ProjectionHealth, ProjectionId, ProjectionKind, Scope,
    };

    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let adapter = SemanticMemoryAdapter::new(bridge.store.clone());
    let config = RuntimeConfig {
        default_scope: Scope::new("general"),
        query: QueryConfig::default(),
        entity: EntityConfig::default(),
        projection: ProjectionConfig::default(),
        strict_temporal: false,
        strict_scope: false,
    };
    let runtime = KnowledgeRuntime::new(config, adapter).expect("runtime");
    let scope_key = knowledge_runtime::ScopeKey::namespace_only("test");
    let proj_id = ProjectionId::new(ProjectionKind::Entity, "test", scope_key);
    let health = runtime.projection_health(&proj_id);
    assert_eq!(health, ProjectionHealth::Missing);
}

// Test plan returns route plan with legs
#[test]
fn plan_query_returns_route_plan() {
    use knowledge_runtime::adapters::semantic_memory::SemanticMemoryAdapter;
    use knowledge_runtime::config::{EntityConfig, ProjectionConfig, QueryConfig, RuntimeConfig};
    use knowledge_runtime::{KnowledgeRuntime, Scope};

    let dir = tempfile::tempdir().unwrap();
    let bridge = open_bridge(dir.path());
    let adapter = SemanticMemoryAdapter::new(bridge.store.clone());
    let config = RuntimeConfig {
        default_scope: Scope::new("general"),
        query: QueryConfig::default(),
        entity: EntityConfig::default(),
        projection: ProjectionConfig::default(),
        strict_temporal: false,
        strict_scope: false,
    };
    let runtime = KnowledgeRuntime::new(config, adapter).expect("runtime");
    let scope = Scope::new("general");
    let plan = runtime.plan("what is rust?", Some(&scope));
    assert!(!plan.legs.is_empty(), "plan should have at least one leg");
    assert_eq!(plan.query, "what is rust?");
}
