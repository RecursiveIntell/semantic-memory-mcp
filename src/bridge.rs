//! Bridge between the MCP server and semantic-memory MemoryStore.
//!
//! semantic-memory's MemoryStore is opened synchronously and search methods
//! are async (tokio). This bridge wraps the store and uses the current
//! tokio runtime handle for async calls (no separate runtime).

#[cfg(feature = "candle-embedder")]
use semantic_memory::embedder::CandleEmbedder;
use semantic_memory::embedder::{Embedder, MockEmbedder, OllamaEmbedder};
use semantic_memory::{EmbeddingConfig, MemoryConfig, MemoryStore, SearchConfig};
use std::path::PathBuf;
use tokio::runtime::Handle;

/// Which embedding backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedderBackend {
    /// In-process Candle embedder (pure-Rust, CPU-only, no Ollama required).
    /// Downloads the model from HuggingFace Hub on first use.
    #[cfg_attr(feature = "candle-embedder", default)]
    Candle,
    /// External Ollama server (requires `ollama serve` running).
    #[cfg_attr(not(feature = "candle-embedder"), default)]
    Ollama,
    /// Mock embedder for testing (deterministic hash-based, no real embeddings).
    Mock,
}

impl std::str::FromStr for EmbedderBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "candle" => Ok(EmbedderBackend::Candle),
            "ollama" => Ok(EmbedderBackend::Ollama),
            "mock" => Ok(EmbedderBackend::Mock),
            other => Err(format!(
                "unknown embedder '{other}', expected: candle, ollama, or mock"
            )),
        }
    }
}

#[derive(Clone)]
pub struct MemoryBridge {
    pub store: MemoryStore,
    pub memory_dir: PathBuf,
}

#[derive(Clone)]
pub struct BridgeConfig {
    /// MCP-005: renamed from db_path to memory_dir — this is a directory,
    /// not a SQLite file path. semantic-memory creates base_dir/memory.db
    /// inside it.
    pub memory_dir: PathBuf,
    pub embedder_backend: EmbedderBackend,
    /// Ollama URL — only used when backend is Ollama.
    pub embedding_url: String,
    /// Embedding model name — used by both Candle (as HF model ID) and Ollama.
    pub embedding_model: String,
    pub embedding_dims: usize,
    /// Enable TurboQuant compressed vector candidate backend.
    pub turbo_quant_enabled: bool,
    /// TurboQuant polar angle bits (default: 8).
    pub turbo_quant_bits: Option<u8>,
    /// TurboQuant QJL projection count (default: 16).
    pub turbo_quant_projections: Option<usize>,
}

impl BridgeConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn from_args(
        memory_dir: &str,
        embedder_backend: EmbedderBackend,
        embedding_url: Option<&str>,
        embedding_model: Option<&str>,
        embedding_dims: Option<usize>,
        turbo_quant_enabled: bool,
        turbo_quant_bits: Option<u8>,
        turbo_quant_projections: Option<usize>,
    ) -> Self {
        Self {
            memory_dir: PathBuf::from(memory_dir),
            embedder_backend,
            embedding_url: embedding_url
                .unwrap_or("http://localhost:11434")
                .to_string(),
            embedding_model: embedding_model.unwrap_or("nomic-embed-text").to_string(),
            embedding_dims: embedding_dims.unwrap_or(768),
            turbo_quant_enabled,
            turbo_quant_bits,
            turbo_quant_projections,
        }
    }
}

impl MemoryBridge {
    /// Open the memory store with the given config.
    /// Store opening is synchronous — no runtime needed here.
    pub fn open(config: BridgeConfig) -> anyhow::Result<Self> {
        let embedding_config = EmbeddingConfig {
            ollama_url: config.embedding_url,
            model: config.embedding_model,
            dimensions: config.embedding_dims,
            batch_size: 32,
            timeout_secs: 60,
        };

        let embedder: Box<dyn Embedder> = match config.embedder_backend {
            EmbedderBackend::Mock => Box::new(MockEmbedder::new(config.embedding_dims)),
            EmbedderBackend::Ollama => Box::new(OllamaEmbedder::try_new(&embedding_config)?),
            EmbedderBackend::Candle => {
                #[cfg(feature = "candle-embedder")]
                {
                    // For Candle, the model field is the HuggingFace model ID.
                    // Map common Ollama model names to HF model IDs.
                    let hf_model_id = match embedding_config.model.as_str() {
                        "nomic-embed-text" | "nomic-embed-text-v1.5" => {
                            "nomic-ai/nomic-embed-text-v1.5"
                        }
                        other => other,
                    };
                    let candle_config = EmbeddingConfig {
                        model: hf_model_id.to_string(),
                        ..embedding_config.clone()
                    };
                    Box::new(CandleEmbedder::try_new(&candle_config)?)
                }
                #[cfg(not(feature = "candle-embedder"))]
                {
                    anyhow::bail!(
                        "candle embedder requested but the 'candle-embedder' feature \
                         is not enabled. Rebuild with --features candle-embedder \
                         or use --embedder ollama"
                    )
                }
            }
        };

        #[allow(unused_mut)]
        let mut search_config = SearchConfig::default();

        // TurboQuant compressed vector candidate backend
        #[cfg(feature = "full")]
        {
            if config.turbo_quant_enabled {
                use semantic_memory::DerivedVectorBackendPolicy;
                search_config.derived_vector_backend =
                    DerivedVectorBackendPolicy::TurboQuantCandidateOnly;
                if let Some(bits) = config.turbo_quant_bits {
                    search_config.turbo_quant_bits = bits;
                }
                if let Some(projs) = config.turbo_quant_projections {
                    search_config.turbo_quant_projections = projs;
                }
            }
        }

        let memory_dir = config.memory_dir.clone();
        let mem_config = MemoryConfig {
            base_dir: config.memory_dir,
            embedding: embedding_config,
            search: search_config,
            ..Default::default()
        };

        let store = MemoryStore::open_with_embedder(mem_config, embedder)?;

        Ok(Self { store, memory_dir })
    }

    /// Get the current tokio runtime handle.
    /// This must be called from within a tokio runtime context.
    pub fn handle() -> Handle {
        Handle::current()
    }
}
