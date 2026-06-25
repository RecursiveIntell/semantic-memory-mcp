#!/bin/bash
# ============================================================================
# RecursiveIntell Agent Stack Installer -- HERMES EDITION
# ============================================================================
# Installs and configures the complete evidence-first agent stack for Hermes:
#   - Hermes Agent (if not already installed)
#   - semantic-memory library + MCP server (from source, with Cargo)
#   - All 6 Hermes shell hooks (primer, recall, autocapture, capture-nudge,
#     dedup-guard, tool-receipts)
#   - Warm HTTP server on port 1738
#   - MCP server registration in Hermes config
#   - Hook allowlist entries
#
# For Claude Code: use the semantic-memory-claude-kit plugin instead:
#   https://github.com/RecursiveIntell/semantic-memory-claude-kit
#
# For Codex: use the AGENTS.md approach with MCP server in config
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/RecursiveIntell/semantic-memory-mcp/main/install.sh | bash
#
# Or from a local clone:
#   bash install.sh
#
# Options:
#   --skip-hermes    Skip Hermes installation (assume already installed)
#   --skip-hooks     Skip hook installation
#   --port PORT      HTTP server port (default: 1738)
#   --memory-dir DIR Memory DB directory (default: ~/.hermes/semantic-memory.db)
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[RI]${NC} $1"; }
warn() { echo -e "${YELLOW}[RI]${NC} $1"; }
err()  { echo -e "${RED}[RI]${NC} $1"; }
info() { echo -e "${BLUE}[RI]${NC} $1"; }

# Defaults
SKIP_HERMES=false
SKIP_HOOKS=false
PORT=1738
MEMORY_DIR="$HOME/.hermes/semantic-memory.db"
HERMES_HOME="$HOME/.hermes"
AGENT_HOOKS="$HERMES_HOME/agent-hooks"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-hermes) SKIP_HERMES=true; shift ;;
        --skip-hooks)  SKIP_HOOKS=true;  shift ;;
        --port)        PORT="$2"; shift 2 ;;
        --memory-dir)  MEMORY_DIR="$2"; shift 2 ;;
        *)             err "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  RecursiveIntell Evidence-First Agent Stack Installer"
echo "  semantic-memory + AiDENs + Hermes hooks"
echo "============================================================"
echo ""

# ============================================================================
# Step 1: Check prerequisites
# ============================================================================
log "Checking prerequisites..."

# Check Rust/Cargo
if ! command -v cargo &>/dev/null; then
    err "Rust/Cargo is required but not installed."
    info "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi
log "  Rust/Cargo: $(cargo --version)"

# Check Python (for Hermes)
if ! command -v python3 &>/dev/null; then
    err "Python 3 is required but not installed."
    exit 1
fi
log "  Python: $(python3 --version)"

# Check git
if ! command -v git &>/dev/null; then
    err "Git is required but not installed."
    exit 1
fi
log "  Git: $(git --version)"

# Determine install location
INSTALL_DIR="${RI_INSTALL_DIR:-$HOME/Coding/Libraries}"
mkdir -p "$INSTALL_DIR"

# ============================================================================
# Step 2: Install Hermes (if not already installed)
# ============================================================================
if [ "$SKIP_HERMES" = false ]; then
    if command -v hermes &>/dev/null; then
        log "Hermes already installed: $(hermes --version 2>&1 | head -1)"
    else
        log "Installing Hermes Agent..."
        curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-setup
        log "Hermes installed."
        
        # Add hermes to PATH if not already there
        HERMES_BIN="$HERMES_HOME/hermes-agent/venv/bin"
        if [[ ":$PATH:" != *":$HERMES_BIN:"* ]]; then
            info "Adding Hermes to PATH..."
            echo "export PATH=\"$HERMES_BIN:\$PATH\"" >> "$HOME/.bashrc"
            export PATH="$HERMES_BIN:$PATH"
        fi
        
        # Create hermes symlink if not exists
        if ! command -v hermes &>/dev/null; then
            mkdir -p "$HOME/.local/bin"
            ln -sf "$HERMES_HOME/hermes-agent/venv/bin/hermes" "$HOME/.local/bin/hermes"
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
else
    warn "Skipping Hermes installation (--skip-hermes)"
fi

# ============================================================================
# Step 3: Clone and build semantic-memory + MCP server
# ============================================================================
log "Setting up semantic-memory..."

SM_DIR="$INSTALL_DIR/semantic-memory"
MCP_DIR="$INSTALL_DIR/semantic-memory-mcp"

# Clone or update semantic-memory
if [ -d "$SM_DIR" ]; then
    log "  semantic-memory already exists at $SM_DIR"
    cd "$SM_DIR" && git pull --ff-only 2>/dev/null || warn "  Could not pull updates (may be a worktree)"
else
    log "  Cloning semantic-memory..."
    git clone https://github.com/RecursiveIntell/semantic-memory.git "$SM_DIR"
fi

# Clone or update semantic-memory-mcp
if [ -d "$MCP_DIR" ]; then
    log "  semantic-memory-mcp already exists at $MCP_DIR"
    cd "$MCP_DIR" && git pull --ff-only 2>/dev/null || warn "  Could not pull updates (may be a worktree)"
else
    log "  Cloning semantic-memory-mcp..."
    git clone https://github.com/RecursiveIntell/semantic-memory-mcp.git "$MCP_DIR"
fi

# Build semantic-memory library
log "  Building semantic-memory library..."
cd "$SM_DIR"
cargo build --release 2>&1 | tail -1

# Build and install MCP server
log "  Building semantic-memory-mcp server..."
cd "$MCP_DIR"
cargo build --release 2>&1 | tail -1

# Install MCP binary
MCP_BINARY="$MCP_DIR/target/release/semantic-memory-mcp"
if [ -f "$MCP_BINARY" ]; then
    mkdir -p "$HOME/.cargo/bin"
    cp "$MCP_BINARY" "$HOME/.cargo/bin/semantic-memory-mcp"
    log "  MCP server installed to ~/.cargo/bin/semantic-memory-mcp"
else
    err "  MCP binary not found at $MCP_BINARY"
    exit 1
fi

# Create memory directory
mkdir -p "$MEMORY_DIR"

# ============================================================================
# Step 4: Install Hermes hooks
# ============================================================================
if [ "$SKIP_HOOKS" = false ]; then
    log "Installing Hermes hooks..."
    mkdir -p "$AGENT_HOOKS"
    
    # Download hook files from the MCP server repo
    HOOKS_URL="https://raw.githubusercontent.com/RecursiveIntell/semantic-memory-mcp/main/hooks"
    
    HOOKS=(
        "sm_http_client.py"
        "sm-primer.py"
        "sm-recall.py"
        "sm-autocapture.py"
        "sm-capture-nudge.py"
        "sm-dedup-guard.py"
        "sm-tool-receipts.py"
        "sm_health_check.py"
    )
    
    for hook in "${HOOKS[@]}"; do
        if [ -f "$AGENT_HOOKS/$hook" ]; then
            log "  $hook already exists"
        else
            # Try downloading from GitHub
            if curl -sSf "$HOOKS_URL/$hook" -o "$AGENT_HOOKS/$hook" 2>/dev/null; then
                log "  Downloaded $hook"
            else
                warn "  Could not download $hook from GitHub -- will create from local copy"
                # Copy from local if available
                if [ -f "$MCP_DIR/hooks/$hook" ]; then
                    cp "$MCP_DIR/hooks/$hook" "$AGENT_HOOKS/$hook"
                    log "  Copied $hook from local repo"
                else
                    warn "  $hook not found -- some hooks may need manual installation"
                fi
            fi
        fi
        chmod +x "$AGENT_HOOKS/$hook" 2>/dev/null
    done
    
    # ============================================================================
    # Step 5: Configure Hermes
    # ============================================================================
    log "Configuring Hermes..."
    
    # Register MCP server
    if command -v hermes &>/dev/null; then
        hermes config set mcp_servers.semantic_memory "{
            \"command\": \"semantic-memory-mcp\",
            \"args\": [\"--memory-dir\", \"$MEMORY_DIR\", \"--embedder\", \"candle\", \"--embedding-model\", \"nomic-embed-text\", \"--embedding-dims\", \"768\", \"--http-port\", \"$PORT\"],
            \"enabled\": true
        }" 2>/dev/null && log "  MCP server registered" || warn "  Could not register MCP server"
        
        # Register hooks
        hermes config set hooks.on_session_start "[{\"command\":\"python3 $AGENT_HOOKS/sm-primer.py\",\"timeout\":15}]" 2>/dev/null
        hermes config set hooks.pre_llm_call "[{\"command\":\"python3 $AGENT_HOOKS/sm-recall.py\",\"timeout\":15}]" 2>/dev/null
        hermes config set hooks.post_llm_call "[{\"command\":\"python3 $AGENT_HOOKS/sm-autocapture.py\",\"timeout\":30}]" 2>/dev/null
        hermes config set hooks.on_session_end "[{\"command\":\"python3 $AGENT_HOOKS/sm-capture-nudge.py\",\"timeout\":5}]" 2>/dev/null
        hermes config set hooks.pre_tool_call "[{\"command\":\"python3 $AGENT_HOOKS/sm-dedup-guard.py\",\"timeout\":15,\"matcher\":\"^sm_(add_fact|ingest_document)$\"}]" 2>/dev/null
        
        # post_tool_call needs YAML list format -- write directly
        python3 -c "
import yaml
with open('$HERMES_HOME/config.yaml') as f:
    config = yaml.safe_load(f)
config.setdefault('hooks', {})['post_tool_call'] = [
    {'command': 'python3 $AGENT_HOOKS/sm-tool-receipts.py', 'timeout': 5}
]
config['hooks_auto_accept'] = True
with open('$HERMES_HOME/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
" 2>/dev/null && log "  All 6 hooks registered" || warn "  Some hooks may need manual registration"
        
        # Create allowlist
        ALLOWLIST="$HERMES_HOME/shell-hooks-allowlist.json"
        python3 -c "
import json, os
al_path = '$ALLOWLIST'
if os.path.exists(al_path):
    with open(al_path) as f:
        al = json.load(f)
else:
    al = {'version': 1, 'approvals': []}

hooks = [
    ('on_session_start', 'python3 $AGENT_HOOKS/sm-primer.py'),
    ('pre_llm_call', 'python3 $AGENT_HOOKS/sm-recall.py'),
    ('post_llm_call', 'python3 $AGENT_HOOKS/sm-autocapture.py'),
    ('on_session_end', 'python3 $AGENT_HOOKS/sm-capture-nudge.py'),
    ('pre_tool_call', 'python3 $AGENT_HOOKS/sm-dedup-guard.py'),
    ('post_tool_call', 'python3 $AGENT_HOOKS/sm-tool-receipts.py'),
]

existing = {(a['event'], a['command']) for a in al['approvals']}
for event, cmd in hooks:
    if (event, cmd) not in existing:
        al['approvals'].append({'event': event, 'command': cmd})

with open(al_path, 'w') as f:
    json.dump(al, f, indent=2)
print('Allowlist updated')
" 2>/dev/null && log "  Hook allowlist updated" || warn "  Could not update allowlist"
    else
        warn "  Hermes not found in PATH -- skipping config registration"
        warn "  You will need to manually configure hooks and MCP server"
    fi
else
    warn "Skipping hook installation (--skip-hooks)"
fi

# ============================================================================
# Step 6: Verify installation
# ============================================================================
log "Verifying installation..."

# Check MCP binary
if command -v semantic-memory-mcp &>/dev/null; then
    log "  MCP server: $(semantic-memory-mcp --version 2>/dev/null || echo 'installed')"
else
    warn "  MCP server not in PATH -- add ~/.cargo/bin to PATH"
fi

# Check Hermes
if command -v hermes &>/dev/null; then
    log "  Hermes: $(hermes --version 2>&1 | head -1)"
    
    # Check hooks
    HOOK_COUNT=$(hermes hooks list 2>/dev/null | grep "total" | grep -o '[0-9]*' || echo "0")
    log "  Hooks registered: $HOOK_COUNT"
    
    # Run hooks doctor
    hermes hooks doctor 2>/dev/null | tail -1 | sed "s/^/  /"
fi

# Check library tests
if [ -d "$SM_DIR" ]; then
    log "  semantic-memory: $(cd "$SM_DIR" && cargo test --all-features 2>/dev/null | grep 'test result:' | awk '{print $4}' | head -1) tests"
fi

# ============================================================================
# Step 7: Print summary
# ============================================================================
echo ""
echo "============================================================"
echo "  RecursiveIntell Agent Stack -- Hermes Edition"
echo "  Installation Complete"
echo "============================================================"
echo ""
echo "  What was installed:"
echo "    - Hermes Agent: $(hermes --version 2>&1 | head -1 || echo 'check PATH')"
echo "    - semantic-memory library: $SM_DIR"
echo "    - semantic-memory-mcp server: $MCP_DIR"
echo "    - MCP binary: ~/.cargo/bin/semantic-memory-mcp"
echo "    - Memory DB: $MEMORY_DIR"
if [ "$SKIP_HOOKS" = false ]; then
    echo "    - 6 Hermes hooks: $AGENT_HOOKS/"
    echo "    - Hook allowlist: $HERMES_HOME/shell-hooks-allowlist.json"
fi
echo "    - HTTP server port: $PORT"
echo ""
echo "  Capabilities (Hermes-specific):"
echo "    - 33 MCP tools"
echo "    - 9 HTTP endpoints"
echo "    - 6 agent hooks (adaptive routing, LLM rerank, autocapture,"
echo "      dedup-guard, tool receipts, integrity check)"
echo "    - Typed memory with admission gates"
echo "    - Claim-ledger integration (provenance)"
echo "    - Bitemporal search (as-of queries)"
echo "    - Verification gates (risk-class)"
echo "    - Boundary compiler (RFC 8785 JCS)"
echo "    - Query provenance with view disclosure"
echo ""
echo "  Also available for other platforms:"
echo "    - Claude Code: https://github.com/RecursiveIntell/semantic-memory-claude-kit"
echo "      (plugin with hooks, skills, commands, MCP config)"
echo "    - Codex: Add the MCP server to your AGENTS.md config"
echo "      (semantic-memory-mcp --memory-dir DIR --embedder candle --http-port $PORT)"
echo ""
echo "  Next steps:"
echo "    1. Start a Hermes session: hermes chat"
echo "    2. The warm HTTP server starts automatically with the MCP server"
echo "    3. The primer hook will check integrity and load project context"
echo "    4. The recall hook will search memory before every response"
echo ""
echo "  To start the MCP server manually:"
echo "    semantic-memory-mcp --memory-dir $MEMORY_DIR --embedder candle --embedding-model nomic-embed-text --embedding-dims 768 --http-port $PORT"
echo ""
echo "  To verify the HTTP server is running:"
echo "    curl http://127.0.0.1:$PORT/health"
echo ""
echo "  Documentation:"
echo "    - semantic-memory: https://github.com/RecursiveIntell/semantic-memory"
echo "    - semantic-memory-mcp: https://github.com/RecursiveIntell/semantic-memory-mcp"
echo "    - Claude Code kit: https://github.com/RecursiveIntell/semantic-memory-claude-kit"
echo "    - Hermes: https://hermes-agent.nousresearch.com/docs"
echo ""