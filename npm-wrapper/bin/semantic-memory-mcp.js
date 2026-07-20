#!/usr/bin/env node
/**
 * npm wrapper for semantic-memory-mcp Rust binary.
 *
 * Tries (in order):
 * 1. A pre-built binary downloaded from GitHub releases
 * 2. A cargo-installed binary from crates.io
 * 3. Falls back to telling the user to install Rust
 */

const { execSync, spawn } = require("child_process");
const os = require("os");
const path = require("path");
const fs = require("fs");

const PLATFORM = os.platform();
const ARCH = os.arch();
const VERSION = "1.1.0";

const binaryName = "semantic-memory-mcp";
const binDir = path.join(__dirname, "..", ".bin-cache");
const binPath = path.join(binDir, binaryName);

function getDownloadUrl() {
  const platformMap = {
    "linux-x64": `https://github.com/RecursiveIntell/semantic-memory-mcp/releases/download/v${VERSION}/semantic-memory-mcp-linux-x64`,
    "darwin-x64": `https://github.com/RecursiveIntell/semantic-memory-mcp/releases/download/v${VERSION}/semantic-memory-mcp-darwin-x64`,
    "darwin-arm64": `https://github.com/RecursiveIntell/semantic-memory-mcp/releases/download/v${VERSION}/semantic-memory-mcp-darwin-arm64`,
  };
  const key = `${PLATFORM}-${ARCH}`;
  return platformMap[key] || null;
}

function ensureBinary() {
  // Already downloaded?
  if (fs.existsSync(binPath)) {
    try { fs.chmodSync(binPath, 0o755); } catch {}
    return binPath;
  }

  // Try downloading pre-built binary
  const url = getDownloadUrl();
  if (url) {
    if (!fs.existsSync(binDir)) fs.mkdirSync(binDir, { recursive: true });
    try {
      execSync(`curl -sL -o "${binPath}" "${url}"`, { stdio: "pipe" });
      fs.chmodSync(binPath, 0o755);
      return binPath;
    } catch {}
  }

  // Fallback: check if cargo-installed binary is on PATH
  try {
    execSync(`which ${binaryName}`, { stdio: "pipe" });
    return binaryName; // Use from PATH
  } catch {}

  // Fallback: try cargo install
  try {
    execSync("cargo install semantic-memory-mcp --locked", { stdio: "inherit" });
    return binaryName; // Use from PATH after install
  } catch {
    process.stderr.write(
      `No pre-built binary for ${PLATFORM}-${ARCH} and cargo is not available.\n` +
      `Install Rust from https://rustup.rs and run: cargo install semantic-memory-mcp --locked\n`
    );
    return null;
  }
}

const binary = ensureBinary();
if (!binary) {
  process.exit(1);
}

// Pass through all arguments
const child = spawn(binary, process.argv.slice(2), {
  stdio: "inherit",
  env: process.env,
});
child.on("exit", (code) => process.exit(code || 0));