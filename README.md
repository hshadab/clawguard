# ClawGuard

**Cryptographically verified guardrails for [OpenClaw](https://openclaw.ai/) agents powered by [Jolt Atlas zkML](https://github.com/ICME-Lab/jolt-atlas).**

ClawGuard sits between your OpenClaw agent and the dangerous things it can do. Before the agent runs a shell command, sends an email, or touches a sensitive file, ClawGuard checks whether that action looks safe. If it doesn't, the action gets blocked. And unlike a simple allowlist, every decision comes with a cryptographic proof that the safety check actually happened — nobody can skip it and nobody can fake the result.

## How It Works (Plain English)

Think of ClawGuard as a security guard at a door. Every time your OpenClaw agent wants to do something sensitive, it has to show the guard what it's about to do. The guard looks at the action, decides if it's safe, and stamps a receipt.

Here's what makes this different from a regular filter:

1. **The guard is a small neural network**, not a list of rules. It looks at patterns — "this command has `sudo` AND a pipe AND targets a system file" — and makes a judgment call, just like a human reviewer would, but instantly.

2. **Every decision gets a cryptographic proof.** This is the zero-knowledge proof part. The proof says "I ran this exact safety model on this exact action and it said DENIED" — and anyone can verify that statement is true. The proof doesn't reveal what the action was (that stays private), just that the check happened honestly.

3. **The proof can't be faked.** If someone tries to skip the check or forge a result, the proof won't verify. This is the same math that secures blockchains, applied to AI safety.

Here's what that looks like in practice:

```
You ask your agent: "Delete all the temp files"

  1. Agent wants to run: rm -rf /tmp/*
  2. Agent sends this to ClawGuard
  3. ClawGuard checks the action against its safety model
  4. Model says: APPROVED (it's just temp files, no sudo, no pipes)
  5. Agent gets the green light and runs the command
  6. A proof is saved that proves step 3-4 actually happened

Now imagine instead you say: "Run sudo rm -rf / | tee log.txt"

  1. Agent wants to run: sudo rm -rf / | tee log.txt
  2. Agent sends this to ClawGuard
  3. ClawGuard checks the action
  4. Model says: DENIED (sudo + destructive command + pipe = dangerous)
  5. Agent is blocked — command never runs
  6. A proof is saved showing the denial was legitimate
```

### Three built-in safety checks

ClawGuard comes with three checks that work immediately, no setup required:

**Action Gatekeeper** — "Is this command dangerous?"
Catches things like `sudo` with pipes, shell injection patterns, and writes to hidden system files. A plain `cat README.md` sails through. A `sudo rm -rf / | tee log.txt` gets blocked.

**PII Shield** — "Does this text contain personal information?"
Scans for social security numbers, credit card numbers, email addresses, phone numbers, and keywords like "password" or "api_key". Catches it before your OpenClaw agent accidentally emails someone's SSN.

**Scope Guard** — "Is this file access within bounds?"
Blocks your OpenClaw agent from reading `/etc/passwd`, poking around in `~/.ssh`, or using `../../../` to escape the project directory. Keeps file access inside your workspace.

### You can also write your own rules

Don't want your OpenClaw agent visiting banking websites? Add a rule:

```toml
[[rules]]
name = "no-banking"
block_domains = ["chase.com", "bankofamerica.com"]
actions = ["network_request"]
```

That rule gets compiled into a tiny neural network and proven in ZK just like the built-in models. More on this [below](#writing-your-own-rules).

### You can bring your own trained model

If you've trained a custom ONNX classifier (for example, a model that detects financial fraud patterns), you can plug it in and ClawGuard will run it with full ZK proof support. See [Custom ONNX Models](#custom-onnx-models).

---

## Quick Start

### What you need

- **Rust 1.80+** — install from [rustup.rs](https://rustup.rs/) if you don't have it
- **Git** — to clone the repos

That's it. No Python, no Docker, no model downloads.

### 1. Clone and build

ClawGuard is a standalone project that uses [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) as a dependency for its ZK proof engine. It does **not** live inside the Jolt Atlas repository — it's a separate repo that must be built alongside a local Jolt Atlas checkout because it depends on workspace-level crate resolution.

```bash
# Clone jolt-atlas (the ZK proof engine ClawGuard depends on)
git clone https://github.com/ICME-Lab/jolt-atlas.git

# Clone clawguard alongside it
git clone https://github.com/hshadab/clawguard.git

# Add clawguard to the jolt-atlas workspace so Cargo can resolve dependencies
cd jolt-atlas
# Add "clawguard" to the workspace members list in Cargo.toml, and create a symlink:
ln -s ../clawguard clawguard

# Build from the workspace root
cargo build -p clawguard --release
```

> **Why this setup?** ClawGuard depends on `onnx-tracer` and `zkml-jolt-core` (crates inside jolt-atlas) via path dependencies, and shares `ark-serialize` via workspace inheritance. This is a build-time requirement only — ClawGuard does not modify, fork, or contribute to the Jolt Atlas repository.

The first build compiles the ZK proof libraries from source. Subsequent builds are fast.

The binary lands at `target/release/clawguard`. Optionally put it on your PATH:

```bash
cp target/release/clawguard ~/.local/bin/
```

### 2. Try a safety check

Block a dangerous command:

```bash
clawguard check \
  --model action-gatekeeper \
  --action run_command \
  --context '{"command":"sudo rm -rf /"}' \
  --dry-run
```

You'll see:

```json
{
  "decision": "DENIED",
  "confidence": 0.95,
  "model_hash": "sha256:a1b2c3...",
  "action": "run_command",
  "timestamp": "2026-01-31T12:00:00Z"
}
```

Allow a safe command:

```bash
clawguard check \
  --model action-gatekeeper \
  --action read_file \
  --context '{"path":"README.md"}' \
  --dry-run
```

```json
{
  "decision": "APPROVED",
  "confidence": 0.72,
  ...
}
```

Check for personal data leaking:

```bash
clawguard check \
  --model pii-shield \
  --action send_email \
  --context 'Please wire $500 to account 4111-1111-1111-1111, my SSN is 123-45-6789' \
  --dry-run
```

```json
{
  "decision": "PII_DETECTED",
  ...
}
```

Block access to a sensitive file:

```bash
clawguard check \
  --model scope-guard \
  --action read_file \
  --context '{"path":"/etc/shadow"}' \
  --dry-run
```

```json
{
  "decision": "OUT_OF_SCOPE",
  ...
}
```

### 3. Generate a real ZK proof

The `--dry-run` flag skips proof generation for speed. To create an actual cryptographic proof, use `--prove` instead:

```bash
clawguard check \
  --model action-gatekeeper \
  --action run_command \
  --context '{"command":"sudo rm -rf /"}' \
  --prove
```

The output now includes a `proof_file` path. That file contains a proof anyone can verify.

### 4. Verify a proof

Take the `model_hash` and `proof_file` from the output above:

```bash
clawguard verify \
  --proof ~/.openclaw/clawguard/proofs/2026-01-31T12-00-00Z.proof.json \
  --model-hash "sha256:a1b2c3..." \
  --model-name action-gatekeeper
```

```json
{
  "valid": true,
  "model_hash_matches": true,
  "proof_file": "~/.openclaw/clawguard/proofs/2026-01-31T12-00-00Z.proof.json"
}
```

`"valid": true` means the proof checks out — the model really did run and really did produce that decision.

### 5. See what's available

```bash
# List built-in models
clawguard models

# View recent check history
clawguard history --limit 5
```

---

## Enforcement: Making It Mandatory

By default, ClawGuard just reports decisions. If you want it to actually block things, set an enforcement level.

Create `~/.openclaw/clawguard/config.toml`:

```toml
[settings]
enforcement = "hard"
```

Now when the CLI denies an action, it exits with code 1 — which tells the calling OpenClaw agent to stop.

| Level | What happens on deny | Exit code | Can the agent override? |
|---|---|---|---|
| `log` | Logs the denial, allows anyway | 0 | Yes |
| `soft` | Prints a warning, allows | 0 | Yes |
| `hard` | Blocks the action | 1 | No |

### Confidence scores

Confidence is the normalized margin between the deny and allow scores: `|deny - allow| / 128`, capped at 1.0. Values near 0 mean the model was uncertain; values near 1.0 mean a strong signal.

Other useful settings:

```toml
[settings]
enforcement = "hard"
require_proof = true              # always generate proofs, not just when --prove is passed
deny_on_error = true              # if the model crashes, treat it as a denial (fail-closed)
proof_dir = "~/.openclaw/clawguard/proofs"
```

### Using enforcement from Rust code

If you're building a Rust-based OpenClaw agent, you can skip the CLI entirely and import the library:

```toml
# In your agent's Cargo.toml
[dependencies]
clawguard = { path = "../clawguard" }
```

```rust
use clawguard::enforcement::{Decision, Guardrail};
use clawguard::GuardModel;

// Loads enforcement level from config.toml automatically
let guardrail = Guardrail::from_config();

let model = GuardModel::ActionGatekeeper;
let context = serde_json::json!({ "command": "sudo rm -rf /" });

match guardrail.check(&model, "run_command", &context)? {
    Decision::Allow { confidence, .. } => {
        println!("Safe to proceed (confidence: {confidence})");
        // run the command
    }
    Decision::Deny { reason, overridable, .. } => {
        println!("Blocked: {reason}");
        if !overridable {
            // hard enforcement — do NOT proceed
            return Err(eyre::eyre!("action denied"));
        }
    }
}
```

There's also an `ActionGuard` trait for middleware-style integration. `Guardrail` implements it — calling `before_action()` runs all applicable models and returns the most restrictive decision:

```rust
use clawguard::enforcement::ActionGuard;

let decision = guardrail.before_action("run_command", &context)?;
```

---

## Writing Your Own Rules

You don't need to train a model to add custom safety checks. Write declarative rules in `config.toml` and ClawGuard compiles them into a small neural network automatically. The compiled model gets ZK-proved just like the built-in ones.

### Example: Block banking sites and protect secrets

```toml
[[rules]]
name = "no-banking"
block_domains = ["chase.com", "bankofamerica.com", "wellsfargo.com"]
actions = ["network_request"]

[[rules]]
name = "protect-credentials"
block_paths = ["~/.ssh", "~/.gnupg", "~/.aws", "*.env"]
block_keywords = ["password", "secret", "api_key", "token"]
actions = ["read_file", "write_file", "run_command"]
```

With these rules in place:

```bash
# This gets blocked — domain matches "chase.com"
clawguard check --model action-gatekeeper --action network_request \
  --context '{"url":"https://chase.com/transfer"}' --dry-run

# This gets blocked — path matches "~/.ssh"
clawguard check --model scope-guard --action read_file \
  --context '{"path":"~/.ssh/id_rsa"}' --dry-run

# This gets blocked — keyword matches "password"
clawguard check --model action-gatekeeper --action run_command \
  --context '{"command":"echo password123 > creds.txt"}' --dry-run
```

### How rules work under the hood

Each rule condition (a domain, a path, a keyword) becomes an input slot. Each rule becomes a neuron that fires when its conditions match. The deny output fires when any rule neuron fires. This all compiles into a 2-layer neural network with the same structure as the built-in models, so ZK proving works identically.

### Rule options

| Field | What it does | Example |
|---|---|---|
| `name` | A label for the rule (for your reference) | `"no-banking"` |
| `block_domains` | Block when URL/context contains this domain | `["chase.com"]` |
| `block_paths` | Block when file path matches | `["~/.ssh", "*.env"]` |
| `block_keywords` | Block when any text in context matches | `["password", "sudo"]` |
| `actions` | Which action types this rule applies to | `["network_request"]` |

---

## Custom ONNX Models

If you've trained your own classifier (in PyTorch, TensorFlow, etc.) and exported it to ONNX format, ClawGuard can load it, run inference, and prove the result in ZK.

### Step 1: Export your model to ONNX

Your model should be a classifier with:
- A fixed input shape (e.g., `[1, 16]`)
- Two output neurons (deny score and allow score)

### Step 2: Create a metadata file

Save this alongside your ONNX file as `my-model.meta.toml`:

```toml
input_shape = [1, 16]
encoding = "action"             # how to encode the context (see below)
labels = ["DENIED", "APPROVED"] # what the two outputs mean
scale = 7                       # fixed-point scale (2^7 = 128)
max_trace_length = 16384        # for ZK proving (optional, auto-detected)
```

### Step 3: Register in config

```toml
[models.my-fraud-detector]
path = "~/.openclaw/clawguard/models/fraud.onnx"
meta = "~/.openclaw/clawguard/models/fraud.meta.toml"
actions = ["send_payment", "transfer_funds"]
```

### Encoding modes

The `encoding` field in the metadata controls how action context gets turned into model input:

| Mode | What it does | When to use |
|---|---|---|
| `action` | One-hot action type + binary context features | General-purpose classifiers |
| `pii` | Regex-based PII pattern counts | Text content classifiers |
| `scope` | Path depth, system dir detection, etc. | File access classifiers |
| `raw` | Reads context directly as a JSON array of integers | When you handle encoding yourself |

### How hashing works

ONNX model identity is the SHA256 hash of the file bytes. This means if someone swaps the model file, the hash changes and old proofs won't verify against the new hash.

---

## All CLI Commands

```bash
# Check an action (instant, no proof)
clawguard check --model <name> --action <type> --context '<json>' --dry-run

# Check an action and generate a ZK proof
clawguard check --model <name> --action <type> --context '<json>' --prove

# Verify a saved proof file
clawguard verify --proof <path> --model-hash <hash> --model-name <name>

# View recent check history
clawguard history [--limit N]

# List available models
clawguard models

# Validate your config file
clawguard config-check
```

**Model names** for the `--model` flag:
- `action-gatekeeper` — command safety
- `pii-shield` — personal data detection
- `scope-guard` — file access boundaries
- Or a path to a real ONNX file

**Action types** for the `--action` flag:
- `run_command` — shell commands
- `send_email` — outbound email
- `read_file` — reading files
- `write_file` — writing files
- `network_request` — HTTP/network calls

---

## Full Configuration Reference

Everything goes in `~/.openclaw/clawguard/config.toml`:

```toml
# --- Enforcement and proof settings ---
[settings]
enforcement = "hard"              # "log", "soft", or "hard"
require_proof = true              # always generate ZK proofs
deny_on_error = true              # treat model errors as denials
proof_dir = "~/.openclaw/clawguard/proofs"
# max_trace_length = 16384       # ZK trace length (auto-detected if omitted)

# --- Custom ONNX models ---
[models.my-custom-model]
path = "~/.openclaw/clawguard/models/custom.onnx"
meta = "~/.openclaw/clawguard/models/custom.meta.toml"
actions = ["run_command", "network_request"]

# --- Declarative policy rules ---
[[rules]]
name = "no-banking"
block_domains = ["chase.com", "bankofamerica.com", "wellsfargo.com"]
actions = ["network_request"]

[[rules]]
name = "protect-credentials"
block_paths = ["~/.ssh", "~/.gnupg", "~/.aws", "*.env"]
block_keywords = ["password", "secret", "api_key"]
actions = ["read_file", "write_file", "run_command"]
```

---

## What This Proves (and What It Doesn't)

**It proves:**
- The safety model **actually ran** on the action (not skipped or bypassed)
- The model produced **exactly the decision claimed** (not forged after the fact)
- The model matches **a known hash** (not secretly swapped for a weaker one)
- All of the above **without revealing** what the action was (zero-knowledge)

**It does not prove:**
- That the safety model itself is good (a bad model produces valid proofs of bad decisions)
- That the OpenClaw agent actually called ClawGuard (architecture enforcement is separate — use the [enforcement library](#using-enforcement-from-rust-code) to make it mandatory)
- That the LLM is safe from prompt injection (that's a different problem)

This is an auditable verification layer. It makes safety checks tamper-proof and independently verifiable.

---

## OpenClaw Integration

[OpenClaw](https://openclaw.ai/) is an open-source personal AI assistant that runs locally on your machine and connects to messaging channels like WhatsApp, Telegram, Discord, and Slack. It can run shell commands, send emails, manage files, and access APIs on your behalf — which is exactly the kind of power that needs guardrails.

ClawGuard integrates as an OpenClaw [skill](https://docs.openclaw.ai/tools/skills). The agent's runtime loads `SKILL.md` into context, which instructs the agent to run `clawguard check` before any sensitive action.

### Install as a workspace skill

The easiest way is to symlink or copy into your workspace skills directory:

```bash
# Workspace skills have the highest loading priority
ln -s /path/to/clawguard ~/.openclaw/workspace/skills/clawguard
```

Alternatively, install to the managed skills directory:

```bash
ln -s /path/to/clawguard ~/.openclaw/skills/clawguard
```

Make sure the `clawguard` binary is on your PATH (or in a location the agent can find). The skill's `metadata.openclaw.requires.bins` field gates loading on the binary being available.

### Verify it loaded

Open a chat session with your OpenClaw agent and ask it to list its skills, or check the gateway logs. ClawGuard should appear with its shield emoji.

### How it works at runtime

When you ask your OpenClaw agent to do something sensitive (run a command, send a message, read a file), the agent sees the SKILL.md instructions and calls `clawguard check` before executing. If the check returns DENIED and enforcement is set to `hard`, the agent stops and tells you why.

### Lobster pipelines

For Lobster pipeline workflows, use the included approval gate:

```bash
lobster run pipelines/guardrail-gate.lobster \
  --input '{"action": "send_email", "context": {"to": "ceo@corp.com"}}'
```

The pipeline halts at an approval gate if the guardrail denies, with a 5-minute timeout before auto-deny.

---

## Running Tests

```bash
# Unit tests — instant, covers all 3 built-in models + policy rule compilation
cargo test --lib

# Integration test — full ZK prove-and-verify round trip + enforcement library
cargo test --test integration --release
```

---

## Architecture

```
Your OpenClaw Agent
    |
    | wants to do something sensitive
    v
Rust Library (clawguard)  -or-  CLI (clawguard)
    |                                   |
    |  guardrail.check()                |  clawguard check ...
    |                                   |
    +-----------------------------------+
                    |
    +---------------+----------------+------------------+
    |               |                |                  |
Built-in Models  Your Rules      ONNX Models      jolt-atlas
(action-gatekeeper, (from config,   (your trained    (ZK proof engine)
 pii-shield,       compiled to      classifiers)        |
 scope-guard)      neural net)                          v
    |               |                |            proof file
    +---------------+----------------+            (JSON + base64)
                    |                                  |
                    v                                  v
              APPROVED / DENIED                 proof stored locally
                    |                           (verifiable forever)
                    v
         agent proceeds or stops
```

### Source layout

```
src/
    lib.rs             -- Library entry point: GuardModel, config, run_guardrail()
    main.rs            -- CLI (thin wrapper over the library)
    action.rs          -- ActionType enum (type-safe action names)
    enforcement.rs     -- Enforcement levels, Decision type, Guardrail struct, ActionGuard trait
    rules.rs           -- Policy rule parsing and compilation to neural network weights
    onnx_support.rs    -- ONNX model loading and metadata
    encoding.rs        -- Turns actions/context into model input vectors
    proving.rs         -- ZK proof generation and verification via JoltSNARK
    models/
      action_gatekeeper.rs  -- Command safety classifier
      pii_shield.rs         -- Personal data detector
      scope_guard.rs        -- File access boundary enforcer
  tests/
    integration.rs     -- End-to-end prove-verify round trip + enforcement test
  examples/
    config.toml        -- Full working configuration example
```

## License

MIT
