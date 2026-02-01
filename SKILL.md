---
name: clawguard
description: Gate sensitive OpenClaw agent actions through built-in guardrail models with zero-knowledge proof verification powered by Jolt Atlas zkML. Ships three ready-to-use models (action-gatekeeper, pii-shield, scope-guard), user-defined policy rules, and custom ONNX model support — all with hard enforcement.
homepage: https://github.com/ICME-Lab/jolt-atlas
user-invocable: true
metadata: {"openclaw":{"emoji":"\uD83D\uDEE1\uFE0F","requires":{"bins":["clawguard"]}}}
---

# ClawGuard — zkML Guardrail Verification

You have access to the `clawguard` CLI tool for verifying [OpenClaw](https://openclaw.ai/) agent actions against guardrail models with cryptographic proofs powered by [Jolt Atlas zkML](https://github.com/ICME-Lab/jolt-atlas).

## Built-in Models

Three guardrail models are available out of the box. Select the model by passing its name as the `--model` argument:

| Model | Detects | Decisions |
|---|---|---|
| `action-gatekeeper` | Dangerous commands (sudo, pipes, dotfile writes) | DENIED / APPROVED |
| `pii-shield` | PII in text (SSN, email, phone, CC, passwords) | PII_DETECTED / CLEAN |
| `scope-guard` | File access outside workspace | OUT_OF_SCOPE / IN_SCOPE |

## When to Use

Before executing any **sensitive action**, you MUST run it through clawguard. Sensitive actions include:

- Running shell commands that modify files, send network requests, or access credentials
- Sending emails, messages, or any outbound communication on behalf of the user
- Accessing, reading, or transmitting personal data (contacts, documents, financial info)
- Making API calls that create, modify, or delete resources
- Any action the user has explicitly flagged as requiring guardrail approval

## How to Use

### Step 1: Choose the right model

- **Commands and system actions** -> `action-gatekeeper`
- **Text content that might contain personal data** -> `pii-shield`
- **File paths and directory access** -> `scope-guard`

### Step 2: Check an action before executing it

```bash
# Dry run (instant, no proof):
clawguard check \
  --model action-gatekeeper \
  --action "run_command" \
  --context '{"command": "sudo rm -rf /tmp/cache"}' \
  --dry-run

# With ZK proof:
clawguard check \
  --model action-gatekeeper \
  --action "run_command" \
  --context '{"command": "sudo rm -rf /tmp/cache"}' \
  --prove
```

The tool will:
1. Encode the action and context into model features
2. Run inference through the selected guardrail model
3. Return a decision with confidence score
4. If `--prove` is passed, generate a zero-knowledge proof that the model was evaluated correctly

### Step 3: Interpret the result

```json
{
  "decision": "DENIED",
  "confidence": 0.95,
  "model_hash": "sha256:a1b2c3d4...",
  "proof_file": "~/.openclaw/clawguard/proofs/2026-01-30T12-00-00Z.proof.json",
  "proof_valid": true,
  "action": "run_command",
  "timestamp": "2026-01-30T12:00:00Z"
}
```

- If `decision` is `APPROVED` / `CLEAN` / `IN_SCOPE`: proceed with the action.
- If `decision` is `DENIED` / `PII_DETECTED` / `OUT_OF_SCOPE`: inform the user that the guardrail blocked the action, include the reason, and ask how they want to proceed.
- If proof generation fails: warn the user that the action cannot be cryptographically attested.

### Step 4: Respect enforcement level

The CLI exits with code 1 on **hard** enforcement denials. Check the exit code:

- **Exit 0**: action allowed (or soft/log enforcement)
- **Exit 1**: action denied with hard enforcement — do NOT proceed

### Step 5: For denied actions

Tell the user:
> "ClawGuard blocked this action. The guardrail model classified [action] as outside permitted bounds. You can override this manually, but no proof will be generated for the override."

### Available Commands

```bash
# Check a single action (dry run)
clawguard check --model <name> --action <type> --context <json> --dry-run

# Check with ZK proof generation
clawguard check --model <name> --action <type> --context <json> --prove

# Verify an existing proof
clawguard verify --proof <path> --model-hash <hash> --model-name <name>

# List recent proof history
clawguard history [--limit N]

# Show available guardrail models
clawguard models
```

### Model Selection by Filename

The `--model` argument matches on filename keywords:
- Contains `action` or `gatekeeper` -> action-gatekeeper
- Contains `pii` or `shield` -> pii-shield
- Contains `scope` -> scope-guard
- Otherwise, attempts to load as a real ONNX file from the path

### Enforcement Levels

The config file (`~/.openclaw/clawguard/config.toml`) controls enforcement behavior:

| Level | CLI behavior | Exit code on deny |
|---|---|---|
| `log` | Always allows, logs denial | 0 |
| `soft` | Prints warning on deny | 0 |
| `hard` | Blocks on deny | 1 |

### User-Defined Policy Rules

Users can define declarative rules in config.toml that are compiled into ZK-provable models:

```toml
[[rules]]
name = "block-banking-sites"
block_domains = ["chase.com", "bankofamerica.com"]
actions = ["network_request"]

[[rules]]
name = "protect-secrets"
block_paths = ["~/.ssh", "~/.aws", "*.env"]
block_keywords = ["password", "secret"]
actions = ["read_file", "write_file", "run_command"]
```

These rules run automatically alongside built-in models. Either one can deny an action.

### Configuration

The user can configure additional ONNX models and policy rules in `~/.openclaw/clawguard/config.toml`:

```toml
[models.default]
path = "~/.openclaw/clawguard/custom-classifier.onnx"
actions = ["*"]

[models.my-custom-model]
path = "~/.openclaw/clawguard/models/custom.onnx"
meta = "~/.openclaw/clawguard/models/custom.meta.toml"
actions = ["run_command", "network_request"]

[settings]
require_proof = true
proof_dir = "~/.openclaw/clawguard/proofs"
deny_on_error = true
enforcement = "hard"
```

### Rust Library Integration

For Rust agents, import the library crate directly instead of using the CLI:

```rust
use clawguard::enforcement::{Decision, Guardrail};
use clawguard::GuardModel;

let guardrail = Guardrail::from_config();
let decision = guardrail.check(&GuardModel::ActionGatekeeper, "run_command", &context)?;
```

### Important Notes

- NEVER skip the guardrail check for sensitive actions, even if the user seems impatient
- If the `clawguard` binary is not available, warn the user that guardrail verification is unavailable
- Proofs are stored locally and can be shared with third parties for independent verification
- The proof reveals NOTHING about the user's data — only that a specific model produced a specific decision
- With `enforcement = "hard"`, a denied action exits with code 1 — treat this as a hard stop
