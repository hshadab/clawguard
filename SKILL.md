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
| `skill-safety` | Malicious/dangerous OpenClaw skills | SAFE / CAUTION / DANGEROUS / MALICIOUS |

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

# Validate config file
clawguard config-check [--ignore-config-errors]

# Manage proof migrations when models update
clawguard migrate-proofs [--dry-run] [--archive]
```

### Model Selection

The `--model` argument accepts exact model names:
- `action-gatekeeper` or `action_gatekeeper` -> action-gatekeeper
- `pii-shield` or `pii_shield` -> pii-shield
- `scope-guard` or `scope_guard` -> scope-guard
- `skill-safety` or `skill_safety` -> skill-safety
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

---

## Skill Safety Guardrail

ClawGuard includes a specialized guardrail for evaluating OpenClaw/ClawHub skills before installation. This helps protect against malicious skills that may contain:

- **Reverse shells** and backdoors
- **Credential exfiltration** from `.env` files
- **LLM context window attacks** (instructing the agent to pass secrets through plaintext)
- **Obfuscated payloads** designed to evade scanners
- **Persistence mechanisms** (cron jobs, systemd services)

### Scanning a Skill

```bash
# Scan a skill from a JSON file
clawguard scan-skill --input skill.json

# Scan a local SKILL.md file
clawguard scan-skill --input ./my-skill/SKILL.md

# Scan with VirusTotal report
clawguard scan-skill --input skill.json --vt-report vt-report.json

# Generate ZK proof for the scan
clawguard scan-skill --input skill.json --prove

# Output as JSON
clawguard scan-skill --input skill.json --format json

# Output full receipt
clawguard scan-skill --input skill.json --prove --format receipt

# Save receipt to file
clawguard scan-skill --input skill.json --prove --output receipt.json
```

### Classification Levels

| Class | Decision | Description |
|-------|----------|-------------|
| `SAFE` | allow | No concerning patterns detected |
| `CAUTION` | allow | Minor concerns (network calls, env reads) but likely functional |
| `DANGEROUS` | deny | Significant risk (credential exposure, excessive permissions) |
| `MALICIOUS` | deny | Active malware indicators (reverse shells, obfuscation) |

### 22-Feature Analysis

The skill safety classifier analyzes 22 features:

1. **Shell execution** patterns (exec, spawn, system)
2. **Network calls** (fetch, curl, axios)
3. **File writes** to disk
4. **Environment access** (.env, process.env)
5. **Credential patterns** in instructions (API key, password, token)
6. **External downloads** of executables
7. **Obfuscation** (base64, eval, dynamic imports)
8. **Privilege escalation** (sudo, chmod)
9. **Persistence mechanisms** (cron, systemd, autostart)
10. **Data exfiltration** patterns (POST to external domains)
11. **SKILL.md complexity** (line count)
12. **Script file count**
13. **Dependency count**
14. **Author account age**
15. **Author skill count**
16. **Stars/endorsements**
17. **Download count**
18. **VirusTotal report** presence
19. **VirusTotal malicious flags**
20. **Password-protected archives** (scanner evasion)
21. **Reverse shell patterns** (nc -e, /dev/tcp, bash -i)
22. **LLM secret exposure** (instructions to pass secrets through context)

### Example Output

```
Skill Safety Scan Results
========================
Skill: moltyverse-email v1.1.0

Classification: DANGEROUS
Decision:       deny
Confidence:     87.0%
Reasoning:      Significant risk patterns detected

Scores:
  SAFE:       3.0%
  CAUTION:    8.0%
  DANGEROUS:  87.0%
  MALICIOUS:  2.0%

Risk Factors:
  - llm_secret_exposure: true (SKILL.md instructs passing secrets through context)
  - credential_patterns: 6 (API key, password, token references)
  - env_access_count: 8 (heavy .env usage)

Receipt ID: gr_safety_7f3a9b2e1d4c
Model Hash: sha256:a4c8e2f1b3d7...
```

### Prover Service

Run ClawGuard as an HTTP service for remote skill evaluation:

```bash
# Start the prover service
clawguard serve --bind 127.0.0.1:8080

# With proof generation enabled
clawguard serve --bind 0.0.0.0:8080 --require-proof

# Limit concurrent proofs
clawguard serve --bind 127.0.0.1:8080 --max-proofs 2

# Configure rate limiting (requests per minute per IP, default: 60)
clawguard serve --bind 127.0.0.1:8080 --rate-limit 120
```

**Endpoints:**

- `GET /health` — Health check with model hash and uptime
- `POST /guardrail/safety` — Evaluate skill safety

**Request format:**

```json
{
  "skill": {
    "name": "my-skill",
    "version": "1.0.0",
    "author": "developer",
    "skill_md": "# My Skill\n\n...",
    "scripts": [],
    "metadata": {
      "stars": 100,
      "downloads": 5000,
      "author_account_created": "2024-01-01T00:00:00Z",
      "author_total_skills": 10
    },
    "files": []
  },
  "vt_report": null,
  "generate_proof": false
}
```

**Response format:**

```json
{
  "success": true,
  "receipt": {
    "version": "1.0.0",
    "receipt_id": "gr_safety_abc123",
    "guardrail": {
      "domain": "safety",
      "action_type": "install_skill",
      "policy_id": "icme:skill-safety-v1",
      "model_hash": "sha256:..."
    },
    "evaluation": {
      "decision": "allow",
      "classification": "SAFE",
      "confidence": 0.85,
      "scores": {
        "SAFE": 0.85,
        "CAUTION": 0.12,
        "DANGEROUS": 0.02,
        "MALICIOUS": 0.01
      }
    },
    "proof": {
      "system": "jolt-atlas",
      "proof_bytes": "...",
      "verification_key_hash": "sha256:..."
    }
  },
  "processing_time_ms": 150
}
```

### Pre-Install Hook Integration

For OpenClaw agents, integrate skill safety as a pre-install check:

```rust
use clawguard::{run_skill_safety, skill::SkillFeatures};

// Before installing any skill:
let features = SkillFeatures::extract(&skill, None);
let (classification, confidence, model_hash, proof_path) =
    run_skill_safety(&features, true, None)?;

if classification.is_deny() {
    eprintln!("Blocked installation of {}: {}", skill.name, classification.as_str());
    return Err(format!("Safety guardrail blocked skill"));
}
```

### Receipt Verification

Receipts are cryptographically verifiable:

1. **Nonce uniqueness** — Receipt ID has not appeared before
2. **Model binding** — Model hash matches registered on-chain model
3. **Input binding** — Feature commitment matches extracted features
4. **Proof verification** — ZK proof validates with verification key
5. **Payment binding** — (if x402) Transaction exists on-chain
6. **Output consistency** — Classification maps correctly to decision
