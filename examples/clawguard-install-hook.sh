#!/usr/bin/env bash
#
# ClawGuard Pre-Install Hook
#
# Usage: clawguard-install-hook.sh <skill-name> [version]
#
# This script scans a skill for safety issues before installation.
# Intended for use as a pre-install hook in the OpenClaw CLI.
#
# Manual testing:
#   chmod +x examples/clawguard-install-hook.sh
#   ./examples/clawguard-install-hook.sh weather-helper 1.0.0
#
# Requirements:
#   - clawguard binary in PATH (or set CLAWGUARD_BIN)
#   - clawhub binary in PATH (or set CLAWHUB_BIN) for live skill export

set -euo pipefail

SKILL_NAME="${1:?Usage: $0 <skill-name> [version]}"
SKILL_VERSION="${2:-}"

CLAWGUARD_BIN="${CLAWGUARD_BIN:-clawguard}"
CLAWHUB_BIN="${CLAWHUB_BIN:-clawhub}"

TMP_DIR=$(mktemp -d)
TMP_JSON="${TMP_DIR}/${SKILL_NAME}.json"
trap 'rm -rf "$TMP_DIR"' EXIT

# Step 1: Export the skill to a temporary JSON file
echo "Exporting skill ${SKILL_NAME}..."
if [ -n "$SKILL_VERSION" ]; then
    "$CLAWHUB_BIN" export "$SKILL_NAME" --version "$SKILL_VERSION" --output "$TMP_JSON"
else
    "$CLAWHUB_BIN" export "$SKILL_NAME" --output "$TMP_JSON"
fi

# Step 2: Scan the skill with ClawGuard
echo "Scanning ${SKILL_NAME} for safety issues..."
SCAN_OUTPUT=$("$CLAWGUARD_BIN" scan-skill --input "$TMP_JSON" --format json 2>/dev/null) || true

# Step 3: Parse the decision from JSON output
DECISION=$(echo "$SCAN_OUTPUT" | grep -o '"decision":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$DECISION" ]; then
    echo "WARNING: Could not determine safety decision. Proceeding with caution."
    exit 0
fi

echo "Decision: $DECISION"

# Step 4: Act on the decision
case "$DECISION" in
    allow)
        echo "Skill ${SKILL_NAME} is safe. Proceeding with installation."
        if [ -n "$SKILL_VERSION" ]; then
            "$CLAWHUB_BIN" install "$SKILL_NAME" --version "$SKILL_VERSION"
        else
            "$CLAWHUB_BIN" install "$SKILL_NAME"
        fi
        ;;
    flag)
        echo "WARNING: Skill ${SKILL_NAME} has been flagged for review."
        echo "$SCAN_OUTPUT" | grep -o '"reasoning":"[^"]*"' | head -1 | cut -d'"' -f4
        read -rp "Do you want to proceed with installation? [y/N] " CONFIRM
        case "$CONFIRM" in
            [yY]|[yY][eE][sS])
                echo "Installing ${SKILL_NAME} despite flag..."
                if [ -n "$SKILL_VERSION" ]; then
                    "$CLAWHUB_BIN" install "$SKILL_NAME" --version "$SKILL_VERSION"
                else
                    "$CLAWHUB_BIN" install "$SKILL_NAME"
                fi
                ;;
            *)
                echo "Installation cancelled."
                exit 1
                ;;
        esac
        ;;
    deny)
        echo "BLOCKED: Skill ${SKILL_NAME} has been denied by ClawGuard."
        echo "$SCAN_OUTPUT" | grep -o '"reasoning":"[^"]*"' | head -1 | cut -d'"' -f4
        exit 1
        ;;
    *)
        echo "Unknown decision: $DECISION"
        exit 1
        ;;
esac
