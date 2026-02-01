# ClawGuard — Lobster approval gate pipeline
#
# Usage: lobster run guardrail-gate.lobster --input '{"action":"send_email","context":{...}}'
#
# This pipeline:
#   1. Runs the action through clawguard with proof generation
#   2. Halts at an approval gate if the guardrail denies
#   3. Stores the proof for audit
#   4. Returns the decision + proof path to the agent

# Step 1: Run guardrail check with proof
step check:
  tool: bash
  run: |
    clawguard check \
      --model ~/.openclaw/clawguard/action-classifier.onnx \
      --action "{{ input.action }}" \
      --context '{{ input.context | tojson }}' \
      --prove
  output: json

# Step 2: Gate on denial — halt and require human approval
step gate:
  needs: [check]
  when: "{{ check.decision == 'DENIED' }}"
  approval:
    message: |
      ClawGuard DENIED action "{{ input.action }}".
      Confidence: {{ check.confidence }}
      Model: {{ check.model_hash }}

      Override and proceed anyway? (no proof will cover the override)
    timeout: 300  # 5 minutes before auto-deny

# Step 3: Return result to agent
step result:
  needs: [check]
  output:
    decision: "{{ check.decision }}"
    proof: "{{ check.proof_file }}"
    model: "{{ check.model_hash }}"
    overridden: "{{ gate.approved | default(false) }}"
