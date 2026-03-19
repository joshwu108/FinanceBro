# FinanceBro Implementation Plan

## 🧠 Core Philosophy

This project follows a **Ralph Wiggins Loop**:

> Build → Critique → Refine → Reset Context → Repeat

Goal:
- Stay below the “dumb zone”
- Avoid context bloat
- Maintain high code quality across iterations

---

# 🔁 Ralph Wiggins Loop (RWL)

Each agent is built using:

1. Builder Prompt
2. Critic Prompt
3. Refine Prompt
4. **Hard Reset Context (CRITICAL)**

---

## ⚠️ Why This Matters

Long-running chats degrade because:
- summarization tokens accumulate
- earlier mistakes propagate
- reasoning quality drops

We fix this by:
> **starting a fresh Claude session every iteration**

---

# 🧱 Agent Implementation Order

## Phase 1 — Core System

1. FeatureAgent
2. BacktestAgent
3. StatsAgent
4. ModelAgent

---

## Phase 2 — Quant Validation

5. WalkForwardAgent
6. OverfittingAgent

---

## Phase 3 — Portfolio + Risk

7. RiskAgent
8. PortfolioAgent

---

## Phase 4 — Differentiation

9. ExplainabilityAgent
10. Live Inference API

---

# 🧩 Prompt Templates

---

## 🟢 Builder Prompt
You are the <AgentName> subagent.

Follow strictly:

CLAUDE.md

specs/<agent>_spec.md

subagents/<agent>.md

Context:

Describe upstream outputs

Task:
Implement backend/agents/<agent>.py

Requirements:

Follow all constraints

No look-ahead bias

Clean, modular code

Output:

Clearly defined outputs


---

## 🔍 Critic Prompt


You are the Overfitting Subagent.

Review the <AgentName> implementation.

Check:

leakage

incorrect assumptions

bad abstractions

edge cases

Be extremely critical.


---

## 🔁 Refine Prompt


Fix all issues identified.

Ensure:

correctness

clean architecture

compliance with CLAUDE.md


---

# 🔄 Context Reset Strategy (CRITICAL)

After each refine:

❌ DO NOT continue conversation  
✅ START A NEW CLAUDE SESSION

---

# 🖥️ Bash Automation (Fresh Context Loop)

Create:

```bash
scripts/rwl.sh

Script
#!/bin/bash

AGENT=$1

echo "Running Ralph Wiggins Loop for $AGENT"

echo "Step 1: Builder"
cat prompts/$AGENT/builder.txt

read -p "Paste into Claude, press enter when done..."

echo "Step 2: Critic"
cat prompts/$AGENT/critic.txt

read -p "Paste into NEW Claude session, press enter when done..."

echo "Step 3: Refine"
cat prompts/$AGENT/refine.txt

read -p "Paste into NEW Claude session, press enter when done..."

echo "Loop complete for $AGENT"