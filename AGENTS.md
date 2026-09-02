# AGENTS.md – Rules for AI Coding Agents Working on Proof Lab

**This file is mandatory reading before any work begins.**

You are an AI coding agent helping build **Proof Lab**, a research-first quantitative trading platform.  
Your job is to implement the system correctly, safely, and incrementally.  
You do **not** own the project. The human owns the architecture, the quality bar, and the merge decisions.

---

## 1. Core Identity of the Project

Proof Lab is **not** “an AI that predicts the market.”

It is a quantitative research and algorithmic execution platform that:
- Classifies predefined trading setups (target / stop / horizon)
- Subjects every strategy to rigorous chronological validation
- Models realistic execution costs
- Requires explicit risk and validation gates before any live capital is risked

The central trust mechanism is the **Proof Engine**.

If you ever feel tempted to optimise for an impressive backtest at the expense of correctness, stop. Correctness always wins.

---

## 2. Absolute Rules (Never Violate)

1. **Never work directly on `development`, `staging`, or `main`.**  
   You may only commit to the current feature/milestone branch (`feat/mXX-...`).

2. **Never implement more than the current milestone allows.**  
   Do not jump ahead to later milestones, UI, live trading, or packaging unless the current milestone contract explicitly includes it.

3. **Never invent fake functionality or fake performance numbers.**  
   If something is not implemented, return or raise a clear “Not implemented”.  
   All metrics must come from real calculations on real data.

4. **Never allow look-ahead bias or data leakage.**  
   Features, labels, scaling, and validation must be strictly causal.

5. **Never use the blind test set for any form of tuning, selection, or calibration.**

6. **Never enable live trading by default.**  
   Live mode must remain disabled until explicitly activated through the proper gates.

7. **Never log or commit secrets** (broker credentials, API keys, etc.).

8. **Never execute arbitrary code from a `.plb` strategy package.**

9. **Every model artifact must contain its full preprocessing pipeline and feature schema.**

10. **The UI (when it exists) must never contain business logic that contradicts the quantitative engine.**

---

## 3. How You Receive Work

- The human will give you **one milestone at a time**.
- You will receive:
  - The current milestone contract (from `PROGRESS.md`)
  - The current state of the codebase on the feature branch
- You must **not** request or assume the entire specification unless the human explicitly provides it.
- Completed milestones remain in `PROGRESS.md` so you accumulate context over time.  
  You still work **only** on the newest (in-progress) milestone unless the human explicitly tells you otherwise.

---

## 4. How You Must Work

- Create **one atomic commit per task** listed in the milestone.
- Write tests together with (or before) the implementation.
- Prefer clear, readable, well-typed Python.
- Follow the package structure defined in the specification.
- When a task is finished, mark it done in `PROGRESS.md` (see Section 10) and stop to wait for human review if required.
- Do not open a pull request or merge anything yourself unless explicitly instructed.

---

## 5. Commit Message Rules (Mandatory)

When creating commits you must follow this exact format.  
Do not wait for the user to provide a diff — inspect the changes yourself with `git status` and `git diff`.

### Header Format

```
<type>(`<path>`): <short description with backticked filename>
```

- `<type>` must be one of: `feat`, `fix`, `refactor`, `style`, `chore`, `perf`, `test`, `docs`, `build`, `ci`
- `<path>` must be the full relative directory path wrapped in backticks
- The primary filename in the description must also be wrapped in backticks
- Keep the entire header under 100 characters
- Use present tense

### Body Format

Exactly two paragraphs, each written as a single unbroken line:

```
<header>

<first paragraph – what was done>

<second paragraph – why it was done>
```

- Blank line after the header
- Blank line between the two paragraphs
- No bullet points, no lists, no line wrapping inside paragraphs

### Workflow

1. Run `git status` and `git diff` (or `git diff --cached`)
2. Stage the relevant changes with `git add`
3. Create exactly one conventional commit following the format above
4. Repeat until the working tree is clean
5. **Never push**

Example:

```
feat(`/src/prooflab/data`): add dataset versioning in `versioning.py`

Implemented immutable dataset versioning with checksums and metadata tracking in `versioning.py`.

This guarantees every experiment can be reproduced against the exact data snapshot that was used.
```

---

## 6. Pull Request Summary Rules (Mandatory)

When asked to generate a PR summary, produce a message that can be used as a squash commit.  
Fetch the commits yourself with `git log development..HEAD` (or equivalent).

### Exact Format

```
pr(`/`): <short description with backticked filename>

<first paragraph summarizing what was done>

<second paragraph explaining why it was done>
```

- Type is always `pr`
- Scope is always `` `/` ``
- Exactly two single-line paragraphs
- No bullet points or lists
- Synthesize the entire PR into one coherent summary

---

## 7. Definition of Done for Any Task

A task is only done when:
- The code is implemented
- Tests covering the happy path and important edge cases pass
- The change is committed with a message that strictly follows the rules in Section 5
- The corresponding checkbox in `PROGRESS.md` has been marked as done
- You have not introduced code that belongs to a future milestone

---

## 8. Where to Find the Current Work

All milestones, branch names, tasks, and human review checklists live in:

→ **`PROGRESS.md`**

Start there.  
Read the Global Working Rules, then locate the milestone marked **🔄 IN PROGRESS**. That is the only milestone you are allowed to work on unless the human says otherwise.

---

## 9. When You Are Unsure

If the specification is ambiguous, or if a requested change would violate any rule above:
- Stop
- Explain the conflict clearly
- Wait for human guidance

Do not guess on matters of research integrity, leakage, or live-trading safety.

---

## 10. How to Update PROGRESS.md (Mandatory)

You must keep `PROGRESS.md` accurate as you work. Follow this exact format.

### Marking a single task complete

Change:

```markdown
- [ ] Implement dataset versioning (id, checksum, metadata)
```

To:

```markdown
- [x] Implement dataset versioning (id, checksum, metadata)
```

Do this as soon as the task is finished and committed.  
Then commit the progress update itself, for example:

```
chore(`/`): mark dataset versioning task complete in `PROGRESS.md`

Marked the dataset versioning task as done in PROGRESS.md after the implementation and tests were committed.

Keeps the milestone tracker accurate for the human reviewer.
```

### Milestone status markers

Use exactly these markers in the milestone heading:

- While work is ongoing:

```markdown
### M01 – Data Engine 🔄 IN PROGRESS
```

- When every task is done and you believe the milestone is ready for human review, you may note it, but **do not** mark the whole milestone complete yourself. Only the human changes it to:

```markdown
### M01 – Data Engine ✅ COMPLETE
```

and updates the status line to something like:

```markdown
**Status:** Merged into `development`
```

### Status line

Under the branch name keep a clear status line:

- Active work:

```markdown
**Branch:** `feat/m01-data-engine`  
**Status:** Active – agent is working here
```

- After the human has merged:

```markdown
**Branch:** `feat/m01-data-engine`  
**Status:** Merged into `development`
```

### What you must never do

- Never delete a completed milestone from `PROGRESS.md`
- Never mark a whole milestone as `✅ COMPLETE` yourself — that is the human’s decision after review
- Never edit the Human Review Checklist checkboxes (those are for the human only)
- Never change the Global Working Rules section

### Example of a correctly updated milestone (in progress)

```markdown
### M01 – Data Engine 🔄 IN PROGRESS

**Branch:** `feat/m01-data-engine`  
**Status:** Active – agent is working here

#### Tasks

- [x] Define the canonical OHLCV schema (and optional tick schema)
- [x] Implement Parquet storage and DuckDB access helpers
- [ ] Implement dataset versioning (id, checksum, metadata)
- [ ] Build the data validator that catches the problems listed above
- [ ] Generate a complete health report for every dataset
- [ ] Implement the cleaning pipeline (no silent forward-fill)
- [ ] Write unit tests using deliberately dirty synthetic data

#### Human Review Checklist
- [ ] Validator rejects all major classes of bad data
- [ ] Cleaning never introduces future information
- [ ] Datasets are immutable once versioned
- [ ] Health report contains the required fields
- [ ] Timestamps are timezone-aware and stored in UTC
- [ ] No feature or label logic has been introduced
```

Follow this pattern exactly.

---

**End of AGENTS.md**  
Now open `PROGRESS.md` and begin only the milestone marked 🔄 IN PROGRESS.
