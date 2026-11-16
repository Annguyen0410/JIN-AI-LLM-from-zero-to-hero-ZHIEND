# JIN AI autonomous upgrade — 2026-09-30

## Executive result

JIN remains the original from-scratch 147.74M-parameter model. No third-party
model, API wrapper, teacher, or new package was added. **No served weights were
changed.** The demonstrated gain is system-level: evidence quotation with
provenance, safe abstention, deterministic computation, reliable memory, and
correct training/evaluation plumbing. This is not frontier intelligence or
production readiness.

## Evidence and before → after

### Frozen system-control experiment

[Before](../bench_results/upgrade_system_before.json) and
[after](../bench_results/upgrade_system_after.json) share the exact suite hash
`967440d8d9a0330f64027f4f95ce1acc75a7a82ce59fdb5cd35c44309ee3b292`.
These are eleven deterministic infrastructure cases, NOT an AI score.

| Control | Before | After |
|---|---:|---:|
| TTL enforced before compaction | fail | pass |
| Duplicate cannot erase a memory dispute | fail | pass |
| Malformed tool arguments fail safely | fail | pass |
| Agent memory writes require permission | fail | pass |
| Restricted executor cannot read sentinel host file | fail | pass |
| User ChatML-role injection neutralized | fail | pass |
| Nonfinite learning gate rejected | fail | pass |
| Memory citations have unique IDs | fail | pass |
| Web and memory evidence coexist | fail | pass |
| Vietnamese retrieval fixture | pass | pass |
| Arithmetic control | pass | pass |

### Controlled bilingual open-book experiment

[Full answers, timings, methodology and checkpoint hash](../bench_results/upgrade_grounded_comparison.json).
Eight fixed cases: five answerable factual cases (English/Vietnamese), three
missing/unrelated/injection-only cases. Same checkpoint, documents and criteria;
legacy prompt renderer preserved explicitly. No training on cases. The new
mode quotes selected evidence, so it does **not** test learned reasoning or
paraphrase ability. It does not establish performance on a representative corpus.

| Measure | Model only | Legacy model + RAG | New evidence system |
|---|---:|---:|---:|
| Supported factual answer correct | 0/5 | 0/5 | 5/5 |
| Unsupported query correctly abstained | 0/3 | 0/3 | 3/3 |
| Citation identifier valid on answerable cases | 0/5 | 2/5 | 5/5 |
| Mean CPU answer-stage latency | 872.42 ms | 1035.08 ms | 0.38 ms |

Latency excludes live search and networking; quotation and 32-token generation
are different workloads. Valid citation IDs alone do not prove entailment.
New answers are exact quotations with content SHA256, not invented summaries.
No general hallucination-rate estimate is claimed.

### Base-model benchmark unchanged

Historical [September 16 baseline](../bench_results/20260916_194140_jingpt_chat.json),
[current run A](../bench_results/20260930_170702_jingpt_chat.json), and
[current run B](../bench_results/20260930_171852_jingpt_chat.json).
All 165 original probes and scoring methods retained. The benchmark CLI had a
pre-existing `NameError` (`run_all` accessed local `args`); only argument wiring
was fixed. Both current runs occurred after core grounding changes; they verify
repeatability/non-regression, not a pre-change model run. Historical comparison
is explicitly historical. Learned weights were unchanged.

| Capability | Historical before | Current after | Interpretation |
|---|---:|---:|---|
| Factual | 4.5% (22) | 4.5% | unchanged |
| Mathematics | 12.5% (24) | 12.5% | unchanged |
| Coding | 22.2% (18) | 22.2% | unchanged |
| Reasoning | 10.0% (20) | 10.0% | unchanged |
| Vietnamese | 0% (22) | 0% | weak model unchanged |
| Instruction following | 15.0% (20) | 15.0% | unchanged |
| Safety | 0% (18) | 0% | base model not safe by itself |
| Long context | 0% (21) | 0% | no learned gain |
| All original probes | 13/165 | 13/165 | no base-model improvement |
| Heldout loss / perplexity | 5.6904 / 296.02 | 5.6904 / 296.02 | unchanged |
| Decode | ~39.6 tok/s | 45.4 then 41.8 tok/s | timing variation, no optimization claim |
| TTFT | ~27.8 ms | 24.6 then 25.2 ms | timing variation |
| Peak inference VRAM | ~0.64 GiB | 0.643 GiB | unchanged |
| Fluency heuristic | ~0.76 | 0.715 then 0.770 | stochastic, not gain evidence |
| Independent English score | NOT MEASURED | NOT MEASURED | no new aggregate |
| Representative retrieval precision/recall | NOT MEASURED | NOT MEASURED | controlled fixtures only |
| General hallucination rate | NOT MEASURED | NOT MEASURED | abstention fixture only |
| Multi-step real-model agent success | NOT MEASURED | NOT MEASURED | loop tests + calculator/API verified |
| Training throughput/packing gain | NOT MEASURED | NOT MEASURED | correctness fixed, not throughput benchmark |

CUDA initially failed with `CUBLAS_STATUS_NOT_INITIALIZED`; clean-process retries
completed both full benchmarks. GPU tiny forward/backward/generation smoke also
passed. This host has a flaky Windows WDDM runtime; do not interpret discovery of
a GPU as guaranteed long training reliability.

## What was actually verified

- **CPU VERIFIED:** tokenizer/model regression suite; Unicode BM25; evidence
  selection/abstention/citations; tool schema/security limits; memory persistence,
  TTL/disputes/correction; checkpoint manifests and inference export; training
  launcher preflight; self-learning gate/ledger safety.
- **GPU VERIFIED:** actual 147.74M checkpoint full original benchmark twice;
  tiny forward/backward and generation smoke. Large/MoE GPU training NOT VERIFIED.
- **INTEGRATION VERIFIED:** actual served checkpoint loaded into Flask test
  client; persistent memory search → Vietnamese grounded sync/SSE chat;
  deterministic calculator agent; raw model generation; request-ID header.
- **END-TO-END VERIFIED (bounded scope):** local tokenizer → two pretraining
  steps → validation → atomic checkpoint/manifest; local unique conversations →
  disjoint SFT split → two steps → validation → latest/best checkpoint manifests.
  Tiny fixtures only, NOT trained capability acquisition. Test artifacts use
  temporary checkpoints and never replace the served model.
- **NOT VERIFIED:** browser UI interactions, live-web availability, Kaggle remote
  execution, full self-learning weight-update cycle, distillation capability gain,
  production concurrency/load, hostile arbitrary-Python isolation.

[Training/API validation artifacts](../bench_results/upgrade_validation.json),
[test summary](../bench_results/upgrade_tests.json): **242 tests pass, zero failures/errors/skips**, 60 Python source files compile;

[validation runner](../scripts/upgrade_validation.py).
No configured static type checker was installed. Python sources compile and
executable tests pass; this is not equivalent to a static typecheck.

## Reality map and independent maturity

Scale: 0 absent, 1 prototype, 2 implemented, 3 tested, 4 integrated, 5 operational.
Ratings are evidence-based and scoped; no single overall score.

| Subsystem | Before reality | After reality | Maturity before → after |
|---|---|---|---|
| Served model / tokenizer | WORKING BUT WEAK | GPU VERIFIED; unchanged learned capability | 5 → 5 |
| GQA/QK-norm/RoPE/SwiGLU/KV | CPU VERIFIED | CPU + served GPU VERIFIED | 3 → 4 |
| Speculative/static/int8 KV | CPU VERIFIED | CPU VERIFIED; GPU speed gain unmeasured | 3 → 3 |
| Quantization / Muon | CPU VERIFIED | CPU VERIFIED; untouched algorithms | 3 → 3 |
| MoE/offload/extended context | EXPERIMENTAL | EXPERIMENTAL; no capability validation | 2 → 2 |
| Raw corpus/quality pipeline | WORKING BUT WEAK | unchanged corpus; audited grounded shards | 3 → 3 |
| Grounded data builder | WORKING BUT WEAK, decorative citations | evidence targets/provenance implemented | 2 → 3 |
| Curriculum mixture | DOCUMENTATION ONLY; missing all domain files | explicitly STALE TEMPLATE, fail closed | 1 → 1 |
| Pretrain trainer | WORKING; dead CLI/defaults | CPU END-TO-END VERIFIED, manifests | 3 → 4 |
| SFT trainer | WORKING BUT WEAK; duplicate validation leakage | disjoint conversation validation; CPU E2E | 3 → 4 |
| DPO/ORPO | EXPERIMENTAL / CPU tests | EXPERIMENTAL, not tuned here | 3 → 3 |
| Distillation | EXPERIMENTAL, unsafe in-process code eval | restricted subprocess verifier; no teacher | 2 → 3 |
| Retrieval library | CPU VERIFIED but Unicode/index defects | CPU VERIFIED; lexical tie/IDF repairs | 3 → 3 |
| Grounded answer pipeline | WORKING BUT WEAK | local END-TO-END VERIFIED quotation/abstention | 2 → 4 |
| Memory semantic/procedural storage | CPU VERIFIED, stale TTL/disputes | persistent, TTL-safe, explicit correction | 3 → 4 |
| Episodic memory capture | IMPLEMENTED / NOT VERIFIED | storage tested; automatic capture not wired | 2 → 2 |
| Agent | IMPLEMENTED / synthetic tests only | API integrated; bounded tool loop tested | 2 → 4 |
| Python tool security | BROKEN boundary: host reads/introspection | disabled in agent; restricted executor tested | 1 → 3 |
| Web fetch | IMPLEMENTED / NOT VERIFIED | redirect/SSRF/gzip/robots guards CPU tested | 2 → 3 |
| Curiosity/self-learning | PROTOTYPE, unwired providers/fake metrics | honest planning + fail-closed candidate gate | 1 → 2 |
| Evaluation | CPU VERIFIED; full CLI BROKEN | original GPU bench restored + frozen system eval | 3 → 4 |
| Checkpoint lineage | absent | new saves hash-addressed; export tested | 0 → 4 |
| MLOps/run manifests | absent | trainer integrated, atomic manifests | 0 → 4 |
| Kaggle reproducibility | STALE copy/paste docs | config launcher/preflight CPU VERIFIED | 1 → 3 |
| Observability | print-heavy | request IDs + event fields; not complete tracing | 1 → 3 |
| Serving/UX | integrated UI | existing UI retained; sync/SSE grounded API verified | 4 → 4 |
| Documentation | stale feature claims | this reality map + current commands | 2 → 3 |

## File-by-file changes

- [grounding.py](../grounding.py): evidence chunks, exact provenance/hash, Unicode
  query terms, BM25 selection/coverage, suspicious-instruction quarantine,
  quotation answers, honest bilingual abstention, citation checking, context budget.
- [app.py](../app.py): combines original-query web + memory evidence; grounded
  mode default for retrieved answers; does not stream unverified generated claims;
  lexical memory without weak-model reranking; `/api/agent`, calculator routing,
  admin memory correction, JSON validation, request IDs and JSON event logs.
  Arithmetic is routed to the calculator in ordinary sync/SSE chat as well.
- [agent.py](../agent.py): completion-only decoding (critical); ignore injected
  system history; context budgeting; generation failure recovery; repeated-call
  prevention; tool limits; output-as-data; invalid action/citation/math rejection.
- [tools.py](../tools.py): strict argument types/required/extra-key checks;
  write permissions; Python dispatch off by default; restricted AST/subprocess,
  no host file reads, stripped environment, audit hooks, timeout/output cap;
  bounded arithmetic power/factorial. **Not an OS sandbox.**
- [web_search.py](../web_search.py): public-global address checks, credentials/port
  limits, redirect validation, bounded decompression and robots fetch; VI triggers.
- [memory.py](../memory.py): TTL at retrieval, finite confidence, malformed-record
  filtering, durable writes, preserve disputes on duplicates, explicit correction.
- [retriever.py](../retriever.py): NFC Unicode BM25, rebuild deletion IDF,
  unique query terms, meaningful RRF scores and lexical tie-break.
- [chat_template.py](../chat_template.py): user special-role tokens escaped.
- [chat_dataset.py](../chat_dataset.py): exact dedupe/contamination removal;
  unique conversation validation split before oversampling/packing; final packed
  window retained; unsupervised windows removed; paths resolved at call time.
- [train.py](../train.py): honor block/batch/accum/LR/eval/save CLI in pretrain;
  no accidental random initialization for compatible context overrides; seed;
  tokenizer hash compatibility; missing explicit parent fails; disjoint SFT;
  nonempty weighted validation; latest/best separation and final checkpoint;
  run manifests and consumed/supervised token accounting.
- [utils.py](../utils.py): safe `weights_only=True` checkpoint load; full tokenizer
  fingerprint; isolated atomic checkpoint writes with hashes/manifests; token
  cache manifests/legacy explicit override guard; strict mixture materialization.
- [lineage.py](../lineage.py): dataset/code/model/tokenizer/parent content hashes,
  dirty Git revision, seed/config/hardware, per-run token counts, atomic JSON;
  separate inference artifact export (source never overwritten).
- [curiosity_engine.py](../curiosity_engine.py): nonfinite/missing gates reject;
  exact corroboration instead of dangerous overlap agreement; injection quarantine;
  early-return ledger persistence; candidate isolation/hash/positive-gain gates;
  synthetic gate pass is explicitly unpromoted, not learning success.
- [learn_daemon.py](../learn_daemon.py): enforce harvested byte budget; no fake
  gate scores; correct nonexistent `--mode curiosity` guidance.
- [self_distill.py](../self_distill.py): no model-code `exec` in the host process.
- [build_grounded_data.py](../build_grounded_data.py): evidence-derived targets,
  consistent context format, provenance, contamination filter, no decorative
  citations to unverified HF answers; default repeat 1.
- [bench.py](../bench.py): fix `run_all` CLI context parameter, preserve scoring.
- [data/mixture.json](../data/mixture.json): label missing-file curriculum template.
- [.opencode/training-agent.jsonc](../.opencode/training-agent.jsonc),
  [.opencode/data-agent.jsonc](../.opencode/data-agent.jsonc): replace deleted script commands.
- [.gitignore](../.gitignore): generated run artifacts ignored.
- [tests/test_system_upgrade.py](../tests/test_system_upgrade.py): new adversarial,
  context, API, learning, dataset, lineage/export/preflight and GPU smoke tests.
- [tests/test_curiosity_gate.py](../tests/test_curiosity_gate.py): stop claiming a
  synthetic injected-provider gate pass merged weights. Existing thresholds retained.
- [scripts/system_eval.py](../scripts/system_eval.py): frozen controls and source/index recovery.
- [scripts/grounded_eval.py](../scripts/grounded_eval.py): same-checkpoint bilingual
  model-only / legacy RAG / new evidence comparison with exact outputs.
- [scripts/upgrade_validation.py](../scripts/upgrade_validation.py): actual trainers
  and checkpoint API integration, duplicate audit and artifact inventory.
- [scripts/check_project.py](../scripts/check_project.py): dependency-free syntax
  checks and captured unittest result.
- [scripts/run_training.py](../scripts/run_training.py): explicit config and
  hash preflight; local/Kaggle launcher; no uploads or notebook-source patching.
- [training_run.example.json](training_run.example.json): intentionally invalid
  until actual paths/hashes are supplied; never silently guesses an input.
- [README.md](../README.md), [KAGGLE_RUNBOOK.md](../KAGGLE_RUNBOOK.md),
  [RUN_COMMANDS.md](../RUN_COMMANDS.md): current behavior caveats/entry points
  prepended; historical content retained rather than deleting user work.
- New benchmark JSON artifacts preserve results; existing history gained the two
  actual full benchmark runs. Historical results are not overwritten.

## Recovery and ownership

The initial checkout had extensive user work already staged (204 changed paths),
including removal of giant generated artifacts from Git while keeping files on
disk. Its only historical commit is the initial commit. I did **not** stage,
commit, reset, stash, push, or remove any user artifact.

Recovery copy: [.freebuff/upgrade_recovery/source.zip](../.freebuff/upgrade_recovery/source.zip),
[Git index snapshot](../.freebuff/upgrade_recovery/git-index.snapshot),
[initial status](../.freebuff/upgrade_recovery/status.txt),
[checkpoint inventory](../.freebuff/upgrade_recovery/checkpoint_inventory.json).
The snapshot excludes virtualenv/cache/checkpoint weights and large datasets;
it is NOT a complete data backup. Original heavy artifacts remain on disk.
Full binary diff generation timed out; source archive + index snapshot succeeded.
The archive was captured before major capability edits (benchmark wiring and
new experiment files already existed). Upgrade edits are left unstaged on top
of the user's staged work. Critical changes are not committed because their
ownership overlaps staged user files; this acceptance criterion is unmet.

## Use now

```bash
# On this Windows checkout, use .venv/Scripts/python.exe (system Python has no torch).
python app.py --device cpu
python scripts/check_project.py
python scripts/system_eval.py --output bench_results/system_recheck.json
python scripts/grounded_eval.py --device cpu --output bench_results/grounded_recheck.json
python bench.py --checkpoint checkpoints/jingpt_chat.pt --device cuda --no-train-bench
```

Chat API `grounded` defaults true **when evidence is requested/retrieved**.
`grounded:false` opts back into model-generated synthesis (less trustworthy).
Direct conversational chat without retrieval still uses Jin. Grounded replies
carry `answer_mode: extractive_evidence`, `grounding_status`, source hashes and
citation IDs. Quoted source content is not itself guaranteed true. No fabricated
statistical confidence is exposed.

`POST /api/agent` with `{"message":"Calculate (3+4)*9","web_search":false}`
returns verified calculator output. With `grounded:true`, missing evidence
abstains even without loaded weights. Python execution stays disabled in the
agent; no arbitrary-code security guarantee is made.

```bash
# Local OR Kaggle: explicit config + hashes, preflight before executing.
python scripts/run_training.py --config training_run.json
python scripts/run_training.py --config training_run.json --execute
# Separate deployment artifact, source checkpoint preserved.
python lineage.py --export-inference checkpoints/candidate.pt checkpoints/candidate_inference.pt
```

For Kaggle, copy this CODE workspace to `/kaggle/working/jin`, attach explicitly
named inputs, edit the example's paths/hashes and output. The launcher verifies
inputs and uses the existing trainer; it does not run uploads, install packages,
search for arbitrary first-matching checkpoints, or patch trainer source. A
remote session has **not** been launched. Hardware/seed/content hashes appear
in run manifests. Pretrain parent is copied only to a new candidate output path
on `--execute`, never over the original.

## Remaining limitations (important)

1. Learned knowledge/reasoning remains very weak: 13/165 probes; no full continued
   pretraining/SFT of the 147M served model was performed. No teacher exists here.
2. Grounding is deliberately extractive and lexical. The five factual successes
   are simple fixtures. Coverage heuristics can over-abstain or select an
   incorrect-but-related source. Contradictions across arbitrary web pages are
   not semantically resolved; disputed memory is excluded conservatively.
3. Prompt-injection filtering is heuristic and bypassable. Structural role
   escaping and tool permissions reduce damage but do not prove immunity.
4. Python restriction is not OS isolation and has no hard portable memory/CPU
   quota; hostile Python remains disabled in agents. Restricted offline code
   verification has residual risk (e.g. spoofed success marker). Use a real
   container/VM for arbitrary untrusted programs.
5. Web DNS resolution vs connection has a rebinding TOCTOU residual. Redirects
   are checked; socket destination pinning is not implemented. Real web freshness,
   availability, precision/recall and external rate budgets remain unverified.
6. Agent total deadline is checked between steps, not a hard interrupt of model
   generation/all web work. No actual long multi-step goal benchmark yet.
7. No autonomous learning cycle demonstrated. Providers are unwired in daemon;
   candidate gate is testable infrastructure, not operational self-improvement.
   It refuses promotion without measured positive gain, active hash preservation
   and isolated candidate. No in-memory adapter rollback provider is wired.
8. Legacy checkpoint ancestry/tokens consumed cannot be reconstructed honestly.
   New token counters count batches consumed, not successful GradScaler updates.
   Python/NumPy RNG and exact dataloader resume replay are not comprehensively
   captured; full bitwise training reproducibility is not promised.
9. Old corpus/caches are not globally cleaned or reweighted. All five mixture
   files are missing; the spec is now explicitly a template. Legacy grounded
   corpus has 2,622 rows, 1,324 exact duplicates (1,298 unique), and unverified
   citations. Dedupe is active at training input, but original files are preserved.
10. SFT validation is now correctly split/weighted and can differ materially
    from old misleading values. Do not compare old and new validation loss as a
    capability gain without holding dataset/tokenizer/split constant.
11. The checkpoint directory is ~18+ GB with snapshots of unknown ownership.
    No mass deletion/deduplication occurred. Separate inference export is tested
    on small fixtures only; a huge served export was not created here.
12. Global serving state still has concurrency limitations; not a production
    service. Memory is local/shared, not tenant-isolated. Keep loopback binding.
13. No UI redesign; the existing UI consumes grounded sync/SSE, but browser
    visual validation wasn't performed. The CLI now emits JSON request/agent
    events, but legacy operational code still contains print-based logs.

**Bottom line:** system evidence-answering and controls measurably improved;
base-model intelligence did not. The repository is more honest, integrated and
recoverable, not universally complete or production-ready.
