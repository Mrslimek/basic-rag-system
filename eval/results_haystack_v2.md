# Haystack v2 eval — external qwen3.8-flash, three modes (2026-10-30)

## Context

First measurement on the v2 (Haystack-native) graph stack after the wave-7 rebuild: `python -m app.graph_rag sync` with the external OpenAI-compatible endpoint (`qwen3.8-flash` at the Aliyun MaaS base URL) rebuilt the live Neo4j graph from the Postgres chunk table — 52 chunks, 52 paid extraction calls, 0 errors, all httpx requests logged against the external host (0 to the old local `127.0.0.1:8080`). The contract fix-threshold (>5 chunks without `MENTIONS`) failed at 23, so ONE selective refill pass ran extraction + the idempotent writer for exactly those 23 chunks (23 paid calls, no wipe): 4 of them gained edges (Entity 59→67, relations 79→86, MENTIONS 118→129) and 19 still have no `MENTIONS` — mostly the structural `# X Specification` (idx 0) and `## Endpoints` (idx 2) section heads, plus content chunks `billing#2`, `billing#3`, `search#1`, `search#3`, `search#4`; the model returned 0 triplets for them on both passes. Per the completion rule no third pass was run.

Live stack at run time: Neo4j 52 `Chunk` / 67 `Entity` (SERVICE 30, ENDPOINT 18, QUEUE 14, DATABASE 5), 86 non-`MENTIONS` typed relations, 129 `MENTIONS` (Entity→Chunk), 19 chunks with zero inbound `MENTIONS` (analytics#0/#2, auth#0/#2, billing#2/#3, ledger#0/#2, notification#0/#2, payment_gateway#0/#2, search#0/#1/#2/#3/#4, user_profile#0/#2); Qdrant `doc_chunks` 52 points; Postgres 52 chunk rows across all 8 services. Synthesis for all three modes served by external `qwen3.8-flash`. Eval protocol: direct library calls (no uvicorn), 1 warmup (Q12 vector, 8.8 s, no fallback prefix — provenance confirmed) + Q1..Q20 × modes vector→hybrid→graph, top_k=4, no service filter, no retries. Calls spent this wave: 52 rebuild + 23 refill + 61 eval (60 measured + 1 warmup) = 136. Flakes: **0** (60/60 ok, no exceptions, no retrieval-fallback replies). Grading identical to r3/r4 (case-insensitive substring; expected services among retrieved sources, 3 frozen literals per question in the reply).

## Comparison vs Round 4 (same 5 multi-hop questions Q12/Q14/Q15/Q17/Q20)

| Q | expected sources | vector src | vector terms | graph src | graph terms | hybrid src | hybrid terms | r4 vector (src/terms) | r4 graph (src/terms) | r4 hybrid (src/terms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q12 | notification, auth | 50% | 67% | 0% | 33% | 50% | 67% | 50% / 67% | 50% / 33% | 50% / 67% |
| Q14 | ledger, analytics, billing | 33% | 67% | 67% | 0% | 33% | 33% | 33% / 33% | 33% / 0% | 33% / 33% |
| Q15 | user_profile, search | 100% | 100% | 0% | 33% | 100% | 100% | 100% / 100% | 50% / 100% | 100% / 100% |
| Q17 | payment_gateway, notification, user_profile | 67% | 67% | 67% | 67% | 67% | 67% | 67% / 67% | 33% / 33% | 67% / 67% |
| Q20 | notification, user_profile, auth | 67% | 67% | 67% | 67% | 67% | 67% | 67% / 67% | 33% / 33% | 67% / 67% |
| **Avg (5 multi-hop)** | — | **62% (8/13)** | **73% (11/15)** | **46% (6/13)** | **40% (6/15)** | **62% (8/13)** | **67% (10/15)** | 62% (8/13) / 67% (10/15) | 38% (5/13) / 40% (6/15) | 62% (8/13) / 67% (10/15) |

Vector and hybrid sources are retrieval-bound and unchanged from r4 (8/13 on both, identical per-question values); vector terms gained one literal (Q14 33%→67%) to resampled synthesis at temperature 0.2 — the vector branch never touches the graph; hybrid terms tie r4 exactly (10/15). Graph gained sources (5/13→6/13) and tied terms (6/15) — with the schema caveat below.

## Full 20-question matrix (v2, new coverage — r4 measured only the 5 above)

| Q | type | expected sources | vec src | vec terms | hyb src | hyb terms | gr src | gr terms | hyb facts /4 |
|---|---|---|---|---|---|---|---|---|---|
| Q1 | single | auth | 100% | 67% | 100% | 67% | 0% | 0% | 3 |
| Q2 | single | billing | 100% | 100% | 100% | 100% | 0% | 33% | 4 |
| Q3 | single | notification | 100% | 67% | 100% | 67% | 0% | 33% | 3 |
| Q4 | single | payment_gateway | 100% | 67% | 100% | 67% | 0% | 67% | 4 |
| Q5 | single | user_profile | 100% | 100% | 100% | 67% | 0% | 67% | 4 |
| Q6 | single | search | 100% | 100% | 100% | 100% | 0% | 33% | 1 |
| Q7 | single | analytics | 100% | 67% | 100% | 100% | 100% | 67% | 4 |
| Q8 | single | ledger | 100% | 100% | 100% | 100% | 100% | 100% | 4 |
| Q9 | single | user_profile | 100% | 100% | 100% | 100% | 0% | 67% | 3 |
| Q10 | single | user_profile | 100% | 100% | 100% | 100% | 100% | 100% | 4 |
| Q11 | multi | billing, payment_gateway, ledger | 67% | 67% | 67% | 67% | 67% | 33% | 4 |
| Q12 | multi | notification, auth | 50% | 67% | 50% | 67% | 0% | 33% | 4 |
| Q13 | multi | billing, payment_gateway, ledger | 67% | 67% | 67% | 67% | 67% | 33% | 4 |
| Q14 | multi | ledger, analytics, billing | 33% | 67% | 33% | 33% | 67% | 0% | 4 |
| Q15 | multi | user_profile, search | 100% | 100% | 100% | 100% | 0% | 33% | 2 |
| Q16 | multi | user_profile, notification, auth | 33% | 100% | 33% | 100% | 67% | 100% | 3 |
| Q17 | multi | payment_gateway, notification, user_profile | 67% | 67% | 67% | 67% | 67% | 67% | 4 |
| Q18 | multi | ledger, analytics | 50% | 100% | 50% | 100% | 100% | 100% | 4 |
| Q19 | multi | billing, payment_gateway, ledger | 33% | 33% | 33% | 67% | 33% | 33% | 4 |
| Q20 | multi | notification, user_profile, auth | 67% | 67% | 67% | 67% | 67% | 67% | 4 |
| **Avg (20)** | — | — | **68% (25/37)** | **80% (48/60)** | **68% (25/37)** | **80% (48/60)** | **49% (18/37)** | **53% (32/60)** | 71/80 |

Sources returned per question: vector 4 / hybrid 4 / graph 4 — the v2 graph branch now fills top_k on every run (r4's v1 graph starved at 2-3 nodes). Hybrid graph-facts coverage is 71/80 retrieved chunks (r4: 5/20 on its 5-question subset; on the same subset v2 scores Q12 4/4, Q14 4/4, Q15 2/4, Q17 4/4, Q20 4/4) — the rebuilt 129-edge `MENTIONS` spread reaches nearly every retrieved chunk.

## Caveats

- **Schema drift v1→v2 makes the graph column not directly comparable to r4's graph row.** r4 ran the LlamaIndex v1 graph: entity label `__Entity__`, `MENTIONS` directed chunk→entity, and the graph retriever chained an `LLMSynonymRetriever` that spent an external LLM call per query to rewrite the question against the graph. v2 is Haystack-native: `Entity`/`Chunk` labels, `MENTIONS` directed entity→chunk, and `EntityGraphRetriever` (embedding-KNN + relation expansion) runs with **no LLM at query time**. Different candidate-generation semantics — treat graph deltas as character change, not pure quality change. The all-20 rows have no r4 counterpart at all (r4 measured only Q12/Q14/Q15/Q17/Q20).
- **19 zero-`MENTIONS` chunks** (structural section heads + billing#2/#3, search#1/#3/#4) cannot contribute graph facts to hybrid cells that retrieve them and are unreachable from the graph seed expansion paths anchored on entity names; visible effects: Q6 facts coverage 1/4 and single-hop graph src 0% on service-head topics (Q1-Q6, Q9). The selective refill pass recovered 4 chunks (auth#5, notification#3, payment_gateway#4, user_profile#3) but the model returned 0 triplets for the rest on both attempts.
- Grading literals and expected-service sets are the frozen r3/r4 ones; the `src_x_y` fractions above are hit-slots over the 5/20 expected services and 3×5/3×20 literals, comparable to r4's "sources /13, terms /15" bookkeeping.

## Conclusion

- **Vector keeps r4-level sources and improves terms on the subset**: 62% (8/13) sources — identical to r4 — and 73% (11/15) terms vs 67% (10/15); on the new full-20 baseline it sits at 68% / 80%, with single-hop sources a clean 10/10. Multi-hop sources remain the ceiling (Q14 33%, Q16 33%, Q19 33%): retrieval, not synthesis, bounds them.
- **Hybrid matches r4 exactly on the subset** (62% / 67%, cell-for-cell identical) and matches vector on all-20 (68% / 80%) — the graph-facts block (71/80 coverage) neither lifts nor poisons the frozen metrics this round; per the r4 recommendation, `retrieve_mode="vector"` stays the safe default since hybrid pays a Neo4j dependency for no measured gain.
- **Graph branch: sources up 38%→46% (5/13→6/13), terms flat at 40% (6/15)** on the subset — but with the schema caveat: v2 graph retrieval no longer under-fills (4/4 sources every run) and wins outright where entity anchoring beats pure vectors (Q14/Q16/Q18 src, Q18/Q16 terms 100%), while losing the r4 graph's few wins (Q15 terms 100%→33%, Q12 src 50%→0%). On all-20 it trails vector/hybrid (49% / 53%) with single-hop service-head questions at 0% src — the zero-`MENTIONS` heads are the named cause.
- **Operational**: 0 flakes in 136 paid calls (52 rebuild + 23 refill + 61 eval), no quota/auth errors, no local-endpoint leakage; the pre-wipe graph was backed up (`backups/neo4j_20261030_020910.tar.gz`).
