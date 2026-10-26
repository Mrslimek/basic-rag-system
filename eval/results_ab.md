# A/B Results — vector vs graph retrieval (`aquery`, top_k=4)

Wave 3 measurement over the 20 questions in [questions.md](questions.md).
For each question both branches of `app.llama_rag.aquery(question, top_k=4, retrieve_mode="vector"|"graph")` were run sequentially against the live stack (Neo4j graph: 102 Chunk / 78 Entity; Qdrant `doc_chunks`: 102 points; llama.cpp Qwen2.5-Coder-1.5B).

Metrics per cell (case-insensitive substring grading, see questions.md for expected answers):
- **sources_hit** — fraction of the question's expected source services present among the retrieved chunks' `service_name` metadata.
- **key_terms_hit** — fraction of the expected answer's distinctive key tokens (literal strings, service names, cluster/bucket ids) present in the LLM reply text.

No LLM flakes occurred (0 retries, 0 retrieval-fallback replies across all 40 runs).

| Q | type | expected sources | vector sources_hit | vector key_terms_hit | graph sources_hit | graph key_terms_hit | graph sources returned |
|---|------|------------------|--------------------|----------------------|-------------------|---------------------|------------------------|
| Q1 | single | auth | 100% | 100% | 100% | 0% | analytics#12, auth#7, payment_gateway#11, search#12, user_profile#11 |
| Q2 | single | billing | 100% | 100% | 0% | 0% | payment_gateway#11 |
| Q3 | single | notification | 100% | 100% | 100% | 0% | notification#10, notification#11 |
| Q4 | single | payment_gateway | 100% | 0% | 100% | 0% | payment_gateway#11, notification#10, ledger#10, search#12 |
| Q5 | single | user_profile | 100% | 100% | 100% | 100% | user_profile#11, notification#11, search#12, search#11, notification#10, user_profile#10 |
| Q6 | single | search | 100% | 100% | 100% | 100% | search#11, search#12, notification#10, search#5 |
| Q7 | single | analytics | 100% | 0% | 100% | 0% | analytics#12 |
| Q8 | single | ledger | 100% | 100% | 100% | 100% | search#10, search#12, ledger#10 |
| Q9 | single | user_profile | 100% | 100% | 100% | 100% | payment_gateway#11, search#12, user_profile#11, search#11, notification#10, user_profile#10 |
| Q10 | single | user_profile | 100% | 100% | 100% | 100% | user_profile#11, user_profile#10 |
| Q11 | multi | billing, payment_gateway, ledger | 67% | 50% | 33% | 0% | notification#11, payment_gateway#11 |
| Q12 | multi | notification, auth | 50% | 100% | 50% | 50% | notification#11, notification#10 |
| Q13 | multi | billing, payment_gateway, ledger | 67% | 0% | 33% | 0% | payment_gateway#11, analytics#12 |
| Q14 | multi | ledger, analytics, billing | 33% | 0% | 33% | 0% | analytics#12 |
| Q15 | multi | user_profile, search | 50% | 100% | 50% | 100% | search#12, search#11, search#5 |
| Q16 | multi | user_profile, notification, auth | 67% | 100% | 33% | 100% | user_profile#11, user_profile#10 |
| Q17 | multi | payment_gateway, notification, user_profile | 67% | 100% | 33% | 50% | payment_gateway#11 |
| Q18 | multi | ledger, analytics | 50% | 100% | 50% | 0% | search#10, search#11, ledger#10, search#12 |
| Q19 | multi | billing, payment_gateway, ledger | 33% | 0% | 33% | 33% | payment_gateway#11 |
| Q20 | multi | notification, user_profile, auth | 67% | 50% | 33% | 0% | notification#11 |
| **Single-hop avg (Q1-Q10)** | — | — | **100%** | **80%** | **90%** | **50%** | — |
| **Multi-hop avg (Q11-Q20)** | — | — | **55%** | **60%** | **38%** | **33%** | — |
| **Overall avg (all 20)** | — | — | **78%** | **70%** | **64%** | **42%** | — |

## Conclusion

- **Vector wins everywhere**: overall sources_hit 78% vs 64% and key_terms_hit 70% vs 42%.
- **Single-hop**: vector takes sources_hit 100% vs graph 90%; the graph branch missed the owning service outright on Q2 (Billing topics) and returned only a single chunk on Q7.
- **Multi-hop**: contrary to the usual GraphRAG expectation, vector still wins both metrics (55% vs 38% sources, 60% vs 33% key terms). Vector filled all 4 slots on every question; the graph retriever returned ≤2 source nodes on 14 of 20 runs (e.g. multi-hop Q13/Q17/Q19/Q20, single-hop Q2/Q7), so hop coverage never materialized into retrieved context; its single clear win is Q19, where the typed-edge listing produced a better dependency chain than vector's billing-only chunks.
- **Tie notes**: many single-hop cells tie at 100%/100% because the 1.5B model is terse and sometimes hallucinates the literal even when the right chunk was retrieved (e.g. Q4 `payment_intent_id`, Q7 `analytics.metrics.live` in *both* modes) — key_terms_hit there measures answer fidelity, not retrieval.
- **Recommendation**: keep `retrieve_mode="vector"` as the default. The graph branch is not yet worth its extra LLM round-trips (LLMSynonymRetriever); revisit after improving graph retriever text recall (e.g. raise sub-retriever limit / always merge chunk neighbors into top_k text nodes).

## Round 2 — MarkdownNodeParser section chunking + model/limit fixes (2026-10-24)

Context: full rebuild from scratch — chunking switched to LlamaIndex `MarkdownNodeParser` (102 → 52 chunks, headings kept inside chunk text), Postgres/Qdrant/Neo4j wiped and rebuilt. Graph stability fixes along the way: extractor `max_tokens` cap (kills the 1.5B repetition loops), client timeout 60s→300s, `raise_on_error=False` (a broken chunk skips instead of aborting sync), and `excluded_llm_metadata_keys` (service_id/doc_file_path no longer leak into the extraction prompt, where the model was extracting UUIDs and repo paths as entities).

Sample: 5 multi-hop questions (Q12, Q14, Q15, Q17, Q20), both modes, top_k=4. Grading identical to Round 1 (sources_hit vs expected services; key_terms_hit = case-insensitive literals from Expected Answer). Scores are % of achievable hits (sources /13, terms /15). **Not directly comparable to Round 1: different chunking, different chunk count.**

| Q | expected sources | v1.5B vec src | v1.5B vec terms | v1.5B graph src | v1.5B graph terms | v8B vec src | v8B vec terms | v8B graph src | v8B graph terms |
|---|---|---|---|---|---|---|---|---|---|
| Q12 | notification, auth | 50% | 33% | 50% | 33% | 50% | 33% | 50% | 33% |
| Q14 | ledger, analytics, billing | 33% | 33% | 33% | 33% | 33% | 33% | 33% | 33% |
| Q15 | user_profile, search | 100% | 100% | 50% | 33% | 100% | 0% | 50% | 100% |
| Q17 | payment_gateway, notification, user_profile | 67% | 33% | 33% | 33% | 67% | 0% | 33% | 0% |
| Q20 | notification, user_profile, auth | 67% | 33% | 67% | 0% | 67% | 0% | 67% | 33% |
| **Sources avg (5 multi-hop)** | — | **62%** | — | **46%** | — | **62%** | — | **46%** | — |
| **Key terms avg (5 multi-hop)** | — | — | **47%** | — | **27%** | — | **13%** | — | **40%** |

Graph after the 8B + prompt fixes: 27 entities, 17 SERVICE / 6 DATABASE / 4 ENDPOINT; 14 semantic edges (6 STORES_IN, 6 USES, 2 PROVIDES) — typed STORES_IN edges the 1.5B never produced, and no UUID/path garbage after the metadata-leak fix.

## Conclusion (round 2)

- **Multi-hop: vector still wins retrieval coverage** (62% vs 46% sources) — same direction as Round 1. The graph retriever under-fills top_k: on Q14/Q17 it returned chunks of the wrong services.
- **8B lifted graph answer quality**: key_terms 27% → 40% (graph), flipping the Round-1 margin — graph replies now often contain the literals when context arrives (Q15: full 100% on `acme-dwh`/S3/avatar via STORES_IN edges), while refusing cleanly when it does not.
- **8B lowered vector answer quality** (47% → 13% terms): verbose "Based on the provided context…" replies paraphrase the literals away; substring grading punishes this. Retrieval (sources) is unchanged — the vector branch never depended on the model.
- **Verdict: draw in our implementation.** Neither mode dominates; graph wins terms on 8B, vector wins sources everywhere. The binding constraint is no longer model size: extraction JSON validity is 0 errors/52. Next levers are entity resolution (Search Microservice vs search-service split nodes), schema coverage (Kafka typed SERVICE not QUEUE; PUBLISHES_TO/AUTHENTICATES_VIA never emitted) and sub-retriever text recall.
- **Default stays `retrieve_mode="vector"`.** Graph branch remains useful as an explicit mode + a live target for the resolution/schema work.

## Round 3 — hybrid mode vs fresh vector, 1.5B llama.cpp (2026-10-24)

Context: the same 5 multi-hop questions as Round 2 (Q12/Q14/Q15/Q17/Q20), top_k=4, default 1.5B model, `retrieve_mode="vector"` then `"hybrid"` per question, one run each, no retries. Live stack at run time: Neo4j 52 Chunk / 26 Entity / 41 relationships; Qdrant `doc_chunks` 52 points (green); Postgres healthy; llama.cpp `llama-server` serving `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (Qwen2.5-Coder-1.5B-Instruct) at 127.0.0.1:8080/v1, ctx 32768, API key `ollama`; backend `uvicorn main:app --port 8000`. `hybrid` = vector-first Qdrant chunks (same top_k/order/scores) with a Neo4j "Graph facts" block appended to each chunk's text (MENTION entities + one-step `X -> REL -> Y` triplets); sources and scores stay the vector ones. Grading identical to Round 2 (case-insensitive substring; sources /13, terms /15). Key-term literals (3/question, re-derived from questions.md expected answers in the Round 2 style — the Q15 set matches the one cited in Round 2's conclusion): Q12 `notification:write`/`jwt`/`auth`; Q14 `ledger.entry.committed`/`billing.subscription.created`/`revenue reports`; Q15 `acme-dwh`/`s3`/`avatar`; Q17 `payment.transaction.processed`/`receipt`/`preferences`; Q20 `notification:write`/`redis-cache-eu-1`/`preferences`.

| Q | expected sources | vector src | vector terms | hybrid src | hybrid terms | sources returned (vec / hyb) |
|---|---|---|---|---|---|---|
| Q12 | notification, auth | 50% | 33% | 50% | 33% | 4 / 4 |
| Q14 | ledger, analytics, billing | 33% | 33% | 33% | 0% | 4 / 4 |
| Q15 | user_profile, search | 100% | 100% | 100% | 100% | 4 / 4 |
| Q17 | payment_gateway, notification, user_profile | 67% | 67% | 67% | 67% | 4 / 4 |
| Q20 | notification, user_profile, auth | 67% | 67% | 67% | 33% | 4 / 4 |
| **Avg (5 multi-hop)** | — | **62% (8/13)** | **60% (9/15)** | **62% (8/13)** | **47% (7/15)** | 4 / 4 |

Source-set identity: vector == hybrid on all 5 questions (same chunks, same order, same scores), so both vector ⊇ hybrid and hybrid ⊇ vector hold — no mismatch. Graph-facts coverage in the hybrid context was sparse: facts blocks landed on Q12 1/4, Q14 0/4, Q15 2/4, Q17 0/4, Q20 2/4 chunks (5/20 total; chunks with no MENTIONS edges pass through untouched).

## Conclusion (round 3)

- **Hybrid does not improve key_terms over a fresh vector run**: 47% vs 60% (7/15 vs 9/15); sources are identical by construction (62% both). Losses: Q14 33%→0% (but zero facts were attached there — identical context in both modes, so that delta is 1.5B sampling variance at temperature 0.2, not graph influence) and Q20 67%→33% (the facts-enriched run produced a terser reply that dropped `redis-cache-eu-1`/24h TTL, which plain vector included).
- **Junk findings, per question**: Q12 hybrid — adds a wrong `billing:write` scope (the payment-gateway's own scope, lifted from the shared payment_gateway#5 chunk) while missing `notification:write` sitting in retrieved notification#3 (missed by vector too). Q17 hybrid — digresses into payment-gateway auth plumbing ("Stripe's network tokenization", "`billing:write` scope issued by the authentication service") that the question never asked for; its vector sibling instead repeated the ledger-journal sentence twice verbatim (bloat). Q20 hybrid — drops the cache literal (`redis-cache-eu-1`) that the vector reply had. Q14 hybrid — no context change (no facts), yet a different, worse reply text (pure sampling). No verbatim "Graph facts:"/"- Entities:" text leaked into any of the 10 replies (0/10).
- **Q12 sanity check**: full hybrid reply — "The Notification Service authenticates its internal delivery calls using service-to-service tokens issued by the authentication service (AuthService) with the `billing:write` scope." — over the same 4 sources as vector (notification#1, notification#5, notification#3, payment_gateway#5). The attached graph facts for Q12 (`Notification Service -> USES -> Kafka`, `Webhooks -> PROVIDES -> Kafka`) do not surface in the answer: the observed hybrid-vs-vector delta is synthesis-emphasis drift into adjacent chunk text, not graph-derived triplet knowledge.
- **Verdict**: with facts blocks on only 5/20 chunks and mostly generic Kafka/USES/PROVIDES triplets, hybrid currently changes no multi-hop answer for the better here; its one visible effect was emphasis drift into wrong-scope/irrelevant chunk text. Flakes: 0 (10/10 HTTP 200, no retrieval-fallback replies, no retries).
- **Recommendation: keep `retrieve_mode="vector"` as the default.**
