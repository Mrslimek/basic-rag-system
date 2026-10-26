# Extraction Gap Report — live re-sync after pipeline fixes

**Date:** 2026-10-26
**Scope:** live execution and measurement only (no code changes, no commits).
Graph rebuilt with the cloud LLM (52 extraction calls, 1 per chunk), then Neo4j audited
against the ground-truth facts from `eval/questions.md` + `docs_mock/`, then Q1–Q3
retriever measurements (vector vs graph, sources only, no answer synthesis).

## 1. What was run

- Infrastructure: `db`, `neo4j`, `qdrant` all Up/healthy. Postgres
  `service_documentation_chunks` = 52 rows (matches expectation).
- `uv run python -m app.graph_rag sync` — full graph rebuild, exit 0.
  Result: `{'postgres_rows': 52, 'graph_chunks': 52, 'graph_entities': 75}`;
  52 cloud LLM extraction calls, no extraction errors in the log.

## 2. Neo4j audit

### (a) Node labels

| labels | count |
|---|---|
| `__Node__`, `Chunk` | 52 |
| `__Node__`, `__Entity__`, `QUEUE` | 39 |
| `__Node__`, `__Entity__`, `SERVICE` | 25 |
| `__Node__`, `__Entity__`, `ENDPOINT` | 10 |
| `__Node__`, `__Entity__`, `SERVICE`, `QUEUE` | 1 (dual-type node `search transactions`) |

Total: 127 nodes, 75 entities. Zero `DATABASE`, `PROTOCOL`, `CONCEPT` entities.

### (b) Relation types (excluding MENTIONS)

| relation | count |
|---|---|
| `PUBLISHES_TO` | 39 |
| `SUBSCRIBES_TO` | 26 |
| `PROVIDES` | 10 |
| `CALLS` | 2 |
| `USES` | 1 |
| `STORES_IN` | 0 |
| `AUTHENTICATES_VIA` | 0 |

### (c) Canonicalization check — FAILED for payment gateway

All 75 entities carry the 3-label stack `__Node__`+`__Entity__`+type (1 node has 4 labels).
The normalizer collapses only `[-_\s]+`, so camelCase / word-form variants stay separate:

- Payment gateway service exists as **4** distinct nodes:
  `payment gateway`, `payment gateway service`, `paymentgatewayservice`, `payments service`.
  All three of the first names independently `PUBLISHES_TO payment.transaction.processed`.
- Other surviving variants: `auth service` / `authservice`; `analytics` / `analytics service`;
  `notification service` / `notify svc`; `user profile service` / `userpreferencesservice`.

### (d) Topics (QUEUE entities)

40 `QUEUE` ids. All 9 ground-truth topic ids are represented (8 exact, 1 mangled):

- exact: `billing.subscription.created`, `billing.refund.settled`, `ledger.entry.committed`,
  `payment.transaction.processed`, `payment.refund.created`, `analytics.live.metrics`,
  `user.profile.updated`, `user.avatar.updated`
- mangled: `billing.invoice.payment_failed` → stored as `billing.invoice.payment failed`
  (the normalizer turned the underscore into a space)

Noise QUEUE entities: `kafka`, `kafka events`, `events`, `redis`, `ledger:write`
(auth scope mis-extracted as a topic), `journal entries`, `audit events`, `transaction journals`,
`refund queue`, `payment intent created`, `analytics input stream`, `streaming kafka topics`, etc.
Several endpoint-like concepts were extracted as `SERVICE` (`webhooks`, `create customer subscription`,
`send transactional message`, `delivery status callback`, `search users`, `api gateway`, `dashboard service`).

## 3. Ground-truth gap table

### Kafka producers (topic-level, 8 facts)

| # | Ground truth | Status | Actual in graph |
|---|---|---|---|
| P1 | billing → billing.subscription.created | ЕСТЬ | `PUBLISHES_TO` (billing service → billing.subscription.created) |
| P2 | billing → billing.invoice.payment_failed | ЕСТЬ* | `PUBLISHES_TO` (billing service → "billing.invoice.payment failed"; underscore lost) |
| P3 | billing → billing.refund.settled | НЕТ | topic exists, but publisher is noise node `webhooks`, not billing |
| P4 | user-profile → user.profile.updated | ЕСТЬ | `PUBLISHES_TO` (user profile service → user.profile.updated) |
| P5 | payment-gateway → payment.transaction.processed | ЕСТЬ | `PUBLISHES_TO` from all 3 payment-gateway name variants |
| P6 | payment-gateway → payment.refund.created | ЕСТЬ | `PUBLISHES_TO` (payment gateway → payment.refund.created) |
| P7 | ledger → ledger.entry.committed | ЕСТЬ | `PUBLISHES_TO` (ledger service → ledger.entry.committed) |
| P8 | analytics → analytics.live.metrics | ЕСТЬ | `PUBLISHES_TO` (analytics → analytics.live.metrics) |

**Producers: 7/8.**

### Kafka consumers (topic-level, 8 facts)

| # | Ground truth | Status | Actual in graph |
|---|---|---|---|
| C1 | notification ← payment.transaction.processed | ЕСТЬ | `SUBSCRIBES_TO` (notification service → payment.transaction.processed) |
| C2 | notification ← user.profile.updated | ЕСТЬ | `SUBSCRIBES_TO` (notification service → user.profile.updated) |
| C3 | analytics ← ledger.entry.committed | ЕСТЬ | `SUBSCRIBES_TO` (analytics service → ledger.entry.committed) |
| C4 | analytics ← billing.subscription.created | НЕТ | no edge |
| C5 | search ← user.profile.updated | ЕСТЬ | `SUBSCRIBES_TO` (search service → user.profile.updated) |
| C6 | search ← user.avatar.updated | ЕСТЬ | `SUBSCRIBES_TO` (search service → user.avatar.updated) |
| C7 | search ← ledger.entry.committed | ЕСТЬ | `SUBSCRIBES_TO` (search service → ledger.entry.committed) |
| C8 | search ← payment.transaction.processed | НЕТ | no edge |

**Consumers: 6/8. Kafka block total: 13/16.**
By service-level rows (5 producers + 3 consumers): every one of the 8 rows has at least
one correct edge — 8/8 rows, 13/16 topic-level facts.
Previous known baseline: 2/7 Kafka links — new: 13/16 topic-level facts.

### Authentication (3 facts)

| # | Ground truth | Status | Actual in graph |
|---|---|---|---|
| A1 | search ← AuthService (scope search:read) | НЕТ | no AUTHENTICATES_VIA / USES edge |
| A2 | notification ← AuthService (scope notification:write) | НЕТ | no AUTHENTICATES_VIA / USES edge |
| A3 | ledger ← AuthService (scope ledger:write) | НЕТ | scope mis-extracted as QUEUE `ledger:write`: ledger service PUBLISHES_TO, auth service SUBSCRIBES_TO (direction inverted) |

**Auth block: 0/3.**

### Other service dependencies (7 facts)

| # | Ground truth | Status | Actual in graph |
|---|---|---|---|
| O1 | payment gateway CALLS ledger-service | ЕСТЬ | `CALLS` (payment gateway service → ledger service) |
| O2 | ledger STORES_IN PostgreSQL cluster ha-pg-eu-1 | НЕТ | no STORES_IN edges, zero DATABASE entities |
| O3 | search USES user-profile-service | НЕТ | no edge |
| O4 | search USES ledger-service | НЕТ | no edge |
| O5 | search USES payment-gateway-service | НЕТ | no edge |
| O6 | notification USES user-profile-service | ЕСТЬ (as CALLS) | `CALLS` (notification service → user profile service) |
| O7 | notification USES Redis | НЕТ | `redis` exists only as a QUEUE published by noise node `delivery status callback` |

**Other: 2/7.**

### Totals

| Block | ЕСТЬ / total |
|---|---|
| Kafka producers | 7 / 8 |
| Kafka consumers | 6 / 8 |
| Kafka total | 13 / 16 |
| Auth | 0 / 3 |
| Other | 2 / 7 |
| **Overall** | **15 / 26** |

## 4. Q1–Q3 retriever measurement

Methods: vector = prod `build_index().as_retriever(similarity_top_k=4)` over Qdrant;
graph = `build_graph_retriever(top_k=4)` (LLMSynonymRetriever + VectorContextRetriever over
Neo4j). Sources only, no answer synthesis; LLM calls: 3 total (synonym generation per question).
Coverage = the 7 ground-truth services (billing, notification, user-profile, payment-gateway,
ledger, analytics, search) mentioned in retrieved source texts.

| Q | Mode | Sources | Services covered | Extra names mentioned |
|---|---|---|---|---|
| Q1 | vector | 4 | 4/7: analytics, billing, ledger, payment-gateway | auth, kafka, postgres |
| Q1 | graph | 3 | 5/7: + search, user-profile | auth, postgres |
| Q2 | vector | 4 | 5/7: analytics, billing, notification, payment-gateway, user-profile | kafka, redis |
| Q2 | graph | 3 | 6/7: analytics, billing, ledger, notification, search, user-profile (missed payment-gateway) | auth, kafka, redis |
| Q3 | vector | 4 | 6/7: analytics, billing, ledger, payment-gateway, search, user-profile | auth, kafka |
| Q3 | graph | 2 | 3/7: ledger, search, user-profile | — |

Averages: vector 4.0 sources / 5.0 of 7 names covered; graph 2.7 sources / 4.7 of 7 names.

Graph-specific notes: Q1 graph sources carry PUBLISHES_TO/SUBSCRIBES_TO/CALLS fact lines
(good cross-service mentions); Q2 first graph source is the search-service chunk
(subscribes user.avatar.updated / ledger.entry.committed / user.profile.updated);
Q3 graph collapsed to two PROVIDES fact-nodes and never surfaced the avatar-metadata
chain, and one source spells `notify svc` (alias variant survives in fact text).

## 5. Conclusion

- Pipeline fixes work end-to-end on the cloud LLM: 52/52 chunks extracted
  (exit 0, no parse failures), 75 entities, entity-to-topic edges are now the dominant
  relation type (39 P + 26 S).
- Kafka connectivity massively improved: old known baseline 2/7 links → 13/16 topic-level
  ground-truth facts (7/8 producers, 6/8 consumers). Remaining misses: billing →
  billing.refund.settled (publisher became noise node `webhooks`), analytics ←
  billing.subscription.created, search ← payment.transaction.processed.
- Two hard extraction gaps remain: **auth scopes** (0/3, `ledger:write` inverted into a
  topic) and **STORES_IN / DATABASE** (PostgreSQL cluster entirely absent — zero DATABASE
  entities), plus near-zero service-to-service USES (only 1 USES + 2 CALLS for 7 expected
  service dependencies).
- Canonicalization still leaks: 4 payment-gateway variants and other camelCase pairs
  (`authservice`, `userpreferencesservice`, `notify svc`); the underscore→space rewrite
  additionally mangled `billing.invoice.payment_failed`.
- Retrieval Q1–Q3: vector returns more sources (avg 4.0 vs 2.7) with comparable name
  coverage (5.0 vs 4.7 of 7); graph wins name coverage on Q1–Q2 via fact-listing text
  but degrades sharply on Q3 (2 sources, 3/7).

---

# Iteration 2 — camelCase canonicalization + prompt rebalance

**Date:** 2026-10-26 (second live run). Same procedure: cloud-LLM re-sync, Neo4j audit,
26-fact gap table, Q1–Q3 retriever measurement.

**Code changes (working tree, not committed, no code touched during the run):**
- `normalize_entity_name`: camelCase splitting added (`PaymentGatewayService` → `payment gateway service`).
- Extraction prompt rebalanced: data stores are DATABASE entities with `STORES_IN`;
  JWT scopes link to AuthService via `AUTHENTICATES_VIA` (never model scopes as QUEUE);
  QUEUE only for dotted topic names (`service.event.action`); do not invent facts.
- `max_triplets_per_chunk=6`; `pytest` 14 passed.

## 1. Sync

Exit 0. `{'postgres_rows': 52, 'graph_chunks': 52, 'graph_entities': 56}` (was 75),
52 extraction calls, 0 parse failures in the log.

## 2. Neo4j audit

### (a) Node labels

| labels | count | iter 1 |
|---|---|---|
| `__Node__`, `Chunk` | 52 | 52 |
| `__Node__`, `__Entity__`, `QUEUE` | 24 | 39 |
| `__Node__`, `__Entity__`, `SERVICE` | 21 | 25 |
| `__Node__`, `__Entity__`, `DATABASE` | 9 | **0** |
| `__Node__`, `__Entity__`, `ENDPOINT` | 1 | 10 |
| `__Node__`, `__Entity__`, `SERVICE`, `QUEUE` | 1 (dual `analytics service`) | 1 (`search transactions`) |

### (b) Relation types (excluding MENTIONS)

| relation | count | iter 1 |
|---|---|---|
| `PUBLISHES_TO` | 26 | 39 |
| `SUBSCRIBES_TO` | 14 | 26 |
| `STORES_IN` | **13** | 0 |
| `AUTHENTICATES_VIA` | **12** | 0 |
| `USES` | 1 | 1 |
| `PROVIDES` | 0 | 10 |
| `CALLS` | 0 | 2 |

### (c) Canonicalization check

- `auth service`: **1 node** — `authservice` collapsed into it. OK.
- payment gateway: **still 2 variants** — `payment gateway` and `payment gateway service`
  (both PUBLISHES_TO payment.transaction.processed). The rule merges camelCase but not
  the "X" vs "X Service" word-form; `payments service` / `paymentgatewayservice` collapsed. FAIL.
- user preferences: `userpreferencesservice` gone; two endpoint-like noise SERVICE nodes
  remain: `get user preferences`, `update user preferences`.
- Other surviving variants: `user profile service` + `user profile microservice`;
  `billing service` + `billing service worker`.

### (d) QUEUE entities

Iteration-1 garbage is gone: `kafka`, `redis`, `events`, `ledger:write`, `journal entries`,
`audit events`, `transaction journals`, `refund queue`, `payment intent created` — all removed.
True dotted topics present: `billing.subscription.created`, `billing.refund.settled`,
`ledger.entry.committed`, `payment.transaction.processed`, `payment.refund.created`,
`user.profile.updated`, `user.avatar.updated`. New plausible topics: `auth.event.user.*`,
`auth.token.*`, `billing.invoice.created`, `billing.payment.completed`, `stripe.payment.*`,
`user.preferences.updated`.

Residual issues: `billing.invoice.payment_failed` still stored as
`billing.invoice.payment failed` (underscore→space rewrite); `analytics.live.metrics`
is MISSING (regression — present in iter 1); noise remains: dual node `analytics service`
with a self-loop `PUBLISHES_TO analytics service`, and mangled `analytics.daily revenue`
/ `analytics.fraud indicators` / `analytics.user activity`.

## 3. Ground-truth gap table (26 facts)

### Kafka producers

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| P1 | billing → billing.subscription.created | ЕСТЬ | PUBLISHES_TO (billing service) |
| P2 | billing → billing.invoice.payment_failed | ЕСТЬ* | as `billing.invoice.payment failed` (underscore lost) |
| P3 | billing → billing.refund.settled | НЕТ | again published by noise node `webhooks` |
| P4 | user-profile → user.profile.updated | ЕСТЬ | PUBLISHES_TO (user profile service) |
| P5 | payment-gateway → payment.transaction.processed | ЕСТЬ | published by both payment-gateway variants |
| P6 | payment-gateway → payment.refund.created | ЕСТЬ | PUBLISHES_TO (payment gateway) |
| P7 | ledger → ledger.entry.committed | ЕСТЬ | PUBLISHES_TO (ledger service) |
| P8 | analytics → analytics.live.metrics | НЕТ | topic node absent (iter 1: ЕСТЬ) — regression |

**Producers: 7/8** (iter 1: 7/8 — different miss).

### Kafka consumers

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| C1 | notification ← payment.transaction.processed | ЕСТЬ | SUBSCRIBES_TO (notification service) |
| C2 | notification ← user.profile.updated | ЕСТЬ | SUBSCRIBES_TO (notification service) |
| C3 | analytics ← ledger.entry.committed | ЕСТЬ | SUBSCRIBES_TO (analytics service) |
| C4 | analytics ← billing.subscription.created | НЕТ | no edge |
| C5 | search ← user.profile.updated | НЕТ | search lost this edge (iter 1: ЕСТЬ) |
| C6 | search ← user.avatar.updated | ЕСТЬ | SUBSCRIBES_TO (search service) |
| C7 | search ← ledger.entry.committed | НЕТ | search lost this edge (iter 1: ЕСТЬ) |
| C8 | search ← payment.transaction.processed | НЕТ | no edge |

**Consumers: 4/8. Kafka block: 11/16** (iter 1: 13/16 — regressed).

### Authentication

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| A1 | search ← AuthService (search:read) | ЕСТЬ | AUTHENTICATES_VIA (search service → auth service) |
| A2 | notification ← AuthService (notification:write) | НЕТ | no edge (12 AUTHENTICATES_VIA edges, notification absent) |
| A3 | ledger ← AuthService (ledger:write) | ЕСТЬ | AUTHENTICATES_VIA (ledger service → auth service) |

**Auth block: 2/3** (iter 1: 0/3 — fixed).

### Other service dependencies

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| O1 | payment gateway CALLS ledger-service | НЕТ | CALLS extinct; no payment-gateway→ledger edge (iter 1: ЕСТЬ) |
| O2 | ledger STORES_IN PostgreSQL cluster ha-pg-eu-1 | ЕСТЬ | STORES_IN (ledger service → `ha pg eu 1`) |
| O3 | search USES user-profile-service | НЕТ | no edge |
| O4 | search USES ledger-service | НЕТ | no edge |
| O5 | search USES payment-gateway-service | НЕТ | no edge |
| O6 | notification USES user-profile-service | НЕТ | CALLS extinct (iter 1: ЕСТЬ as CALLS) |
| O7 | notification USES Redis | ЕСТЬ | STORES_IN (notification service → `redis`, `redis cache eu 1`) |

**Other: 2/7** (iter 1: 2/7 — different rows: O1/O6 swapped for O2/O7).

### Totals

| Block | Iter 2 | Iter 1 |
|---|---|---|
| Kafka producers | 7/8 | 7/8 |
| Kafka consumers | 4/8 | 6/8 |
| Kafka total | **11/16** | 13/16 |
| Auth | **2/3** | 0/3 |
| Other | 2/7 | 2/7 |
| **Overall** | **15/26** | 15/26 |

## 4. Q1–Q3 retriever measurement (same methodology: sources only, 3 synonym LLM calls)

| Q | Mode | Sources | Services covered | Extra names |
|---|---|---|---|---|
| Q1 | vector | 4 | 4/7: analytics, billing, ledger, payment-gateway | auth, kafka, postgres |
| Q1 | graph | 3 | **7/7**: all seven services | auth, redis |
| Q2 | vector | 4 | 5/7: analytics, billing, notification, payment-gateway, user-profile | kafka, redis |
| Q2 | graph | 4 | 5/7: billing, ledger, notification, search, user-profile (missed analytics, payment-gateway; 2 noise fact-nodes) | auth, redis |
| Q3 | vector | 4 | 6/7: analytics, billing, ledger, payment-gateway, search, user-profile | auth, kafka |
| Q3 | graph | 3 | 6/7: billing, ledger, notification, payment-gateway, search, user-profile (missed analytics) | auth, redis |

Averages: vector 4.0 sources, **5.0/7** names (iter 1: 5.0/7 — unchanged, Qdrant untouched);
graph 3.3 sources, **6.0/7** names (iter 1: 4.7/7 — improved).

## 5. Iteration-2 conclusion

- **Fixed:** auth scopes now extract as AUTHENTICATES_VIA (2/3, `ledger:write` no longer a QUEUE);
  DATABASE entities and STORES_IN appeared (9 nodes, 13 edges, ledger→ha-pg-eu-1 and
  notification→redis captured); QUEUE garbage eliminated (kafka/redis/events/ledger:write gone);
  entity count down 75→56; ENDPOINT noise 10→1.
- **Regressed:** CALLS and PROVIDES vanished entirely (payment gateway→ledger and
  notification→user-profile service edges lost; PROVIDES 10→0); `analytics.live.metrics`
  missing (producer P8 lost); search lost 2 consumer edges (user.profile.updated,
  ledger.entry.committed).
- **Still broken:** payment gateway has 2 word-form variants; underscore mangling of
  `billing.invoice.payment_failed`; noise SERVICE nodes from endpoint text
  (`webhooks`, `search users`, etc.); dual-type node `analytics service` with self-loop.
- **Retrieval:** graph name coverage rose 4.7→6.0/7 (Q1 now covers all 7 services via
  STORES_IN/AUTHENTICATES_VIA/SUBSCRIBES fact lines); vector stays 5.0/7. Graph still
  returns fewer sources (avg 3.3 vs 4.0) and sometimes noise fact-nodes with no service names.

---

# Iteration 3 — topic names kept verbatim + all seven relations defined

**Date:** 2026-10-26 (third live run). Same procedure: cloud-LLM re-sync, Neo4j audit,
26-fact gap table, Q1–Q3 retriever measurement.

**Code changes (working tree, not committed, no code touched during the run):**
- `normalize_entity_name` dot-guard: names containing a dot are kept verbatim
  (strip + lowercase only), so `billing.invoice.payment_failed` is no longer mangled.
- Extraction prompt now defines all 7 relations: USES (general dependency / reading data),
  CALLS (synchronous call), PROVIDES (service exposes an ENDPOINT) added to the previous four.
- `max_triplets_per_chunk=8`; `pytest` 15 passed.

## 1. Sync

Exit 0. `{'postgres_rows': 52, 'graph_chunks': 52, 'graph_entities': 67}` (iter 2: 56),
52 extraction calls, 0 parse failures.

## 2. Neo4j audit

### (a) Node labels

| labels | count | iter 2 |
|---|---|---|
| `__Node__`, `Chunk` | 52 | 52 |
| `__Node__`, `__Entity__`, `SERVICE` | 29 | 21 |
| `__Node__`, `__Entity__`, `ENDPOINT` | 18 | 1 |
| `__Node__`, `__Entity__`, `QUEUE` | 15 | 24 (+1 dual) |
| `__Node__`, `__Entity__`, `DATABASE` | 5 | 9 |
| dual-type (SERVICE+QUEUE) nodes | **0** | 1 (`analytics service`) |

### (b) Relation types (excluding MENTIONS) — all 7 alive

| relation | count | iter 2 |
|---|---|---|
| `SUBSCRIBES_TO` | 19 | 14 |
| `PROVIDES` | **17** | 0 |
| `PUBLISHES_TO` | 17 | 26 |
| `USES` | **10** | 1 |
| `AUTHENTICATES_VIA` | 7 | 12 |
| `STORES_IN` | 7 | 13 |
| `CALLS` | **6** | 0 |

CALLS and PROVIDES are back (0 → 6 and 0 → 17); USES grew 1 → 10.

### (c) Topic name integrity (full QUEUE list, 15 nodes)

All 15 QUEUE ids are topic-shaped; no `kafka` / `redis` / `events` / scope garbage,
and no dual-type node:

`analytics.daily_revenue`, `analytics.fraud_indicators`, `analytics.live.metrics`,
`analytics.user_activity`, `billing.invoice.payment_failed`, `billing.refund.settled`,
`billing.subscription.created`, `ledger.balance.updated`, `ledger.entry.committed`,
`payment.refund.created`, `payment.subscription.canceled`, `payment.transaction.processed`,
`user.avatar.updated`, `user.preferences.updated`, `user.profile.updated`.

- **`billing.invoice.payment_failed` — exact, with underscore** (dot-guard works; iter 1/2 had `payment failed`).
- **`analytics.live.metrics` — returned** (was missing in iter 2).
- All 9 ground-truth topic ids now present in exact form.

### (d) Name duplicates (the four probed entities)

| entity | variants | count |
|---|---|---|
| payment gateway | `payment gateway`, `payment gateway service` | 2 (word-form alias remains) |
| user profile | `user profile service` | 1 |
| billing | `billing service`, `billing service worker`, `billing microservice` | 3 |
| auth | `auth service` | 1 |

Additional variants seen: `analytics` / `analytics service`; `ledger` / `ledger service`.
The `payment gateway` vs `payment gateway service` split is deterministic after
normalization — both are separate strings and cannot be merged without an LLM
entity-resolution stage.

## 3. Ground-truth gap table (26 facts)

### Kafka producers

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| P1 | billing → billing.subscription.created | ЕСТЬ | PUBLISHES_TO (billing service) |
| P2 | billing → billing.invoice.payment_failed | ЕСТЬ | PUBLISHES_TO — exact underscore name |
| P3 | billing → billing.refund.settled | НЕТ | again published by noise `webhooks` |
| P4 | user-profile → user.profile.updated | ЕСТЬ | PUBLISHES_TO (user profile service) |
| P5 | payment-gateway → payment.transaction.processed | ЕСТЬ | both payment-gateway variants publish it |
| P6 | payment-gateway → payment.refund.created | ЕСТЬ | PUBLISHES_TO (payment gateway) |
| P7 | ledger → ledger.entry.committed | ЕСТЬ | PUBLISHES_TO (ledger service) |
| P8 | analytics → analytics.live.metrics | НЕТ | node returned, but direction wrong: `analytics` SUBSCRIBES_TO it, nobody publishes |

**Producers: 6/8.**

### Kafka consumers

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| C1 | notification ← payment.transaction.processed | ЕСТЬ | SUBSCRIBES_TO (notification service) |
| C2 | notification ← user.profile.updated | ЕСТЬ | SUBSCRIBES_TO (notification service) |
| C3 | analytics ← ledger.entry.committed | ЕСТЬ | SUBSCRIBES_TO (analytics service) |
| C4 | analytics ← billing.subscription.created | НЕТ | no edge |
| C5 | search ← user.profile.updated | ЕСТЬ | recovered (iter 2: НЕТ) |
| C6 | search ← user.avatar.updated | ЕСТЬ | SUBSCRIBES_TO (search service) |
| C7 | search ← ledger.entry.committed | ЕСТЬ | recovered (iter 2: НЕТ) |
| C8 | search ← payment.transaction.processed | ЕСТЬ | new (iter 1/2: НЕТ) |

**Consumers: 7/8. Kafka block: 13/16** (iter 2: 11/16).

### Authentication

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| A1 | search ← AuthService (search:read) | НЕТ | edge lost vs iter 2 (7 AUTHENTICATES_VIA edges, search absent) |
| A2 | notification ← AuthService (notification:write) | НЕТ | still absent |
| A3 | ledger ← AuthService (ledger:write) | ЕСТЬ | AUTHENTICATES_VIA (ledger service → auth service) |

**Auth block: 1/3** (iter 2: 2/3).

### Other service dependencies

| # | Fact | Status | Actual in graph |
|---|---|---|---|
| O1 | payment gateway CALLS ledger-service | ЕСТЬ | CALLS (payment gateway service → ledger service) + USES (payment gateway → ledger service) |
| O2 | ledger STORES_IN PostgreSQL cluster ha-pg-eu-1 | ЕСТЬ | STORES_IN (ledger service → `ha pg eu 1`) |
| O3 | search USES user-profile-service | НЕТ | no USES edge from search |
| O4 | search USES ledger-service | НЕТ | no USES edge from search |
| O5 | search USES payment-gateway-service | НЕТ | no USES edge from search |
| O6 | notification USES user-profile-service | ЕСТЬ | both CALLS and USES (notification service → user profile service) |
| O7 | notification USES Redis | ЕСТЬ | STORES_IN (notification service → `redis`, `redis cache eu 1`) |

**Other: 4/7** (iter 2: 2/7).

### Totals (all three iterations)

| Block | Iter 1 | Iter 2 | Iter 3 |
|---|---|---|---|
| Kafka producers | 7/8 | 7/8 | 6/8 |
| Kafka consumers | 6/8 | 4/8 | **7/8** |
| Kafka total | 13/16 | 11/16 | **13/16** |
| Auth | 0/3 | **2/3** | 1/3 |
| Other | 2/7 | 2/7 | **4/7** |
| **Overall** | 15/26 | 15/26 | **18/26** |

Six rows lost in iter 2, in iter 3: `search←user.profile.updated` ✓ recovered,
`search←ledger.entry.committed` ✓ recovered, `gateway CALLS ledger` ✓ recovered,
`notification USES user-profile` ✓ recovered (4/6); `analytics.live.metrics` — node
back but wrong direction (still НЕТ); `search→auth` regressed to НЕТ.

### Relation types across iterations

| relation | Iter 1 | Iter 2 | Iter 3 |
|---|---|---|---|
| PUBLISHES_TO | 39 | 26 | 17 |
| SUBSCRIBES_TO | 26 | 14 | 19 |
| PROVIDES | 10 | 0 | 17 |
| CALLS | 2 | 0 | 6 |
| USES | 1 | 1 | 10 |
| STORES_IN | 0 | 13 | 7 |
| AUTHENTICATES_VIA | 0 | 12 | 7 |

## 4. Q1–Q3 retriever measurement (same methodology: sources only, 3 synonym LLM calls)

| Q | Mode | Sources | Services covered | Extra names |
|---|---|---|---|---|
| Q1 | vector | 4 | 4/7: analytics, billing, ledger, payment-gateway | auth, kafka, postgres |
| Q1 | graph | 5 | **7/7**: all seven services | auth, kafka, redis |
| Q2 | vector | 4 | 5/7: analytics, billing, notification, payment-gateway, user-profile | kafka, redis |
| Q2 | graph | 2 | 5/7: billing, ledger, notification, search, user-profile | auth, redis |
| Q3 | vector | 4 | 6/7: analytics, billing, ledger, payment-gateway, search, user-profile | auth, kafka |
| Q3 | graph | 2 | 3/7: ledger, search, user-profile | — |

Averages: vector 4.0 sources, **5.0/7** names (unchanged across all iterations —
Qdrant untouched); graph 3.0 sources, **5.0/7** names (iter 1: 4.7, iter 2: 6.0).

| Coverage | Iter 1 | Iter 2 | Iter 3 |
|---|---|---|---|
| vector avg names | 5.0/7 | 5.0/7 | 5.0/7 |
| graph avg names | 4.7/7 | 6.0/7 | 5.0/7 |
| graph avg sources | 2.7 | 3.3 | 3.0 |

## 5. Iteration-3 conclusion

- **What became the maximum:** graph edges now span all 7 relation types
  (CALLS 6, PROVIDES 17, USES 10 alive); all 9 ground-truth topics exist in exact
  spelling (`billing.invoice.payment_failed` kept its underscore, `analytics.live.metrics`
  returned); zero queue/scope garbage and zero dual-type nodes; consumers 7/8 and
  overall ground-truth coverage 18/26 — best of all three runs (iter 1: 15, iter 2: 15);
  all four iter-2-lost service/fact rows recovered; notifications USES user-profile.
- **What remains residual:** (1) `payment gateway` / `payment gateway service` word-form
  aliases deterministically cannot be merged by string normalization — these need an
  LLM entity-resolution stage (separate topic); same for `analytics`/`analytics service`,
  `ledger`/`ledger service`, and the `billing service worker`/`billing microservice` names.
  (2) P3 `billing → billing.refund.settled` keeps linking to the noise node `webhooks`
  in every iteration. (3) P8 `analytics.live.metrics` has the wrong direction
  (analytics subscribes instead of publishing). (4) Auth is unstable across iterations:
  0/3 → 2/3 → 1/3 (`search→auth` and `notification→auth` flip-flop depending on the LLM).
  (5) search's three USES rows remain unexpressed (its dependencies appear only as
  SUBSCRIBES_TO edges). (6) Graph retrieval variance remains: 4.7 → 6.0 → 5.0/7 name
  coverage; Q3 again drops to 2 sources/3 names because top-4 gets filled with
  narrow PROVIDES fact-nodes.