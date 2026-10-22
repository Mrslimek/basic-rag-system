# Eval Questions — Multi-hop RAG Learning

## Single-Hop Questions (Answerable from One Document)

### Q1
**Question:** What JWT signing algorithm does the Auth Service use?
**Expected Answer:** Asymmetric Ed25519 key pairs for token signing, with public keys rotated every 90 days via the JWKS endpoint.
**Source:** auth_service.md

### Q2
**Question:** Which Kafka event topics does the Billing Service emit?
**Expected Answer:** `billing.subscription.created`, `billing.invoice.payment_failed`, and `billing.refund.settled`.
**Source:** billing_service.md

### Q3
**Question:** What is the deprecated legacy alias for the Notification Service?
**Expected Answer:** `notify-svc` (deprecated since v2.1).
**Source:** notification_service.md

### Q4
**Question:** What is the idempotency key format for Payment Gateway Service payment intents?
**Expected Answer:** `pi_<year>_<sequence>` format, e.g., `pi_2026_9912`.
**Source:** payment_gateway_service.md

### Q5
**Question:** What is the TTL for user preferences cached in Redis by the User Profile Service?
**Expected Answer:** 24 hours.
**Source:** user_profile_service.md

### Q6
**Question:** What is the maximum page size for Search Service user search?
**Expected Answer:** 50 results per page.
**Source:** search_service.md

### Q7
**Question:** What Kafka topic does the Analytics Service use for real-time metrics streaming?
**Expected Answer:** `analytics.live.metrics`.
**Source:** analytics_service.md

### Q8
**Question:** What scope does the Ledger Service require for write operations?
**Expected Answer:** `ledger:write` scope issued by the Auth Service.
**Source:** ledger_service.md

### Q9
**Question:** What is the S3 bucket name where the User Profile Service caches avatars?
**Expected Answer:** `acme-dwh`.
**Source:** user_profile_service.md

### Q10
**Question:** What Kafka topic is emitted when user profile data is updated?
**Expected Answer:** `user.profile.updated`.
**Source:** user_profile_service.md

---

## Multi-Hop Questions (Require Chaining 2–3 Documents)

### Q11
**Question:** After a successful payment, where does the transaction journal get committed?
**Expected Answer:** The Payment Gateway Service forwards the transaction journal to the Ledger Service, which commits it to PostgreSQL cluster `ha-pg-eu-1` (primary in Dublin, synchronous failover replica in Zurich).
**Hop Path:** billing_service.md → payment_gateway_service.md → ledger_service.md

### Q12
**Question:** How does the Notification Service authenticate its internal delivery calls?
**Expected Answer:** It requires a `notification:write` scope JWT issued by the Auth Service.
**Hop Path:** notification_service.md → auth_service.md

### Q13
**Question:** What is the end-to-end data path for a refund event?
**Expected Answer:** The Billing Service emits `billing.refund.settled`, the Payment Gateway Service emits `payment.refund.created`, and the Ledger Service commits the entry to `ha-pg-eu-1`.
**Hop Path:** billing_service.md → payment_gateway_service.md → ledger_service.md

### Q14
**Question:** How does the Analytics Service compute monthly revenue?
**Expected Answer:** It consumes `ledger.entry.committed` events from the Ledger Service and `billing.subscription.created` events from the Billing Service, aggregating into revenue reports.
**Hop Path:** ledger_service.md → analytics_service.md → billing_service.md

### Q15
**Question:** How does the Search Service obtain avatar metadata for indexing?
**Expected Answer:** The User Profile Service caches avatars in the `acme-dwh` S3 bucket; the Search Service reads S3 metadata from that bucket.
**Hop Path:** user_profile_service.md → search_service.md

### Q16
**Question:** What happens when a user updates their profile, end-to-end?
**Expected Answer:** The User Profile Service emits `user.profile.updated` to Kafka, which the Notification Service consumes for welcome emails, while the Auth Service validates tokens for the update operation.
**Hop Path:** user_profile_service.md → notification_service.md → auth_service.md

### Q17
**Question:** How does a payment trigger a receipt email?
**Expected Answer:** The Payment Gateway Service emits `payment.transaction.processed`, the Notification Service consumes it and sends the receipt, fetching user preferences from the User Profile Service.
**Hop Path:** payment_gateway_service.md → notification_service.md → user_profile_service.md

### Q18
**Question:** Where does the Ledger Service write committed entries, and which service reads them?
**Expected Answer:** The Ledger Service commits to PostgreSQL cluster `ha-pg-eu-1` (sync failover replica in Zurich); the Analytics Service reads from the read replica in Frankfurt.
**Hop Path:** ledger_service.md → analytics_service.md

### Q19
**Question:** What is the full dependency chain for a billing refund?
**Expected Answer:** Billing Service → Payment Gateway Service → Ledger Service → PostgreSQL cluster `ha-pg-eu-1` (Zurich failover replica).
**Hop Path:** billing_service.md → payment_gateway_service.md → ledger_service.md

### Q20
**Question:** How does the notification delivery flow authenticate and retrieve preferences?
**Expected Answer:** The Notification Service validates `notification:write` tokens from the Auth Service and fetches user notification preferences from the User Profile Service (cached in `redis-cache-eu-1`).
**Hop Path:** notification_service.md → user_profile_service.md → auth_service.md
