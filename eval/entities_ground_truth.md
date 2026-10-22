# Entities Ground Truth — Canonical Entity List with Name Variants

## Services

### Auth Service
**Canonical Name:** Auth Service
**Variants:**
- "Auth Service" (auth_service.md, notification_service.md, user_profile_service.md, ledger_service.md, search_service.md, analytics_service.md)
- "the authentication service" (billing_service.md, payment_gateway_service.md)
- "AuthService" (notification_service.md, user_profile_service.md, ledger_service.md)
- "auth-svc" (legacy alias, deprecated since v1.5) — appears in auth_service.md only

### Billing Service
**Canonical Name:** Billing Service
**Variants:**
- "Billing Service" (billing_service.md, notification_service.md, analytics_service.md)
- "the billing service" (payment_gateway_service.md, ledger_service.md)
- "BillingService" (notification_service.md, analytics_service.md)
- "billing-svc" (legacy alias, deprecated since v2.0) — appears in billing_service.md only

### Notification Service
**Canonical Name:** Notification Service
**Variants:**
- "Notification Service" (notification_service.md, user_profile_service.md, payment_gateway_service.md, analytics_service.md)
- "the notification service" (user_profile_service.md, search_service.md)
- "NotificationService" (notification_service.md, user_profile_service.md)
- "notify-svc" (legacy alias, deprecated since v2.1) — appears in notification_service.md only

### User Profile Service
**Canonical Name:** User Profile Service
**Variants:**
- "User Profile Service" (user_profile_service.md, notification_service.md, search_service.md, analytics_service.md)
- "the user profile service" (notification_service.md, ledger_service.md)
- "UserProfileService" (user_profile_service.md, notification_service.md)
- "profile-svc" (legacy alias, deprecated since v1.8) — appears in user_profile_service.md only

### Payment Gateway Service
**Canonical Name:** Payment Gateway Service
**Variants:**
- "Payment Gateway Service" (payment_gateway_service.md, notification_service.md, analytics_service.md)
- "the payment gateway service" (ledger_service.md, search_service.md)
- "PaymentGatewayService" (payment_gateway_service.md, notification_service.md)
- "payments-svc" (legacy alias, deprecated since v1.9) — appears in payment_gateway_service.md only

### Ledger Service
**Canonical Name:** Ledger Service
**Variants:**
- "Ledger Service" (ledger_service.md, payment_gateway_service.md, analytics_service.md)
- "the ledger service" (payment_gateway_service.md, search_service.md)
- "LedgerService" (ledger_service.md, payment_gateway_service.md)
- "ledger-svc" (legacy alias, deprecated since v1.7) — appears in ledger_service.md only

### Search Service
**Canonical Name:** Search Service
**Variants:**
- "Search Service" (search_service.md, user_profile_service.md, analytics_service.md)
- "the search service" (user_profile_service.md)
- "SearchService" (search_service.md)
- "search-svc" (legacy alias, deprecated since v3.0) — appears in search_service.md only

### Analytics Service
**Canonical Name:** Analytics Service
**Variants:**
- "Analytics Service" (analytics_service.md, ledger_service.md, search_service.md)
- "the analytics service" (ledger_service.md, payment_gateway_service.md)
- "AnalyticsService" (analytics_service.md, ledger_service.md)
- "analytics-svc" (legacy alias, deprecated since v2.1) — appears in analytics_service.md only

---

## Infrastructure Hubs

### PostgreSQL Cluster: ha-pg-eu-1
**Canonical Name:** ha-pg-eu-1
**Variants:**
- "PostgreSQL cluster `ha-pg-eu-1`" (payment_gateway_service.md, ledger_service.md, analytics_service.md)
- "PostgreSQL cluster `ha-pg-eu-1`, whose synchronous failover replica sits in Zurich" (payment_gateway_service.md, ledger_service.md)
- "read replica in Frankfurt" (ledger_service.md, analytics_service.md)
- "primary cluster resides in Dublin" (ledger_service.md)
- "ha-pg-eu-1" (all three files)

**Appears in:** payment_gateway_service.md, ledger_service.md, analytics_service.md (3+ files)

### Redis Cluster: redis-cache-eu-1
**Canonical Name:** redis-cache-eu-1
**Variants:**
- "Redis" (notification_service.md)
- "redis-cache-eu-1" (user_profile_service.md, notification_service.md)
- "Redis cache" (user_profile_service.md)
- "Redis cluster" (notification_service.md)

**Appears in:** notification_service.md, user_profile_service.md (2 files — note: requirement was 3+ files, but only 2 reference it explicitly)

### Kafka Topic Naming Scheme: domain.event.version
**Canonical Scheme:** domain.event.version
**Concrete Topics (4+):**
- `billing.subscription.created` (billing_service.md, analytics_service.md)
- `billing.invoice.payment_failed` (billing_service.md, notification_service.md)
- `billing.refund.settled` (billing_service.md, analytics_service.md)
- `payment.transaction.processed` (payment_gateway_service.md, notification_service.md, search_service.md, analytics_service.md)
- `payment.refund.created` (payment_gateway_service.md, analytics_service.md)
- `payment.subscription.canceled` (payment_gateway_service.md)
- `ledger.entry.committed` (ledger_service.md, analytics_service.md, search_service.md)
- `ledger.balance.updated` (ledger_service.md)
- `user.profile.updated` (user_profile_service.md, notification_service.md, search_service.md, analytics_service.md)
- `user.avatar.updated` (user_profile_service.md, search_service.md)
- `user.preferences.updated` (user_profile_service.md)
- `notification.delivery.attempted` (notification_service.md)
- `analytics.daily_revenue` (analytics_service.md)
- `analytics.user_activity` (analytics_service.md)
- `analytics.fraud_indicators` (analytics_service.md)
- `analytics.live.metrics` (analytics_service.md)

**Appears in:** billing_service.md, notification_service.md, payment_gateway_service.md, ledger_service.md, user_profile_service.md, search_service.md, analytics_service.md (7 files)

### S3 Bucket: acme-dwh
**Canonical Name:** acme-dwh
**Variants:**
- "S3 bucket `acme-dwh`" (user_profile_service.md, search_service.md)
- "`acme-dwh` S3 bucket" (user_profile_service.md, search_service.md)
- "acme-dwh" (user_profile_service.md, search_service.md)

**Appears in:** user_profile_service.md, search_service.md (2 files — note: requirement was 3+ files, but only 2 reference it explicitly)

---

## Split Facts (Cross-File Dependencies)

### Fact 1: Transaction Journal Path
- **Half A (payment_gateway_service.md):** "transaction journals go to the ledger-service"
- **Half B (ledger_service.md):** "ledger-service commits to PostgreSQL cluster `ha-pg-eu-1`, whose synchronous failover replica sits in Zurich"

### Fact 2: User Preferences Storage
- **Half A (notification_service.md):** "User preferences ... are fetched from user-profile-service at queue time"
- **Half B (user_profile_service.md):** "User preferences are cached in redis-cache-eu-1 with 24h TTL"

### Fact 3: Refund Event Flow
- **Half A (billing_service.md):** "The service emits the following event topics to Kafka: ... billing.refund.settled"
- **Half B (analytics_service.md):** "analytics-service consumes billing.refund.settled and writes aggregates to acme-dwh S3 bucket partitioned by date"

### Fact 4: Search Index Trigger
- **Half A (search_service.md):** "index rebuild triggered by user.profile.updated"
- **Half B (user_profile_service.md):** "user-profile-service emits user.profile.updated to Kafka after profile changes"

### Fact 5: Ledger Auth Scope
- **Half A (ledger_service.md):** "ledger entries require auth token from AuthService"
- **Half B (auth_service.md):** "AuthService issues tokens with scope ledger:write for service accounts"

### Fact 6: Payment Receipt Flow
- **Half A (payment_gateway_service.md):** "payment gateway validates cards via Stripe and emits payment.transaction.processed"
- **Half B (notification_service.md):** "notification-service listens to payment.transaction.processed for receipt emails"

### Fact 7: Analytics Read Replica
- **Half A (analytics_service.md):** "analytics-service reads from ha-pg-eu-1 read replica in Frankfurt"
- **Half B (ledger_service.md):** "ha-pg-eu-1 primary in Dublin, async replica in Frankfurt, sync replica in Zurich"

### Fact 8: Avatar Storage & Indexing
- **Half A (user_profile_service.md):** "user-profile-service caches avatars in acme-dwh S3 bucket"
- **Half B (search_service.md):** "search-service indexes avatar metadata from acme-dwh S3 bucket"

---

## Dependency Chains (Depth 3+)

### Chain 1: notification → user_profile → auth
1. notification_service.md references user_profile_service.md for preferences
2. user_profile_service.md references auth_service.md for token validation
3. auth_service.md provides JWT tokens

### Chain 2: billing → payment_gateway → ledger
1. billing_service.md references payment_gateway_service.md for payment processing
2. payment_gateway_service.md references ledger_service.md for journal commits
3. ledger_service.md references ha-pg-eu-1 PostgreSQL cluster

### Chain 3: user_profile → notification → auth
1. user_profile_service.md emits user.profile.updated
2. notification_service.md consumes user.profile.updated
3. auth_service.md validates tokens for notification calls

### Chain 4: ledger → analytics → billing
1. ledger_service.md emits ledger.entry.committed
2. analytics_service.md consumes ledger.entry.committed
3. billing_service.md emits billing.subscription.created (also consumed by analytics)