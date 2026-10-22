# Analytics Microservice Specification

## Overview
The Analytics Service computes business intelligence dashboards, revenue reports, and usage metrics. It aggregates data from the ledger-service, payment-gateway-service, and user-profile-service, and exposes results via REST and streaming Kafka topics. Previously known as analytics-svc (deprecated since v2.1).

## Endpoints

### 1. Generate Revenue Report
`GET /api/v1/analytics/reports/revenue`

Returns monthly revenue breakdown by plan tier and region.

**Response (200 OK):**
```json
{
  "report_id": "rep_20261022_001",
  "period": "2026-09-01 to 2026-09-30",
  "totals": {
    "total_revenue_usd": 124500.00,
    "by_plan": {
      "enterprise": 89000.00,
      "professional": 34500.00,
      "basic": 30000.00
    },
    "by_region": {
      "north_america": 85000.00,
      "europe": 45000.00,
      "asia_pacific": 25000.00
    }
  }
}
```

### 2. Real-time Metrics Stream
`GET /api/v1/analytics/metrics/live`

Streams real-time counters for active users, pending payments, and failed transactions.

**Response (200 OK):**
```json
{
  "stream": "analytics.live.metrics",
  "data": {
    "active_users": 14230,
    "pending_transactions": 87,
    "failed_transactions": 3,
    "avg_response_time_ms": 45
  }
}
```

## Data Sources
The analytics-service consumes:
- `payment.transaction.processed` — from payment-gateway-service for revenue tracking
- `ledger.entry.committed` — from ledger-service for balance and reconciliation
- `user.profile.updated` — from user-profile-service for user growth metrics
- `billing.subscription.created` — from billing-service for churn analysis

All ingestion pipelines validate JWT tokens from AuthService before processing. The analytics-service requires `analytics:read` scope.

## Event Topics
The analytics-service publishes derived metrics to Kafka:
- `analytics.daily_revenue` — daily revenue summaries
- `analytics.user_activity` — hourly active user counts
- `analytics.fraud_indicators` — suspicious transaction alerts

These topics feed into the dashboard service and alerting pipeline.