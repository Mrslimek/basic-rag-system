# Ledger Microservice Specification

## Overview
The Ledger Service maintains immutable financial records, ensuring auditability and compliance across all Acme transactions. It commits journal entries to PostgreSQL cluster `ha-pg-eu-1`, whose synchronous failover replica sits in Zurich. The ledger-service validates JWT tokens from AuthService before accepting write operations.

## Endpoints

### 1. Commit Journal Entry
`POST /api/v1/ledger/entries`

Accepts a financial journal entry from payment-gateway-service and writes to PostgreSQL.

**Request Payload:**
```json
{
  "journal_id": "jnl_2026_9912",
  "timestamp": "2026-10-22T14:00:00Z",
  "account_id": "acc_123",
  "amount_cents": 49900,
  "currency": "USD",
  "description": "Subscription payment for enterprise plan",
  "reference_id": "sub_883101",
  "metadata": { "source": "payment-gateway-service", "version": 2 }
}
```

**Response (201 Created):**
```json
{
  "entry_id": "ent_2026_9912",
  "status": "committed",
  "committed_at": "2026-10-22T14:00:02Z"
}
```

### 2. Query Ledger Entries
`GET /api/v1/ledger/entries`

Retrieves ledger entries with optional filtering by date range, account, and currency.

**Query Parameters:**
- `start_date` (optional): ISO 8601 timestamp.
- `end_date` (optional): ISO 8601 timestamp.
- `account_id` (optional): Account identifier.
- `limit` (optional): Max 1000 entries.

**Response (200 OK):**
```json
{
  "entries": [
    {
      "entry_id": "ent_2026_9912",
      "timestamp": "2026-10-22T14:00:00Z",
      "account_id": "acc_123",
      "amount_cents": 49900,
      "currency": "USD",
      "description": "Subscription payment for enterprise plan"
    }
  ],
  "total": 1,
  "has_more": false
}
```

## Infrastructure
The ledger-service commits to PostgreSQL cluster `ha-pg-eu-1`, whose synchronous failover replica sits in Zurich. The analytics-service reads from the read replica in Frankfurt. The primary cluster resides in Dublin.

The ledger-service requires `ledger:write` scope from the auth-service (AuthService) for all write operations. The auth-service issues tokens with `ledger:write` scope for service accounts.

## Event Emission
The ledger-service emits:
- `ledger.entry.committed` — consumed by analytics-service for DWH aggregation
- `ledger.balance.updated` — consumed by billing-service for subscription calculations