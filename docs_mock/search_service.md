# Search Microservice Specification

## Overview
The Search Service provides full-text and faceted search across user profiles, subscriptions, and transaction history. It indexes data from Kafka events and S3 metadata, serving queries with sub-100ms latency. Previously known as search-svc (deprecated since v3.0).

## Endpoints

### 1. Search Users
`GET /api/v1/search/users`

Searches user profiles by name, email, or metadata fields.

**Query Parameters:**
- `q` (required): Search query string.
- `filters` (optional): JSON-encoded filter object.
- `page` (optional): Page number, default 1.
- `size` (optional): Results per page, max 50.

**Response (200 OK):**
```json
{
  "results": [
    {
      "user_id": "usr_77219",
      "email": "aisha@example.com",
      "name": "Aisha Patel",
      "avatar_url": "https://acme-dwh.s3.eu-central-1.amazonaws.com/avatars/usr_77219.png"
    }
  ],
  "total": 1,
  "page": 1,
  "size": 20
}
```

### 2. Search Transactions
`GET /api/v1/search/transactions`

Searches ledger entries and payment records.

**Query Parameters:**
- `q` (required): Search query string.
- `account_id` (optional): Filter by account.
- `date_from` (optional): ISO 8601 start date.
- `date_to` (optional): ISO 8601 end date.

## Indexing Pipeline
The search-service rebuilds indexes by consuming:
- `user.profile.updated` — from user-profile-service for profile changes
- `user.avatar.updated` — from user-profile-service for avatar metadata
- `ledger.entry.committed` — from ledger-service for transaction search
- `payment.transaction.processed` — from payment-gateway-service for payment search

Avatar metadata is indexed from the `acme-dwh` S3 bucket where user-profile-service caches avatars. The search-service reads S3 object metadata for content-type and size.

The search-service requires `search:read` scope from AuthService for query endpoints.