# Billing Microservice API Specification

## Overview
The Billing Service is responsible for managing customer subscriptions, payment methods, recurring invoices, and tax calculations. It integrates with Stripe for credit card processing and handles webhooks for asynchronous transaction notifications.

## Authentication & Authorization
All endpoints require an HMAC signature or JWT issued by Auth Service. The header `X-Service-Token` or `Authorization: Bearer <token>` must accompany requests.

## Endpoints

### 1. Create Customer Subscription
`POST /api/v1/subscriptions`

Creates an active recurring subscription for an organization.

**Request Payload:**
```json
{
  "customer_id": "cust_99214a",
  "plan_id": "plan_enterprise_tier_2",
  "currency": "USD",
  "payment_method_id": "pm_tok_visa_4242",
  "auto_renew": true
}
```

**Response (201 Created):**
```json
{
  "subscription_id": "sub_883101",
  "status": "active",
  "current_period_end": "2027-11-01T00:00:00Z",
  "amount_due_cents": 49900
}
```

### 2. Refund Transaction
`POST /api/v1/payments/{payment_id}/refund`

Initiates a partial or full refund for a processed charge.

**Parameters:**
- `payment_id` (path, required): The transaction identifier to refund.
- `amount_cents` (body, optional): Amount in cents to refund. If omitted, full refund is issued.
- `reason` (body, optional): String specifying `requested_by_customer`, `duplicate`, or `fraudulent`.

**Status Codes:**
- `200 OK`: Refund queued successfully.
- `400 Bad Request`: Charge already fully refunded.
- `404 Not Found`: Transaction not found.

## Webhooks
The service emits the following event topics to Kafka:
- `billing.subscription.created`
- `billing.invoice.payment_failed`
- `billing.refund.settled`
