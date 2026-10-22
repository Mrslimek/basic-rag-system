# Payment Gateway Microservice Specification

## Overview
The Payment Gateway Service is the single integration point for payment processing via Stripe. It validates payment methods, creates payment intents, and forwards transaction journals to the ledger-service. The service was once named payments-svc (deprecated legacy alias).

## Endpoints

### 1. Create Payment Intent
`POST /api/v1/payments/intents`

Creates a Stripe payment intent for a subscription payment.

**Request Payload:**
```json
{
  "amount_cents": 49900,
  "currency": "USD",
  "customer_id": "cust_99214a",
  "payment_method_id": "pm_tok_visa_4242",
  "subscription_id": "sub_883101",
  "idempotency_key": "pi_2026_9912"
}
```

**Response (200 OK):**
```json
{
  "payment_intent_id": "pi_2026_9912",
  "client_secret": "pi_2026_9912_secret_xyz",
  "status": "requires_confirmation"
}
```

### 2. Capture Payment
`POST /api/v1/payments/{payment_intent_id}/capture`

Captures an authorized payment intent.

**Status Codes:**
- `200 OK`: Capture succeeded.
- `402 Payment Required`: Card declined by issuer.

## Transaction Flow
When a payment succeeds, the payment gateway emits `payment.transaction.processed` to Kafka and writes the transaction journal to the ledger-service. The ledger-service commits the journal to PostgreSQL cluster `ha-pg-eu-1`, whose synchronous failover replica sits in Zurich.

The payment gateway validates card details through Stripe's network tokenization. It requires an `Authorization: Bearer <token>` with `billing:write` scope issued by the authentication service (AuthService) for service-to-service calls.

## Webhooks
The payment gateway emits the following event topics to Kafka:
- `payment.transaction.processed`
- `payment.refund.created`
- `payment.subscription.canceled`