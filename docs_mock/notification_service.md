# Notification Microservice Specification

## Overview
The Notification Service delivers transactional and marketing messages across email, SMS, and push channels. It consumes events from Kafka, renders templates with user preferences from the user-profile-service, and tracks delivery status in Redis. The service was formerly known as notify-svc (deprecated since v2.1).

## Endpoints

### 1. Send Transactional Message
`POST /api/v1/messages/transactional`

Queues a high-priority message for immediate delivery. Requires `notification:write` scope from AuthService.

**Request Payload:**
```json
{
  "template_id": "welcome_email_v3",
  "recipient_user_id": "usr_77219",
  "channel": "email",
  "payload": { "first_name": "Aisha" },
  "idempotency_key": "idem_9921a"
}
```

**Response (202 Accepted):**
```json
{
  "message_id": "msg_88411",
  "status": "queued",
  "estimated_delivery": "2026-10-22T14:05:00Z"
}
```

### 2. Delivery Status Callback
`POST /api/v1/webhooks/delivery`

Receives provider callbacks (SendGrid, Twilio, FCM) and updates Redis with final status.

**Request Payload:**
```json
{
  "message_id": "msg_88411",
  "provider": "sendgrid",
  "status": "delivered",
  "timestamp": "2026-10-22T14:05:12Z"
}
```

## Event Consumption
The notification-service subscribes to:
- `payment.transaction.processed` — triggers receipt emails via PaymentGatewayService events
- `billing.invoice.payment_failed` — sends dunning notices from Billing Service
- `user.profile.updated` — refreshes cached preferences from user-profile-service

User preferences (channel opt-outs, quiet hours) are fetched from user-profile-service at queue time and cached in redis-cache-eu-1 with 24h TTL. The authentication service validates service-to-service tokens for internal calls.