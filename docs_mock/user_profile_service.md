# User Profile Microservice Specification

## Overview
The User Profile Service manages user account settings, preferences, and profile data. It stores user preferences in Redis cache, validates JWT tokens from AuthService, and emits profile update events to Kafka. Previously known as profile-svc (deprecated since v1.8).

## Endpoints

### 1. Get User Preferences
`GET /api/v1/users/{user_id}/preferences`

Fetches user-specific settings including notification channels, language, and time zone. Requires `user:read` scope from AuthService.

**Response (200 OK):**
```json
{
  "user_id": "usr_77219",
  "theme": "dark",
  "notifications": {
    "email": true,
    "sms": false,
    "push": true
  },
  "language": "en-US",
  "timezone": "America/New_York",
  "last_updated": "2026-10-21T18:30:00Z"
}
```

### 2. Update User Preferences
`PATCH /api/v1/users/{user_id}/preferences`

Updates user preference settings. Requires `user:write` scope from AuthService.

**Request Payload:**
```json
{
  "email_notifications": true,
  "sms_notifications": false,
  "push_notifications": true,
  "language": "en-GB",
  "timezone": "Europe/London"
}
```

**Response (200 OK):**
```json
{
  "user_id": "usr_77219",
  "updated_at": "2026-10-22T14:00:00Z",
  "version": 17
}
```

## Event Emission
The user-profile-service emits:
- `user.profile.updated` — when user settings change (consumed by notification-service for welcome emails)
- `user.avatar.updated` — when profile picture changes (consumed by search-service for image indexing)
- `user.preferences.updated` — when notification settings change (consumed by notification-service for channel opt-outs)

User preferences are cached in redis-cache-eu-1 with 24h TTL. The ledger-service consumes user.profile.updated to update billing records. The auth-service validates JWT tokens for all profile operations.