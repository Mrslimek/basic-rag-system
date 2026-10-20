# Authentication & Identity Microservice Specification

## Overview
The Auth Service issues, validates, and revokes JWT bearer tokens for human users and machine-to-machine service accounts. It uses asymmetric Ed25519 cryptographic key pairs for token signing. Public keys are rotated every 90 days and exposed via a standard JWKS endpoint.

## Endpoints

### 1. Token Exchange (OAuth2 Password & Client Credentials)
`POST /oauth/token`

Exchanges user credentials or service credentials for a signed short-lived access token and refresh token.

**Request Payload (Service Account):**
```json
{
  "grant_type": "client_credentials",
  "client_id": "billing-service-worker",
  "client_secret": "sec_live_920f782c918341"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJFZERTQSI...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "billing:read billing:write"
}
```

### 2. JSON Web Key Set (JWKS)
`GET /.well-known/jwks.json`

Provides downstream microservices with active public keys to verify JWT signatures locally without round-tripping to Auth Service.

**Response (200 OK):**
```json
{
  "keys": [
    {
      "kty": "OKP",
      "crv": "Ed25519",
      "kid": "auth-key-2026-q4",
      "x": "11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo"
    }
  ]
}
```

### 3. Revoke Session
`POST /api/v1/sessions/revoke`

Revokes an active user session or blacklist a specific token JTI in Redis with a TTL matching token expiration.
