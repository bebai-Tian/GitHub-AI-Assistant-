"""Validity of project memory is independent of its historical storage."""

from datetime import datetime, timezone


def memory_is_active(metadata: dict | None) -> bool:
    metadata = metadata or {}
    if metadata.get("memory_status", "active") != "active":
        return False
    if metadata.get("valid_until"):
        try:
            expiry = datetime.fromisoformat(str(metadata["valid_until"]).replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return expiry > datetime.now(timezone.utc)
        except ValueError:
            return False
    return True
