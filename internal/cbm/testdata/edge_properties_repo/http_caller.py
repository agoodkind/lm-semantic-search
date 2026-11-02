from __future__ import annotations

import requests


def fetch() -> str:
    return requests.get("/api/items")
