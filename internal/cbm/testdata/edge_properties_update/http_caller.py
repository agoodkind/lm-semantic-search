from __future__ import annotations

import requests


def fetch() -> str:
    return requests.get("/api/items")


def fetch_other() -> str:
    return requests.get("/api/other")
