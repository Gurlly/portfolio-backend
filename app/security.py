import os
import time
from typing import Deque, Dict
from collections import deque
from fastapi import Request, HTTPException

# Development and Production URL
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "")

# Simple per-IP sliding window limiter
class RateLimiter:
    def __init__(self, limit: int = 60, window_seconds: int = 60):
        self.limit = limit
        self.window = window_seconds
        self.hits: Dict[str, Deque[float]] = {}

    def check(self, key: str):
        now = time.time()
        q = self.hits.setdefault(key, deque())
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)

# 60 requrest every 2 minutes
limiter = RateLimiter(limit=60, window_seconds=120)

def client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "unknown"

# Token Verification
def verify_bearer(req: Request):
    if not API_BEARER_TOKEN:
        return
    auth = req.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.removeprefix("Bearer ").strip()
    if token != API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
