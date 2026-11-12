"""Which proxies' ``X-Forwarded-For`` this service believes.

With nothing configured, ``request.client`` is whatever connected: behind a load
balancer, the balancer. Believing the header from anybody instead would let a
caller name its own address. So the header is believed only from a listed proxy,
and the client is its right-most hop that is not one — uvicorn's
``ProxyHeadersMiddleware``, which takes IPs and CIDRs. Configured in the app
rather than on the uvicorn command line so the decision is the same however the
server is started, and tested. Nothing listed adds nothing: uvicorn's own default
(``127.0.0.1``) still holds.
"""

from __future__ import annotations

from fastapi import FastAPI
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware


def trust_proxies(app: FastAPI, trusted: list[str]) -> None:
    """Believe ``X-Forwarded-For`` from these proxies only; nothing if empty."""
    hosts = [host.strip() for host in trusted if host.strip()]
    if hosts:
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=hosts)
