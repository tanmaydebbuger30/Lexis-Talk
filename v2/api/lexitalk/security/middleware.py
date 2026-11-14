"""Request id, security headers, and body-size guard."""

from __future__ import annotations

import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_RID = re.compile(r"^[A-Za-z0-9-]{8,64}$")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), geolocation=(), microphone=()",
    "Cross-Origin-Resource-Policy": "same-site",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cache-Control": "no-store",
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_bytes: int, hsts: bool) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes
        self.hsts = hsts

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _RID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = rid

        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > self.max_body_bytes:
            resp: Response = JSONResponse(
                status_code=413,
                content={
                    "error": {"code": "payload_too_large", "message": "Request body too large.", "request_id": rid}
                },
            )
        else:
            resp = await call_next(request)

        resp.headers["X-Request-ID"] = rid
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        if self.hsts:
            resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return resp


def client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
