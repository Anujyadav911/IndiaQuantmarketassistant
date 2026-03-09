"""Cloud entry point — serves MCP over SSE so Claude Desktop can connect remotely."""

from __future__ import annotations

import json
import logging
import os
import time

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route

# MCP SSE transport
from mcp.server.sse import SseServerTransport

# Re-use the fully wired server object from main.py
# (all 10 tools are already registered there)
from main import init_db, server

logger = logging.getLogger("indiaquant.http")
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO"), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

PORT      = int(os.getenv("PORT", "8000"))
HOST      = os.getenv("HOST", "0.0.0.0")
AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")   # empty = no auth

_START_TIME = time.time()

sse = SseServerTransport("/messages/")


def _check_auth(request: Request) -> bool:
    """Return True if request is authorised (or auth is disabled)."""
    if not AUTH_TOKEN:
        return True
    header = request.headers.get("Authorization", "")
    return header == f"Bearer {AUTH_TOKEN}"


async def handle_sse(request: Request) -> Response:
    """SSE endpoint — MCP clients connect here."""
    if not _check_auth(request):
        return Response("Unauthorized", status_code=401)

    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )

    return Response()                       # Starlette needs a return value


async def health(request: Request) -> JSONResponse:
    """Liveness/readiness probe — used by Render, Fly.io, UptimeRobot."""
    uptime_s = int(time.time() - _START_TIME)
    return JSONResponse(
        {
            "status": "ok",
            "service": "indiaquant-mcp",
            "tools": 10,
            "uptime_seconds": uptime_s,
            "mcp_transport": "SSE",
            "endpoints": {
                "sse":      "/sse",
                "messages": "/messages/",
                "health":   "/health",
            },
        }
    )


async def homepage(request: Request) -> PlainTextResponse:
    """Simple text status page — confirms the server is alive."""
    return PlainTextResponse(
        "IndiaQuant MCP Server is running.\n\n"
        "Connect your MCP client to:  <this-url>/sse\n"
        "Health check:                <this-url>/health\n\n"
        "Tools available (10):\n"
        "  get_live_price, get_options_chain, analyze_sentiment,\n"
        "  generate_signal, get_portfolio_pnl, place_virtual_trade,\n"
        "  calculate_greeks, detect_unusual_activity, scan_market,\n"
        "  get_sector_heatmap\n"
    )


app = Starlette(
    routes=[
        Route("/sse",      endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
        Route("/health",   endpoint=health),
        Route("/",         endpoint=homepage),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    ],
)


def run() -> None:
    init_db()
    logger.info("Starting IndiaQuant MCP (HTTP/SSE) on %s:%d", HOST, PORT)
    uvicorn.run(
        "server_http:app",
        host=HOST,
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        # Reload only in dev; disabled for production / containers
        reload=False,
    )


if __name__ == "__main__":
    run()
