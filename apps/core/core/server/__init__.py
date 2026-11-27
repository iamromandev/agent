# Spike findings (step 9, throwaway script deleted), Python 3.15.0, macOS arm64:
# - starlette==1.10.0 and uvicorn==0.54.0 (plain, no [standard]) install and run on 3.15.0.
# - One SSE endpoint served over a Unix socket and read by
#   `httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=path))` from a worker inside Textual `App.run_test()`:
#   200 chunks (5 ms apart), median latency 0.27 ms, max 1.4 ms (limit was 50 ms).
# - Socket permissions: do NOT use `Config(uds=...)`. uvicorn chmods the socket to 0o666 (`uvicorn/server.py:166-174`,
#   `config.py:570-576`) or keeps the mode of an existing file. The server must create the AF_UNIX socket itself
#   (umask 0o177 around bind, or fchmod before listen; a test asserts mode 0600) and pass it to
#   `Server.serve(sockets=[sock])`.
# - Peer uid check: subclass `uvicorn.protocols.http.h11_impl.H11Protocol` and wire it with `Config(http=...)`.
#   In `connection_made(transport)` wrap the WHOLE check in `except Exception` -> `transport.close()` (asyncio only
#   logs exceptions from `connection_made` and leaves the transport open). Read `transport.get_extra_info("socket")`
#   (an `asyncio.TransportSocket`, which works with `core.server.peer.peer_uid`), reject non-AF_UNIX sockets, and on
#   mismatch or error close the transport and return WITHOUT calling super().connection_made, so the rejected
#   connection never enters uvicorn's connection set. Also override `connection_lost` to do nothing for a rejected
#   connection: the base does `self.transport.close()` and raises AttributeError with transport None. The client sees
#   `httpx.RemoteProtocolError`.
# - macOS: `getsockopt(SOL_LOCAL=0, LOCAL_PEERCRED=1, 76)` returns struct xucred; cr_version (u32, must be 0) then
#   cr_uid (u32). Linux uses SO_PEERCRED (struct ucred: pid signed, uid and gid unsigned). Linux is untested here.
# - uvicorn's own `capture_signals` raises a caught SIGINT/SIGTERM again after `serve` returns, which would kill the
#   process before the daemon's cleanup: `serve._Server.handle_exit` does not record the signal.
"""The per-workspace daemon behind `agent serve` and `agent --shared`: `paths` (the private run folder), `files` (the
pid lock and the 0600 socket), `peer` and `protocol` (the peer-uid check), `daemon` (runtime, sessions, trust prompts,
idle exit), `snapshot`, `app` (the HTTP/SSE routes) and `serve` (foreground, background start, stop)."""
