"""One language-server process: JSON-RPC over stdio, diagnostics for a file by pull or by push.

Never raises to its caller (but for a cancel): a hang, crash, bad frame or flood is a state and a reason, and the
process is not restarted within the run."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from core.engine.lsp.config import ServerSpec, changed_since_launch
from core.engine.lsp.protocol import encode, read_message
from core.engine.text import escape

State = Literal["idle", "starting", "running", "exited", "timed out", "off", "untrusted"]
QUIET = 0.5  # push servers: seconds without a new publication before the list counts as final
STOP_GRACE = 1.0
MAX_ITEMS = 1_000  # diagnostics kept per file
MESSAGE_CUT = 300
LANGUAGES = {
    ".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "typescriptreact", ".js": "javascript",
    ".jsx": "javascriptreact", ".mjs": "javascript", ".cjs": "javascript", ".go": "go", ".rs": "rust",
}


@dataclass(frozen=True)
class Diagnostic:
    line: int  # 1-based
    column: int  # 1-based
    severity: int  # 1 error, 2 warning, 3 information, 4 hint
    message: str
    code: str | None


class _Gone(Exception):
    """The server ended while a request was waiting."""


class _RpcError(Exception):
    pass


def _key(path: Path) -> str:
    try:
        return str(path.resolve())
    except (OSError, RuntimeError):
        return str(path)


def cut_words(text: str, limit: int) -> str:
    """`text` cut to `limit` characters at a word end, so a token the cut falls inside is dropped whole rather than
    left as a prefix no redaction pattern knows. A run with no space in it is cut as it is (it is then long)."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if text[limit].isspace():
        return head.rstrip() + "…"
    word_end = max(head.rfind(" "), head.rfind("\t"), head.rfind("\n"))
    return (head[:word_end].rstrip() if word_end > 0 else head) + "…"


def _parse(items: object) -> list[Diagnostic]:
    out: list[Diagnostic] = []
    if not isinstance(items, list):
        return out
    for item in items[:MAX_ITEMS]:
        try:
            start = item["range"]["start"]
            line, column = int(start["line"]) + 1, int(start["character"]) + 1
            severity = item.get("severity", 1)
            message = item["message"]
            code = item.get("code")
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if not isinstance(severity, int) or isinstance(severity, bool) or not isinstance(message, str):
            continue
        shown = None if code is None or isinstance(code, bool) or not isinstance(code, (str, int)) else str(code)
        out.append(Diagnostic(max(line, 1), max(column, 1), severity, cut_words(message, MESSAGE_CUT), shown))
    return out


class LspServer:
    def __init__(self, spec: ServerSpec, workspace: Path, env: Mapping[str, str]) -> None:
        self.spec = spec
        self.workspace = workspace
        self._env = {**env, **spec.env}
        self.state: State = "idle"
        self.reason: str | None = None
        self.pgid: int | None = None
        self._withdrawn = False
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[object]] = {}
        self._pull = False
        self._versions: dict[str, int] = {}
        # resolved path of a file this server was sent -> (version, diagnostics, arrival number)
        self._published: dict[str, tuple[object, list[Diagnostic], int]] = {}
        self._uris: dict[str, str] = {}  # the URI sent for a file -> its resolved path
        self._seq = 0
        self._changed = asyncio.Condition()

    # -- process ----------------------------------------------------------------------------------------------

    def _dead(self) -> bool:
        return self._withdrawn or self.state in ("exited", "off", "untrusted")

    def _fail(self, reason: str) -> None:
        """The server is not usable and is not tried again: exited with `reason`, its process group killed."""
        if self.state not in ("off", "untrusted", "exited"):
            self.state, self.reason = "exited", reason
        self.kill()

    async def _start(self) -> None:
        if self._dead():
            return
        self.state = "starting"
        if self.spec.fingerprint is not None:  # a program from `.venv/bin`: it must still be the one seen at launch
            if await asyncio.to_thread(changed_since_launch, self.spec.command, self.spec.fingerprint):
                self._fail(f"{escape(self.spec.command)} changed since launch")
                return
            if self._dead():
                return
        if os.sep not in self.spec.command:  # the spawn would look a bare name up through the inherited PATH
            self._fail(f"could not start: {escape(self.spec.command)} is not a path")
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.spec.command,
                *self.spec.args,
                cwd=self.workspace,
                env=self._env,
                start_new_session=True,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError) as exc:
            self._fail(f"could not start: {escape(str(exc))}")
            return
        self.pgid = self._proc.pid  # set before the flag is read: a `withdraw` from another thread either sees the
        self._reader = asyncio.ensure_future(self._read_loop(self._proc))  # group id or is seen here
        if self._withdrawn:  # withdrawn while the process was being spawned: `kill` found no group then
            self.kill()
            return
        root = self.workspace.as_uri()
        try:
            result = await self._request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "clientInfo": {"name": "agent"},
                    "rootUri": root,
                    "workspaceFolders": [{"uri": root, "name": self.workspace.name or "workspace"}],
                    "capabilities": {
                        "textDocument": {
                            "publishDiagnostics": {"versionSupport": True},
                            "diagnostic": {"dynamicRegistration": False},
                        },
                        "workspace": {"configuration": True},
                        "window": {"workDoneProgress": True},
                    },
                },
            )
        except _RpcError as exc:
            self._fail(f"could not start: {escape(str(exc))}")
            return
        caps = result.get("capabilities") if isinstance(result, dict) else None
        self._pull = isinstance(caps, dict) and bool(caps.get("diagnosticProvider"))
        if self._dead():
            return
        self._notify("initialized", {})
        if not self._dead():
            self.state = "running"

    def _write(self, obj: dict[str, object]) -> None:
        if self._proc is None or self._proc.stdin is None or self._dead():
            raise _Gone
        try:
            self._proc.stdin.write(encode(obj))
        except (OSError, RuntimeError) as exc:
            raise _Gone from exc

    def _notify(self, method: str, params: object = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, **({} if params is None else {"params": params})})

    async def _request(self, method: str, params: object = None) -> object:
        if self._dead():
            raise _Gone
        self._next_id += 1
        ident = self._next_id
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self._pending[ident] = future
        try:
            self._write({"jsonrpc": "2.0", "id": ident, "method": method, **({} if params is None else {"params": params})})
            assert self._proc is not None and self._proc.stdin is not None
            with contextlib.suppress(OSError, RuntimeError):
                await self._proc.stdin.drain()
            return await future
        finally:
            self._pending.pop(ident, None)

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while True:
                message = await read_message(proc.stdout)
                try:
                    await self._dispatch(message)
                except Exception:  # a malformed message of a known kind is ignored, not fatal
                    continue
        except EOFError:
            if self.state != "off":  # stdout closed but the process may not be: it cannot answer any more
                self.kill()
        except Exception:  # ProtocolError, or anything else the stream does
            self.kill()
        code = await proc.wait()
        await self._ended(code)

    async def _ended(self, code: int) -> None:
        if self.state not in ("off", "untrusted", "exited"):
            self.state = "exited"
            self.reason = f"exited (code {code}); diagnostics off for this run"
        self._signal(signal.SIGKILL)  # helpers the server left behind, once; the group id is not used again
        self.pgid = None
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(_Gone())
        async with self._changed:
            self._seq += 1
            self._changed.notify_all()

    async def _dispatch(self, message: dict[str, object]) -> None:
        method = message.get("method")
        ident = message.get("id")
        if isinstance(method, str):
            if ident is not None:
                self._answer(method, ident, message.get("params"))
            elif method == "textDocument/publishDiagnostics":
                params = message.get("params")
                key = self._known(params.get("uri")) if isinstance(params, dict) else None
                if key is not None and isinstance(params, dict):  # only for a file this server was sent
                    async with self._changed:
                        self._seq += 1
                        self._published[key] = (params.get("version"), _parse(params.get("diagnostics")), self._seq)
                        self._changed.notify_all()
            return
        if isinstance(ident, int) and (future := self._pending.get(ident)) is not None and not future.done():
            if "error" in message:
                future.set_exception(_RpcError(str(message["error"])[:200]))
            else:
                future.set_result(message.get("result"))

    def _known(self, uri: object) -> str | None:
        """The resolved path of the file this URI names, when it is one the client sent; no file system access."""
        if not isinstance(uri, str):
            return None
        if (key := self._uris.get(uri)) is not None:
            return key
        parsed = urlparse(uri)
        path = unquote(parsed.path)
        return path if parsed.scheme == "file" and path in self._versions else None

    def _answer(self, method: str, ident: object, params: object) -> None:
        reply: dict[str, object] = {"jsonrpc": "2.0", "id": ident}
        if method == "workspace/configuration":
            items = params.get("items") if isinstance(params, dict) else None
            reply["result"] = [None] * (len(items) if isinstance(items, list) else 0)
        elif method in ("window/workDoneProgress/create", "client/registerCapability", "client/unregisterCapability"):
            reply["result"] = None
        else:
            reply["error"] = {"code": -32601, "message": "method not found"}
        with contextlib.suppress(_Gone):
            self._write(reply)

    # -- diagnostics --------------------------------------------------------------------------------------------

    async def diagnostics(self, path: Path, text: str, *, wait: float, start_wait: float) -> list[Diagnostic] | None:
        if self._dead():
            return None
        starting = False
        try:
            deadline: float | None = None  # a call that starts the server has one deadline for the start and the wait
            async with self._start_lock:
                if self.state == "idle":
                    starting = True
                    deadline = asyncio.get_running_loop().time() + start_wait
                    try:
                        await asyncio.wait_for(self._start(), start_wait)
                    except TimeoutError:
                        self._fail(f"did not answer within {start_wait:g} s")  # not tried again
                        return None
                    starting = False
            if self._dead() or self.state == "starting":
                return None
            limit, shown = wait, wait
            if deadline is not None and (left := deadline - asyncio.get_running_loop().time()) < wait:
                limit, shown = max(left, 0.0), start_wait
            try:
                found = await asyncio.wait_for(self._collect(path, text), limit)
            except TimeoutError:
                if not self._dead():
                    self.state, self.reason = "timed out", f"did not answer within {shown:g} s"
                return None
            if self._dead():  # withdrawn (or ended) while it answered: nothing from it is shown
                return None
            if self.state == "timed out":
                self.state, self.reason = "running", None
            return found
        except (_Gone, _RpcError):
            return None
        except asyncio.CancelledError:
            if starting:  # Esc while it started: it never answered, so it is not tried again
                self._fail("start interrupted")
            raise
        except Exception:  # a bug here must not fail the tool call: drop the server
            if not self._dead():
                self.state, self.reason = "exited", "exited (code -1); diagnostics off for this run"
                self.kill()
            return None

    async def _collect(self, path: Path, text: str) -> list[Diagnostic] | None:
        key = _key(path)
        uri = Path(key).as_uri()
        self._uris[uri] = key
        seen = self._seq
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        if version == 1:
            self._notify(
                "textDocument/didOpen",
                {"textDocument": {"uri": uri, "languageId": LANGUAGES.get(path.suffix.lower(), path.suffix.lower().lstrip(".")), "version": 1, "text": text}},
            )
        else:
            self._notify(
                "textDocument/didChange",
                {"textDocument": {"uri": uri, "version": version}, "contentChanges": [{"text": text}]},
            )
        if self._pull:
            result = await self._request("textDocument/diagnostic", {"textDocument": {"uri": uri}})
            if isinstance(result, dict) and result.get("kind") == "full":
                found = _parse(result.get("items"))
                self._published[key] = (version, found, self._seq)
                return found
            if isinstance(result, dict) and result.get("kind") == "unchanged" and key in self._published:
                return self._published[key][1]
            return []
        return await self._push(key, version, seen)

    def _moved(self, seen: int) -> Callable[[], bool]:
        return lambda: self._seq != seen

    def _fresh(self, key: str, version: int, seen: int) -> list[Diagnostic] | None:
        """The list published for this version, or (a server that sends no version) the one that arrived after the
        change was sent."""
        entry = self._published.get(key)
        if entry is not None and (entry[0] == version or (entry[0] is None and entry[2] > seen)):
            return entry[1]
        return None

    async def _push(self, key: str, version: int, seen: int) -> list[Diagnostic] | None:
        async with self._changed:
            await self._changed.wait_for(lambda: self._dead() or self._fresh(key, version, seen) is not None)
        while not self._dead():
            try:
                async with self._changed:
                    await asyncio.wait_for(self._changed.wait_for(self._moved(self._seq)), QUIET)
            except TimeoutError:
                break
        if self._dead():
            return None
        return self._fresh(key, version, seen)

    # -- stopping -----------------------------------------------------------------------------------------------

    async def stop(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            self.kill()
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._request("shutdown"), STOP_GRACE)
            self._notify("exit")
        self.state = "off" if self.state != "exited" else self.state
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), STOP_GRACE)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc.returncode is not None:
                break
            self._signal(sig)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), STOP_GRACE)
        self._signal(signal.SIGKILL)  # helpers the server started may outlive it
        self.pgid = None  # gone: a later kill must not hit a reused group id
        if self._reader is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self._reader, STOP_GRACE)

    def _signal(self, sig: int) -> None:
        if self.pgid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.pgid, sig)

    def kill(self) -> None:
        """Synchronous SIGKILL of the whole process group."""
        self._signal(signal.SIGKILL)

    def withdraw(self, reason: str) -> None:
        """Stop now and for the rest of the run: state `off` with `reason` (so no `exited` notice follows), then the
        group is killed. Safe from any thread (the config guard calls it from a worker): it only sets the flag and
        the state, which a running call reads at its next check, and `os.killpg`, which needs no loop. A process
        still being spawned has no group yet; `_start` reads the flag right after the spawn returns and kills it."""
        self._withdrawn = True
        self.state, self.reason = "off", reason
        self.kill()
