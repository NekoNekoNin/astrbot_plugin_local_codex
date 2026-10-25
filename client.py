"""Call the installed Codex app-server using its existing account session."""

import asyncio
import base64
import binascii
import contextlib
import copy
import json
import logging
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

from .request_state import REQUESTS


class CodexClient:
    """Manage one local app-server and its transient inference threads."""

    def __init__(self, home: Path, workspace: Path):
        self.home = home
        self.workspace = workspace
        self.process = None
        self.pending = {}
        self.threads = {}
        self.tool_requests = {}
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.tasks = []
        self.catalog = {}
        self.catalog_at = None
        REQUESTS.clients.add(self)

    async def start(self):
        """Initialize the installed CLI without copying its credentials."""
        async with self.lock:
            if self.process and self.process.returncode is None:
                return
            candidates = list(
                (Path.home() / "AppData/Local/OpenAI/Codex/bin").glob("*/codex.exe")
            )
            executable = shutil.which("codex")
            if not executable and candidates:
                executable = str(max(candidates, key=lambda p: p.stat().st_mtime))
            if not executable:
                raise RuntimeError("The local Codex CLI was not found.")
            config_file = self.home / "config.toml"
            config = (
                tomllib.loads(config_file.read_text(encoding="utf-8-sig"))
                if config_file.exists()
                else {}
            )
            overrides = {
                "features.shell_tool": False,
                "features.unified_exec": False,
                "features.shell_snapshot": False,
                "features.apps": False,
                "features.plugins": False,
                "features.remote_plugin": False,
                "features.multi_agent": False,
                "features.computer_use": False,
                "features.browser_use": False,
                "features.in_app_browser": False,
                "features.image_generation": False,
                "features.view_image": False,
                "features.sleep_tool": False,
                "features.hooks": False,
                "features.memories": False,
                "features.skip_host_skill_discovery": True,
                "tools.view_image": False,
                "project_doc_max_bytes": 0,
                "web_search": "disabled",
            }
            for name in config.get("mcp_servers", {}):
                overrides[f"mcp_servers.{name}.enabled"] = False
            arguments = [executable, "app-server", "--listen", "stdio://"]
            for key, value in overrides.items():
                arguments.extend(["-c", f"{key}={json.dumps(value)}"])
            self.workspace.mkdir(parents=True, exist_ok=True)
            environment = {**os.environ, "CODEX_HOME": str(self.home)}
            self.process = await asyncio.create_subprocess_exec(
                *arguments,
                cwd=self.workspace,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=16 * 1024 * 1024,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.tasks = [
                asyncio.create_task(self.read_messages()),
                asyncio.create_task(self.drain_stderr()),
            ]
            try:
                await self.request(
                    "initialize",
                    {
                        "clientInfo": {
                            "name": "astrbot_local_codex",
                            "title": "AstrBot Local Codex",
                            "version": "1.0.0",
                        },
                        "capabilities": {"experimentalApi": True},
                    },
                )
                await self.write({"method": "initialized", "params": {}})
            except BaseException:
                await self.close()
                raise

    async def write(self, message):
        """Write one protocol message; never log account or conversation data.

        Args:
            message: JSON-RPC request, notification, or response.
        """
        if not self.process or self.process.returncode is not None:
            raise RuntimeError("The local Codex connection is closed.")
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def request(self, method, params):
        """Await one app-server response.

        Args:
            method: App-server method name.
            params: Method parameters.

        Returns:
            The result returned by app-server.
        """
        self.sequence += 1
        request_id = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.write({"id": request_id, "method": method, "params": params})
            return await asyncio.wait_for(future, 60)
        finally:
            self.pending.pop(request_id, None)

    async def drain_stderr(self):
        """Drain diagnostic output without recording potentially sensitive text."""
        while await self.process.stderr.read(8192):
            pass

    async def read_messages(self):
        """Dispatch responses and per-thread events from the CLI."""
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                method = message.get("method")
                if not method and message.get("id") in self.pending:
                    future = self.pending[message["id"]]
                    if future.done():
                        continue
                    if "error" in message:
                        future.set_exception(
                            RuntimeError(message["error"].get("message", "Codex error"))
                        )
                    else:
                        future.set_result(message.get("result", {}))
                    continue
                params = message.get("params", {})
                thread_id = params.get("threadId")
                if thread_id in self.threads:
                    await self.threads[thread_id]["queue"].put(message)
                elif method and "id" in message:
                    await self.write(
                        {
                            "id": message["id"],
                            "error": {"code": -32601, "message": "Not available"},
                        }
                    )
        finally:
            error = RuntimeError("The local Codex app-server disconnected.")
            for future in list(self.pending.values()):
                if not future.done():
                    future.set_exception(error)
            for state in self.threads.values():
                state["queue"].put_nowait({"disconnect": True})

    async def models(self):
        """Read the model catalog from the user's installed CLI.

        Returns:
            Model identifiers returned by Codex.
        """
        await self.start()
        result = await self.request(
            "model/list", {"limit": 100, "includeHidden": False}
        )
        self.catalog = {
            item.get("model", item["id"]): item for item in result.get("data", [])
        }
        self.catalog_at = time.monotonic()
        return list(self.catalog)

    async def supported_efforts(self, model):
        """Return current account-supported efforts, using a short catalog cache.

        Args:
            model: Exact model identifier selected in AstrBot.

        Returns:
            Supported effort names, or an empty set for an unknown catalog entry.
        """
        if self.catalog_at is None or time.monotonic() - self.catalog_at > 300:
            await self.models()
        item = self.catalog.get(model, {})
        return {
            choice["reasoningEffort"]
            for choice in item.get("supportedReasoningEfforts", [])
            if "reasoningEffort" in choice
        }

    async def finish_request(self, request_id):
        """Release native turns left suspended by an interrupted tool execution.

        Args:
            request_id: Exact owning pipeline identifier.
        """
        for thread_id, state in list(self.threads.items()):
            if state.get("request_id") == request_id:
                await self.finish_thread(thread_id, interrupt=True)

    async def finish_thread(self, thread_id, interrupt=False):
        """Release a completed, cancelled, expired, or failed inference thread.

        Args:
            thread_id: Transient Codex thread ID.
            interrupt: Whether a turn may still be running.
        """
        state = self.threads.pop(thread_id, None)
        if not state:
            return
        for call_id, pending in list(self.tool_requests.items()):
            if pending[0] == thread_id:
                self.tool_requests.pop(call_id, None)
        if interrupt and state.get("turn_id"):
            with contextlib.suppress(Exception):
                await self.request(
                    "turn/interrupt",
                    {"threadId": thread_id, "turnId": state["turn_id"]},
                )
        with contextlib.suppress(Exception):
            await self.request("thread/unsubscribe", {"threadId": thread_id})

    async def generate(
        self,
        messages,
        model,
        tools=None,
        effort="xhigh",
        timeout=300,
        request_record=None,
        image_resolver=None,
    ):
        """Stream text or AstrBot-owned function calls from a transient thread.

        Args:
            messages: OpenAI-format messages prepared by AstrBot.
            model: Exact model identifier.
            tools: OpenAI function declarations owned and executed by AstrBot.
            effort: Codex reasoning effort.
            timeout: Maximum time for this inference step in seconds.
            request_record: Explicit owning chat request; background calls use None.
            image_resolver: Async converter for image references left by preprocessing.

        Yields:
            Text chunks followed by one result containing text or function calls.
        """
        record = request_record
        await self.start()
        for old_id, old_state in list(self.threads.items()):
            if time.monotonic() - old_state["created"] > 900:
                await self.finish_thread(old_id, interrupt=True)
        tool_messages = [
            item
            for item in messages
            if item.get("role") == "tool"
            and item.get("tool_call_id") in self.tool_requests
        ]
        if tool_messages:
            thread_id = self.tool_requests[tool_messages[0]["tool_call_id"]][0]
            state = self.threads[thread_id]
            for item in tool_messages:
                pending = self.tool_requests.pop(item["tool_call_id"])
                if pending[0] != thread_id:
                    raise RuntimeError(
                        "Tool results belong to different conversations."
                    )
                text = item.get("content", "")
                if not isinstance(text, str):
                    text = json.dumps(text, ensure_ascii=False)
                await self.write(
                    {
                        "id": pending[1],
                        "result": {
                            "contentItems": [{"type": "inputText", "text": text}],
                            "success": True,
                        },
                    }
                )
        else:
            system = "\n\n".join(
                str(item.get("content", ""))
                for item in messages
                if item.get("role") in ("system", "developer")
            )
            transcript = copy.deepcopy(
                [x for x in messages if x.get("role") not in ("system", "developer")]
            )
            images = []
            for item in transcript:
                if isinstance(item.get("content"), list):
                    for part in item["content"]:
                        if part.get("type") == "image_url":
                            image_info = part.get("image_url")
                            image_ref = (
                                image_info.get("url", "")
                                if isinstance(image_info, dict)
                                else ""
                            )
                            if not isinstance(image_ref, str):
                                image_ref = ""
                            if (
                                not image_ref.startswith("data:image/")
                                and image_resolver
                            ):
                                try:
                                    image_ref = await image_resolver(image_ref)
                                except Exception:
                                    image_ref = None
                            valid_image = False
                            if isinstance(image_ref, str):
                                header, separator, encoded = image_ref.partition(",")
                                if (
                                    header.startswith("data:image/")
                                    and header.endswith(";base64")
                                    and separator
                                ):
                                    try:
                                        valid_image = bool(
                                            base64.b64decode(encoded, validate=True)
                                        )
                                    except (binascii.Error, ValueError):
                                        pass
                            if not valid_image:
                                logging.getLogger(__name__).warning(
                                    "Skipped an unavailable image before sending Codex input."
                                )
                                part.clear()
                                part.update(
                                    {
                                        "type": "text",
                                        "text": "[Image unavailable: it could not be read or decoded. Do not infer its contents; ask the user to resend it if needed.]",
                                    }
                                )
                                continue
                            images.append({"type": "image", "url": image_ref})
                            part.clear()
                            part.update(
                                {
                                    "type": "text",
                                    "text": f"[attached image {len(images)}]",
                                }
                            )
                        elif part.get("type") == "input_audio":
                            raise ValueError(
                                "Configure an AstrBot STT provider for audio input."
                            )
            dynamic_tools = [
                {
                    "type": "function",
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "inputSchema": tool["function"].get(
                        "parameters", {"type": "object"}
                    ),
                    "deferLoading": False,
                }
                for tool in tools or []
            ]
            result = await self.request(
                "thread/start",
                {
                    "model": model,
                    "cwd": str(self.workspace),
                    "ephemeral": True,
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "baseInstructions": (
                        "You are the language model behind AstrBot. Answer the supplied "
                        "conversation as its next assistant. Use only the supplied dynamic "
                        "tools when useful. The user message contains the conversation as "
                        "JSON, with role and content fields. Respect its conversation order. "
                        "Do not mention the transport, Codex, or local environment unless asked."
                    ),
                    "developerInstructions": system or "Respond naturally to the user.",
                    "dynamicTools": dynamic_tools,
                },
            )
            thread_id = result["thread"]["id"]
            state = {
                "queue": asyncio.Queue(),
                "created": time.monotonic(),
                "request_id": record["id"] if record else None,
            }
            self.threads[thread_id] = state
            try:
                result = await self.request(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [
                            {
                                "type": "text",
                                "text": json.dumps(transcript, ensure_ascii=False),
                            },
                            *images,
                        ],
                        "effort": effort,
                    },
                )
            except BaseException:
                await self.finish_thread(thread_id, interrupt=True)
                raise
            state["turn_id"] = result["turn"]["id"]
        suspended = False
        text_parts = []
        phases = {}
        completed_messages = []
        usage = {}
        try:
            async with asyncio.timeout(timeout):
                while True:
                    message = await state["queue"].get()
                    if message.get("disconnect"):
                        raise RuntimeError(
                            "The local Codex connection was interrupted."
                        )
                    method = message.get("method", "")
                    params = message.get("params", {})
                    if method == "item/tool/call":
                        call_id = params["callId"]
                        self.tool_requests[call_id] = (thread_id, message["id"])
                        suspended = True
                        yield {
                            "kind": "result",
                            "text": "".join(text_parts),
                            "tools": [
                                {
                                    "id": call_id,
                                    "name": params["tool"],
                                    "arguments": params["arguments"],
                                }
                            ],
                            "usage": usage,
                        }
                        return
                    if "id" in message:
                        await self.write(
                            {
                                "id": message["id"],
                                "error": {
                                    "code": -32601,
                                    "message": "Use AstrBot tools only",
                                },
                            }
                        )
                    if method == "item/started":
                        item = params.get("item", {})
                        phases[item.get("id")] = item.get("phase")
                    elif method == "item/agentMessage/delta":
                        if phases.get(params.get("itemId")) != "commentary":
                            delta = params.get("delta", "")
                            text_parts.append(delta)
                            yield {"kind": "text", "text": delta}
                    elif method == "item/completed":
                        item = params.get("item", {})
                        if (
                            item.get("type") == "agentMessage"
                            and item.get("phase") != "commentary"
                        ):
                            completed_messages.append(item.get("text", ""))
                    elif method == "thread/tokenUsage/updated":
                        usage = params.get("tokenUsage", {}).get("last", {})
                    elif method == "turn/completed":
                        turn = params["turn"]
                        if turn.get("status") != "completed":
                            error = turn.get("error") or {}
                            raise RuntimeError(
                                error.get("message", f"Codex turn {turn.get('status')}")
                            )
                        text = "\n".join(completed_messages) or "".join(text_parts)
                        if not text:
                            raise RuntimeError(
                                "Codex completed without an assistant response."
                            )
                        await self.finish_thread(thread_id)
                        yield {
                            "kind": "result",
                            "text": text,
                            "tools": [],
                            "usage": usage,
                        }
                        return
        except Exception as exc:
            REQUESTS.update(
                record, "failed", error="模型请求失败：" + type(exc).__name__
            )
            raise
        finally:
            if not suspended:
                await self.finish_thread(thread_id, interrupt=True)

    async def close(self):
        """Stop this adapter's child process and release waiting requests."""
        if self.process and self.process.returncode is None:
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.process = None
        self.threads.clear()
        self.tool_requests.clear()
