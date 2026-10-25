"""Track bounded, in-memory request progress without storing chat transcripts."""

import asyncio
import contextvars
import time
import uuid
import weakref
from collections import OrderedDict

TERMINAL = {"sent", "failed", "cancelled", "skipped", "no_delivery"}


class RequestRegistry:
    """Own request metadata and per-event cancellation, isolated by request ID."""

    def __init__(self):
        self.records = OrderedDict()
        self.owners = {}
        self.clients = weakref.WeakSet()
        self.context = contextvars.ContextVar("local_codex_request", default=None)

    def current(self, session_id=None):
        record = self.records.get(self.context.get())
        return (
            record
            if record
            and record.get("inference_active")
            and session_id == record["session"]
            else None
        )

    def begin(self, event):
        """Associate the current pipeline task with a single incoming event.

        Args:
            event: AstrBot event owning the request.

        Returns:
            Mutable metadata record; no model reasoning or full prompt is retained.
        """
        record = self.records.get(event.get_extra("_local_codex_request_id"))
        if record:
            self.context.set(record["id"])
            return record
        request_id = uuid.uuid4().hex[:16]
        record = {
            "id": request_id,
            "session": event.unified_msg_origin,
            "sender": event.get_sender_name() or event.get_sender_id(),
            "preview": (event.get_message_str() or "[图片或附件]")[:120],
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "queued",
            "inference_active": False,
            "accepted_sends": 0,
            "tool_count": 0,
            "steps": [],
        }
        self.records[request_id] = record
        task = asyncio.current_task()
        self.owners[request_id] = (weakref.ref(event), weakref.ref(task))
        event.set_extra("_local_codex_request_id", request_id)
        self.context.set(request_id)
        self.update(record, "queued")

        def completed(done):
            record["inference_active"] = False
            self.owners.pop(request_id, None)
            for client in list(self.clients):
                cleanup = asyncio.create_task(client.finish_request(request_id))
                cleanup.add_done_callback(
                    lambda result: (
                        result.exception() if not result.cancelled() else None
                    )
                )
            if done.cancelled() or record["status"] == "cancelling":
                self.update(record, "cancelled", force=True)
            elif done.exception() is not None:
                self.update(record, "failed", error="处理任务异常，请查看 AstrBot 日志")
            elif record["status"] not in TERMINAL:
                if record["accepted_sends"]:
                    self.update(record, "sent")
                elif record["status"] == "queued":
                    self.update(record, "skipped")
                else:
                    self.update(record, "no_delivery")
            while len(self.records) > 200:
                removable = next(
                    (
                        key
                        for key, item in self.records.items()
                        if item["status"] in TERMINAL
                    ),
                    None,
                )
                if removable is None:
                    break
                self.records.pop(removable)

        task.add_done_callback(completed)
        return record

    def update(self, record, status=None, *, force=False, **values):
        """Apply metadata and a bounded visible phase transition.

        Args:
            record: Existing record or None for non-chat model calls.
            status: Optional phase name.
            force: Allow a terminal transition during explicit cancellation.
            **values: Public metadata, excluding prompts and tool arguments.
        """
        if record is None:
            return
        record.update(values)
        record["updated_at"] = time.time()
        if status and (force or record["status"] not in TERMINAL):
            if status != record["status"] or not record["steps"]:
                record["steps"].append({"status": status, "at": time.time()})
                record["steps"] = record["steps"][-40:]
            record["status"] = status

    async def cancel(self, request_id):
        """Cancel only the selected pipeline, including queued work and tools.

        Args:
            request_id: Exact request identifier selected by the administrator.

        Returns:
            Whether cancellation was accepted. Sending cannot be revoked.
        """
        record = self.records.get(request_id)
        owner = self.owners.get(request_id)
        if (
            not record
            or not owner
            or record["status"] in TERMINAL | {"sending", "cancelling"}
        ):
            return False
        event, task = owner[0](), owner[1]()
        if event is None or task is None or task.done():
            return False
        self.update(record, "cancelling")
        record["inference_active"] = False
        event.set_extra("agent_stop_requested", True)
        stop_signal = event.get_extra("_background_stop_signal")
        if stop_signal is not None:
            stop_signal.set()
        event.stop_event()
        task.cancel()
        return True


REQUESTS = RequestRegistry()
