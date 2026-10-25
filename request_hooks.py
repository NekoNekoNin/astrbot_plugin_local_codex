"""Observe chat lifecycle through supported AstrBot plugin hooks."""

from quart import jsonify, request

from .request_state import REQUESTS, TERMINAL


class RequestHooks:
    """Register progress APIs and track actual adapter send outcomes."""

    def register_request_api(self, context):
        async def list_requests():
            session = request.args.get("session", "")
            items = [
                {key: value for key, value in item.items() if key != "inference_active"}
                for item in reversed(REQUESTS.records.values())
                if not session or item["session"] == session
            ]
            return jsonify({"ok": True, "items": items[:200]})

        async def cancel_request():
            body = await request.get_json()
            request_id = body.get("id") if isinstance(body, dict) else None
            if not isinstance(request_id, str):
                return jsonify({"ok": False, "message": "请选择要取消的请求"}), 400
            accepted = await REQUESTS.cancel(request_id)
            return jsonify(
                {
                    "ok": accepted,
                    "message": "已请求取消"
                    if accepted
                    else "请求已结束或正在发送，无法取消",
                }
            ), 200 if accepted else 409

        context.register_web_api(
            "/astrbot_plugin_local_codex/requests", list_requests, ["GET"], "请求状态"
        )
        context.register_web_api(
            "/astrbot_plugin_local_codex/requests/cancel",
            cancel_request,
            ["POST"],
            "取消指定请求",
        )

    async def track_waiting(self, event):
        """Capture the pipeline task before it waits on the session lock."""
        record = REQUESTS.begin(event)
        if event.get_extra("_local_codex_send_wrapped"):
            return
        event.set_extra("_local_codex_send_wrapped", True)
        original_send = event.send
        original_stream = event.send_streaming

        async def tracked_send(message):
            if message is None:
                return await original_send(message)
            if record["status"] in {"cancelling", "cancelled"}:
                return None
            REQUESTS.update(record, "sending")
            try:
                result = await original_send(message)
            except Exception:
                REQUESTS.update(
                    record, "failed", error="平台发送失败，请查看 AstrBot 日志"
                )
                raise
            if getattr(message, "chain", None):
                record["accepted_sends"] += 1
            return result

        async def tracked_stream(*args, **kwargs):
            if record["status"] in {"cancelling", "cancelled"}:
                return None
            REQUESTS.update(record, "sending")
            try:
                result = await original_stream(*args, **kwargs)
            except Exception:
                REQUESTS.update(
                    record, "failed", error="平台流式发送失败，请查看 AstrBot 日志"
                )
                raise
            if not record["accepted_sends"]:
                REQUESTS.update(record, "no_delivery")
            return result

        event.send = tracked_send
        event.send_streaming = tracked_stream

    async def track_request(self, event, req):
        record = REQUESTS.begin(event)
        record["inference_active"] = True
        REQUESTS.update(record, "preparing")

    async def track_agent(self, event, run_context):
        record = REQUESTS.begin(event)
        record["inference_active"] = True
        REQUESTS.update(record, "thinking")

    async def track_tool(self, event, tool, tool_args):
        record = REQUESTS.records.get(event.get_extra("_local_codex_request_id"))
        if record:
            REQUESTS.update(
                record,
                "tool",
                tool=getattr(tool, "name", "工具"),
                tool_count=record["tool_count"] + 1,
            )

    async def track_tool_done(self, event, tool, tool_args, tool_result):
        record = REQUESTS.records.get(event.get_extra("_local_codex_request_id"))
        REQUESTS.update(record, "thinking")

    async def track_result(self, event, run_context, response):
        record = REQUESTS.records.get(event.get_extra("_local_codex_request_id"))
        if record:
            record["inference_active"] = False
            REQUESTS.update(record, "generated")

    async def track_delivered(self, event):
        record = REQUESTS.records.get(event.get_extra("_local_codex_request_id"))
        if record and record["accepted_sends"] and record["status"] not in TERMINAL:
            REQUESTS.update(record)
