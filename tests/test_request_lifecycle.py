"""Exercise real task cancellation and adapter outcomes without network traffic."""

import asyncio
import importlib.util
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

PACKAGE_NAME = "local_codex_lifecycle_under_test"
PACKAGE_PATH = Path(__file__).parents[1]
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = PACKAGE

for module_name in ("request_state", "request_hooks", "client", "effort_policy"):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.{module_name}", PACKAGE_PATH / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

STATE = sys.modules[f"{PACKAGE_NAME}.request_state"]
HOOKS = sys.modules[f"{PACKAGE_NAME}.request_hooks"]
CLIENT = sys.modules[f"{PACKAGE_NAME}.client"]

ASTRBOT_STUBS = {}
for module_name, attributes in {
    "astrbot.api": {
        "AstrBotConfig": dict,
        "logger": types.SimpleNamespace(warning=lambda *args, **kwargs: None),
    },
    "astrbot.api.event": {
        "filter": types.SimpleNamespace(
            **{
                hook: lambda **kwargs: lambda handler: handler
                for hook in (
                    "on_waiting_llm_request",
                    "on_llm_request",
                    "on_agent_begin",
                    "on_using_llm_tool",
                    "on_llm_tool_respond",
                    "on_agent_done",
                    "after_message_sent",
                )
            }
        )
    },
    "astrbot.api.star": {"Context": object, "Star": object},
    "astrbot.core.provider.entities": {
        "LLMResponse": types.SimpleNamespace,
        "TokenUsage": types.SimpleNamespace,
    },
    "astrbot.core.provider.register": {
        "provider_cls_map": {},
        "provider_registry": [],
        "register_provider_adapter": lambda *args, **kwargs: lambda cls: cls,
    },
    "astrbot.core.provider.sources.openai_source": {"ProviderOpenAIOfficial": object},
    "astrbot.core.utils.astrbot_path": {"get_astrbot_data_path": lambda: "."},
}.items():
    stub = types.ModuleType(module_name)
    stub.__dict__.update(attributes)
    ASTRBOT_STUBS[module_name] = stub

with patch.dict(sys.modules, ASTRBOT_STUBS):
    MAIN_SPEC = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.main", PACKAGE_PATH / "main.py"
    )
    MAIN = importlib.util.module_from_spec(MAIN_SPEC)
    sys.modules[MAIN_SPEC.name] = MAIN
    MAIN_SPEC.loader.exec_module(MAIN)


class FakeEvent:
    """Represent one event with the same scope and asynchronous send boundaries."""

    def __init__(self, session="qq:group:123", text="question"):
        self.unified_msg_origin = session
        self.text = text
        self.extras = {}
        self.stopped = False
        self.sent = []
        self.send_error = None

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_sender_name(self):
        return "Test sender"

    def get_sender_id(self):
        return "456"

    def get_message_str(self):
        return self.text

    def stop_event(self):
        self.stopped = True

    async def send(self, message):
        await asyncio.sleep(0)
        if self.send_error:
            raise self.send_error
        if message is not None and message.chain:
            self.sent.append(message)

    async def send_streaming(self, generator, *args, **kwargs):
        async for message in generator:
            await self.send(message)


class NativeStub(CLIENT.CodexClient):
    """Replace only native I/O, retaining the production generator and cleanup."""

    def __init__(self):
        super().__init__(Path("unused-home"), Path("unused-workspace"))
        self.rpc_calls = []
        self.writes = []
        self.turn_started = asyncio.Queue()
        self.next_thread = 0

    async def start(self):
        return None

    async def request(self, method, params):
        self.rpc_calls.append((method, params))
        if method == "thread/start":
            self.next_thread += 1
            return {"thread": {"id": f"thread-{self.next_thread}"}}
        if method == "turn/start":
            await self.turn_started.put(params["threadId"])
            return {"turn": {"id": f"turn-{params['threadId']}"}}
        if method == "model/list":
            return {
                "data": [
                    {
                        "id": "catalog-fast",
                        "model": "fast-model",
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": "low"},
                            {"reasoningEffort": "medium"},
                        ],
                    },
                    {
                        "id": "catalog-deep",
                        "model": "deep-model",
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": "high"},
                            {"reasoningEffort": "xhigh"},
                        ],
                    },
                ]
            }
        return {}

    async def write(self, message):
        self.writes.append(message)


class RequestLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.registry = STATE.RequestRegistry()
        HOOKS.REQUESTS = self.registry
        CLIENT.REQUESTS = self.registry
        MAIN.REQUESTS = self.registry
        self.hooks = HOOKS.RequestHooks()
        self.started_tasks = []

    async def asyncTearDown(self):
        for task in self.started_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.started_tasks, return_exceptions=True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.started_tasks.append(task)
        return task

    async def test_cancel_queued_event_keeps_same_session_other_events_alive(self):
        lock = asyncio.Lock()
        first_running = asyncio.Event()
        release_first = asyncio.Event()
        queued_ready = asyncio.Event()
        completed = []
        events = [FakeEvent(text=f"event-{index}") for index in range(3)]

        async def pipeline(index):
            await self.hooks.track_waiting(events[index])
            if index == 1:
                queued_ready.set()
            async with lock:
                if index == 0:
                    first_running.set()
                    await release_first.wait()
                completed.append(index)

        first = self.spawn(pipeline(0))
        await first_running.wait()
        queued = self.spawn(pipeline(1))
        await queued_ready.wait()
        third = self.spawn(pipeline(2))
        request_id = events[1].get_extra("_local_codex_request_id")
        self.assertTrue(await self.registry.cancel(request_id))
        with self.assertRaises(asyncio.CancelledError):
            await queued
        release_first.set()
        await asyncio.gather(first, third)
        await asyncio.sleep(0)
        self.assertEqual(completed, [0, 2])
        self.assertTrue(events[1].stopped)
        self.assertFalse(events[0].stopped)
        self.assertFalse(events[2].stopped)
        self.assertEqual(self.registry.records[request_id]["status"], "cancelled")
        self.assertFalse(await self.registry.cancel(request_id))

    async def test_cancel_is_recorded_when_plugin_hook_swallows_task_cancel(self):
        event = FakeEvent()
        hook_waiting = asyncio.Event()

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            hook_waiting.set()
            try:
                await asyncio.Event().wait()
            except BaseException:
                # The framework hook dispatcher catches BaseException, including cancellation.
                return

        owner = self.spawn(pipeline())
        await hook_waiting.wait()
        request_id = event.get_extra("_local_codex_request_id")
        self.assertTrue(await self.registry.cancel(request_id))
        await owner
        await asyncio.sleep(0)
        self.assertTrue(event.stopped)
        self.assertEqual(self.registry.records[request_id]["status"], "cancelled")

    async def test_concurrent_event_contexts_stay_separate(self):
        both_ready = asyncio.Event()
        count = 0
        observations = []

        async def pipeline(event):
            nonlocal count
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            count += 1
            if count == 2:
                both_ready.set()
            await both_ready.wait()
            await asyncio.sleep(0)
            observations.append(
                (
                    event.unified_msg_origin,
                    self.registry.current(event.unified_msg_origin)["session"],
                )
            )

        await asyncio.gather(
            self.spawn(pipeline(FakeEvent(session="qq:group:1"))),
            self.spawn(pipeline(FakeEvent(session="qq:private:1"))),
        )
        self.assertEqual(
            sorted(observations),
            [("qq:group:1", "qq:group:1"), ("qq:private:1", "qq:private:1")],
        )

    async def test_background_learning_does_not_inherit_request_ownership(self):
        observed = []

        async def background_learning():
            await asyncio.sleep(0)
            observed.append(self.registry.current())

        async def pipeline():
            event = FakeEvent()
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            background = self.spawn(background_learning())
            await background

        await self.spawn(pipeline())
        self.assertEqual(observed, [None])

    async def test_cancelling_chat_keeps_inherited_background_native_call_alive(self):
        client = NativeStub()
        event = FakeEvent()
        background_created = asyncio.Event()
        background_tasks = []

        async def background_learning():
            async for _ in client.generate(
                [{"role": "user", "content": "Organize learned memories"}],
                "fast-model",
            ):
                pass

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            background_tasks.append(self.spawn(background_learning()))
            background_created.set()
            async for _ in client.generate(
                [{"role": "user", "content": "Answer the chat"}],
                "fast-model",
                request_record=self.registry.current(event.unified_msg_origin),
            ):
                pass

        owner = self.spawn(pipeline())
        await background_created.wait()
        thread_ids = [
            await asyncio.wait_for(client.turn_started.get(), 1),
            await asyncio.wait_for(client.turn_started.get(), 1),
        ]
        request_id = event.get_extra("_local_codex_request_id")
        background_thread = next(
            thread_id
            for thread_id in thread_ids
            if client.threads[thread_id].get("request_id") is None
        )
        chat_thread = next(
            thread_id
            for thread_id in thread_ids
            if client.threads[thread_id].get("request_id") == request_id
        )
        self.assertTrue(await self.registry.cancel(request_id))
        with self.assertRaises(asyncio.CancelledError):
            await owner
        await asyncio.sleep(0)
        self.assertNotIn(chat_thread, client.threads)
        self.assertIn(background_thread, client.threads)
        self.assertFalse(background_tasks[0].done())
        interrupted = [
            params["threadId"]
            for method, params in client.rpc_calls
            if method == "turn/interrupt"
        ]
        self.assertEqual(interrupted, [chat_thread])

    async def test_sending_is_not_cancelled_after_adapter_delivery_started(self):
        event = FakeEvent()
        sending = asyncio.Event()
        release_send = asyncio.Event()

        async def adapter_send(message):
            sending.set()
            await release_send.wait()

        event.send = adapter_send

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_result(event, None, None)
            await event.send(types.SimpleNamespace(chain=["reply"]))
            await self.hooks.track_delivered(event)

        owner = self.spawn(pipeline())
        await sending.wait()
        request_id = event.get_extra("_local_codex_request_id")
        self.assertFalse(await self.registry.cancel(request_id))
        self.assertFalse(owner.cancelled())
        self.assertFalse(event.stopped)
        release_send.set()
        await owner
        await asyncio.sleep(0)
        self.assertEqual(self.registry.records[request_id]["status"], "sent")

    async def test_segment_failure_remains_failed_after_later_success(self):
        event = FakeEvent()

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            await self.hooks.track_result(event, None, None)
            await event.send(types.SimpleNamespace(chain=["first segment"]))
            event.send_error = OSError("adapter disconnected")
            with self.assertRaises(OSError):
                await event.send(types.SimpleNamespace(chain=["failed segment"]))
            event.send_error = None
            await event.send(types.SimpleNamespace(chain=["last segment"]))
            await self.hooks.track_delivered(event)

        await self.spawn(pipeline())
        await asyncio.sleep(0)
        record = self.registry.records[event.get_extra("_local_codex_request_id")]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["accepted_sends"], 2)
        self.assertNotIn("sent", [step["status"] for step in record["steps"]])

    async def test_segmented_reply_only_finishes_after_delivery_hook(self):
        event = FakeEvent()
        states = []

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_result(event, None, None)
            for text in ("first", "second"):
                await event.send(types.SimpleNamespace(chain=[text]))
                states.append(
                    self.registry.records[event.get_extra("_local_codex_request_id")][
                        "status"
                    ]
                )
            await self.hooks.track_delivered(event)

        await self.spawn(pipeline())
        await asyncio.sleep(0)
        record = self.registry.records[event.get_extra("_local_codex_request_id")]
        self.assertEqual(states, ["sending", "sending"])
        self.assertEqual(record["status"], "sent")
        self.assertEqual(record["accepted_sends"], 2)

    async def test_empty_stream_does_not_claim_delivery(self):
        event = FakeEvent()

        async def empty_stream():
            if False:
                yield None

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_result(event, None, None)
            await event.send_streaming(empty_stream())

        await self.spawn(pipeline())
        await asyncio.sleep(0)
        record = self.registry.records[event.get_extra("_local_codex_request_id")]
        self.assertEqual(record["status"], "no_delivery")
        self.assertEqual(record["accepted_sends"], 0)

    async def test_stream_send_failure_cannot_become_sent(self):
        event = FakeEvent()

        async def stream():
            yield types.SimpleNamespace(chain=["one"])
            event.send_error = OSError("adapter disconnected")
            yield types.SimpleNamespace(chain=["two"])

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_result(event, None, None)
            with self.assertRaises(OSError):
                await event.send_streaming(stream())
            await self.hooks.track_delivered(event)

        await self.spawn(pipeline())
        await asyncio.sleep(0)
        record = self.registry.records[event.get_extra("_local_codex_request_id")]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["accepted_sends"], 1)

    async def test_cancelling_native_model_only_interrupts_its_own_thread(self):
        client = NativeStub()
        events = [FakeEvent(session=f"qq:group:{index}") for index in range(2)]

        async def pipeline(event):
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            async for _ in client.generate(
                [{"role": "user", "content": event.text}],
                "fast-model",
                request_record=self.registry.current(event.unified_msg_origin),
            ):
                pass

        first = self.spawn(pipeline(events[0]))
        first_thread = await asyncio.wait_for(client.turn_started.get(), 1)
        second = self.spawn(pipeline(events[1]))
        second_thread = await asyncio.wait_for(client.turn_started.get(), 1)
        self.assertTrue(
            await self.registry.cancel(events[0].get_extra("_local_codex_request_id"))
        )
        with self.assertRaises(asyncio.CancelledError):
            await first
        await asyncio.sleep(0)
        self.assertNotIn(first_thread, client.threads)
        self.assertIn(second_thread, client.threads)
        self.assertFalse(second.done())
        interrupts = [
            params["threadId"]
            for method, params in client.rpc_calls
            if method == "turn/interrupt"
        ]
        self.assertEqual(interrupts, [first_thread])
        self.assertTrue(
            await self.registry.cancel(events[1].get_extra("_local_codex_request_id"))
        )
        with self.assertRaises(asyncio.CancelledError):
            await second

    async def test_tool_suspension_is_interrupted_when_pipeline_is_cancelled(self):
        client = NativeStub()
        event = FakeEvent()
        tool_running = asyncio.Event()

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            async for result in client.generate(
                [{"role": "user", "content": "lookup"}],
                "fast-model",
                request_record=self.registry.current(event.unified_msg_origin),
            ):
                self.assertEqual(result["tools"][0]["id"], "tool-call-1")
            await self.hooks.track_tool(event, types.SimpleNamespace(name="lookup"), {})
            tool_running.set()
            await asyncio.Event().wait()

        owner = self.spawn(pipeline())
        thread_id = await asyncio.wait_for(client.turn_started.get(), 1)
        await client.threads[thread_id]["queue"].put(
            {
                "id": 99,
                "method": "item/tool/call",
                "params": {"callId": "tool-call-1", "tool": "lookup", "arguments": {}},
            }
        )
        await asyncio.wait_for(tool_running.wait(), 1)
        self.assertIn("tool-call-1", client.tool_requests)
        self.assertTrue(
            await self.registry.cancel(event.get_extra("_local_codex_request_id"))
        )
        with self.assertRaises(asyncio.CancelledError):
            await owner
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(client.threads, {})
        self.assertEqual(client.tool_requests, {})
        self.assertIn(
            ("turn/interrupt", {"threadId": thread_id, "turnId": f"turn-{thread_id}"}),
            client.rpc_calls,
        )

    async def test_model_timeout_marks_failed_and_cleans_native_state(self):
        client = NativeStub()
        event = FakeEvent()

        async def pipeline():
            await self.hooks.track_waiting(event)
            await self.hooks.track_request(event, None)
            with self.assertRaises(TimeoutError):
                async for _ in client.generate(
                    [{"role": "user", "content": "hello"}],
                    "fast-model",
                    timeout=0.01,
                    request_record=self.registry.current(event.unified_msg_origin),
                ):
                    pass

        await self.spawn(pipeline())
        await asyncio.sleep(0)
        record = self.registry.records[event.get_extra("_local_codex_request_id")]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(client.threads, {})
        self.assertEqual(client.tool_requests, {})

    async def test_provider_setup_failures_stay_failed_after_error_reply(self):
        with TemporaryDirectory() as data_directory:
            for failing_stage in ("prepare", "catalog", "native_start"):
                with self.subTest(stage=failing_stage):
                    event = FakeEvent(text=f"test {failing_stage}")
                    provider = MAIN.LocalCodexProvider.__new__(MAIN.LocalCodexProvider)
                    provider.provider_config = {"reasoning_effort": "high"}
                    provider.timeout = 1
                    provider._image_ref_to_data_url = AsyncMock(return_value=None)
                    provider._prepare_chat_payload = AsyncMock(
                        return_value=(
                            {
                                "model": "fast-model",
                                "messages": [{"role": "user", "content": event.text}],
                            },
                            None,
                        )
                    )
                    provider.codex = NativeStub()
                    provider.codex.supported_efforts = AsyncMock(
                        return_value={"low", "medium", "high"}
                    )
                    if failing_stage == "prepare":
                        provider._prepare_chat_payload.side_effect = OSError(
                            "prepare failed"
                        )
                    elif failing_stage == "catalog":
                        provider.codex.supported_efforts.side_effect = OSError(
                            "catalog failed"
                        )
                    else:
                        provider.codex.start = AsyncMock(
                            side_effect=OSError("native startup failed")
                        )

                    async def pipeline():
                        await self.hooks.track_waiting(event)
                        await self.hooks.track_request(event, None)
                        try:
                            async for _ in provider.text_chat_stream(
                                prompt=event.text, session_id=event.unified_msg_origin
                            ):
                                pass
                        except OSError:
                            await event.send(
                                types.SimpleNamespace(chain=["error notification"])
                            )
                        await self.hooks.track_delivered(event)

                    with patch.object(
                        MAIN, "get_astrbot_data_path", return_value=data_directory
                    ):
                        await self.spawn(pipeline())
                    await asyncio.sleep(0)
                    record = self.registry.records[
                        event.get_extra("_local_codex_request_id")
                    ]
                    self.assertEqual(record["status"], "failed")
                    self.assertEqual(record["accepted_sends"], 1)
                    self.assertIn("OSError", record["error"])

    async def test_catalog_uses_each_models_actual_supported_efforts(self):
        client = NativeStub()
        self.assertEqual(
            await client.supported_efforts("fast-model"), {"low", "medium"}
        )
        self.assertEqual(
            await client.supported_efforts("deep-model"), {"high", "xhigh"}
        )
        self.assertEqual(await client.supported_efforts("unknown-model"), set())
        self.assertEqual(
            sum(method == "model/list" for method, _ in client.rpc_calls), 1
        )
        client.catalog_at = time.monotonic() - 301
        await client.supported_efforts("fast-model")
        self.assertEqual(
            sum(method == "model/list" for method, _ in client.rpc_calls), 2
        )


if __name__ == "__main__":
    unittest.main()
