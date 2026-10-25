"""Expose the locally authenticated Codex CLI as an AstrBot provider."""

import json
import weakref
from contextlib import aclosing
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.register import (
    provider_cls_map,
    provider_registry,
    register_provider_adapter,
)
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .client import CodexClient
from .effort_policy import resolve_effort
from .request_hooks import RequestHooks
from .request_state import REQUESTS

INSTANCES = weakref.WeakSet()
TEMPLATE = {
    "id": "local-codex",
    "provider": "local-codex",
    "type": "local_codex_openai",
    "provider_type": "chat_completion",
    "enable": True,
    "key": ["local-codex-managed"],
    "timeout": 300,
    "reasoning_effort": "xhigh",
}


@register_provider_adapter(
    "local_codex_openai",
    "Use the local Codex account session through app-server",
    default_config_tmpl=TEMPLATE,
    provider_display_name="本地 Codex / OpenAI",
)
class LocalCodexProvider(ProviderOpenAIOfficial):
    """Reuse AstrBot message preparation and Codex-managed authentication."""

    def __init__(self, provider_config, provider_settings):
        config = {**TEMPLATE, **provider_config}
        super().__init__(config, provider_settings)
        self.codex = CodexClient(
            Path.home() / ".codex",
            Path(get_astrbot_data_path()) / "plugin_data/local_codex/workspace",
        )
        INSTANCES.add(self)

    async def get_models(self):
        """Return the installed CLI's model catalog."""
        return await self.codex.models()

    async def text_chat(self, *args, **kwargs):
        """Collect one ordinary AstrBot model response."""
        async with aclosing(self.text_chat_stream(*args, **kwargs)) as stream:
            async for response in stream:
                if not response.is_chunk:
                    return response
        raise RuntimeError("Codex returned no final response.")

    async def text_chat_stream(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        audio_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        tool_choice="auto",
        **kwargs,
    ):
        """Stream text and preserve AstrBot-owned function calls."""
        record = REQUESTS.current(session_id)
        try:
            payload, _ = await self._prepare_chat_payload(
                prompt,
                image_urls,
                audio_urls,
                contexts,
                system_prompt,
                tool_calls_result,
                model=model,
                extra_user_content_parts=extra_user_content_parts,
                **kwargs,
            )
            tools = (
                func_tool.openai_schema() if func_tool and not func_tool.empty() else []
            )
            selected = "auto"
            settings_path = (
                Path(get_astrbot_data_path())
                / "config/astrbot_plugin_local_codex_config.json"
            )
            if settings_path.exists():
                try:
                    settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
                    if not isinstance(settings, dict):
                        raise ValueError("Reasoning settings must be an object.")
                    selected = settings.get("reasoning_effort", "auto")
                except (OSError, ValueError) as exc:
                    selected = "inherit"
                    logger.warning(
                        "Could not load local Codex reasoning settings; using provider defaults (%s).",
                        type(exc).__name__,
                    )
            extra_body = self.provider_config.get("custom_extra_body", {})
            decision = resolve_effort(
                payload["messages"],
                configured=selected,
                provider_effort=self.provider_config.get("reasoning_effort", "xhigh"),
                override=(
                    extra_body.get("reasoning_effort")
                    if isinstance(extra_body, dict)
                    else None
                ),
                supported_efforts=await self.codex.supported_efforts(payload["model"]),
            )
            effort = decision.effort
            continuing = any(
                item.get("role") == "tool"
                and item.get("tool_call_id") in self.codex.tool_requests
                for item in payload["messages"]
            )
            if not continuing:
                REQUESTS.update(
                    record,
                    effort=effort,
                    effort_reason=decision.reason,
                    effort_source=decision.source,
                    model=payload["model"],
                )
            async for event in self.codex.generate(
                payload["messages"],
                payload["model"],
                tools=tools,
                effort=effort,
                timeout=self.timeout,
                request_record=record,
                image_resolver=self._image_ref_to_data_url,
            ):
                if event["kind"] == "text":
                    yield LLMResponse(
                        role="assistant", completion_text=event["text"], is_chunk=True
                    )
                    continue
                calls = event.get("tools", [])
                usage = event.get("usage", {})
                cached = usage.get("cachedInputTokens", 0)
                yield LLMResponse(
                    role="tool" if calls else "assistant",
                    completion_text=event["text"],
                    tools_call_name=[call["name"] for call in calls],
                    tools_call_args=[call["arguments"] for call in calls],
                    tools_call_ids=[call["id"] for call in calls],
                    usage=TokenUsage(
                        input_other=max(0, usage.get("inputTokens", 0) - cached),
                        input_cached=cached,
                        output=usage.get("outputTokens", 0),
                    ),
                )
        except Exception as exc:
            REQUESTS.update(
                record, "failed", error="模型请求失败：" + type(exc).__name__
            )
            raise

    async def terminate(self):
        """Release this provider's local subprocess and unused HTTP client."""
        await self.codex.close()
        await self.client.close()
        INSTANCES.discard(self)


class Main(RequestHooks, Star):
    """Register the local provider for the desktop application's lifecycle."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context, config)
        self.register_request_api(context)

    @filter.on_waiting_llm_request(priority=100)
    async def request_waiting(self, event):
        await self.track_waiting(event)

    @filter.on_llm_request(priority=-100)
    async def request_preparing(self, event, req):
        await self.track_request(event, req)

    @filter.on_agent_begin(priority=100)
    async def request_started(self, event, run_context):
        await self.track_agent(event, run_context)

    @filter.on_using_llm_tool(priority=100)
    async def request_tool(self, event, tool, tool_args):
        await self.track_tool(event, tool, tool_args)

    @filter.on_llm_tool_respond(priority=-100)
    async def request_tool_done(self, event, tool, tool_args, tool_result):
        await self.track_tool_done(event, tool, tool_args, tool_result)

    @filter.on_agent_done(priority=-100)
    async def request_generated(self, event, run_context, response):
        await self.track_result(event, run_context, response)

    @filter.after_message_sent(priority=-200)
    async def request_sent(self, event):
        await self.track_delivered(event)

    async def terminate(self):
        """Stop adapter-owned connections when the plugin is unloaded."""
        for provider in list(INSTANCES):
            await provider.terminate()
        metadata = provider_cls_map.get("local_codex_openai")
        if metadata and metadata.cls_type is LocalCodexProvider:
            provider_cls_map.pop("local_codex_openai")
            if metadata in provider_registry:
                provider_registry.remove(metadata)
