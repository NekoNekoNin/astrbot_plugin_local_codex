"""Choose reasoning effort from the current task without another model call."""

import re
from dataclasses import dataclass

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
CONFIGURED_EFFORTS = ("auto", "inherit", "low", "medium", "high", "xhigh", "max")
_INJECTED_BLOCKS = re.compile(
    r"<(system_context|system_reminder|memories|user_profiles|notes)\b[^>]*>"
    r".*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FOLLOW_UP = re.compile(
    r"^(?:那|那么|这个|那个|这|它|上面|刚才|继续|然后|再|换成|如果|为什么|怎么|还有|也|so\b|then\b|what about\b)",
    re.IGNORECASE,
)
_CASUAL = re.compile(
    r"^(?:你好|您好|嗨|哈[喽啰]|早[上安]?|晚安|谢谢|谢啦|好的?|行|嗯+|哦+|哈哈+|嘿嘿+|拜拜|再见|收到|辛苦了|不客气|没事|抱抱|摸摸|hi|hello|hey|thanks?|ok(?:ay)?|good night)[呀啊哦呢啦哟~～!！。,.，\s]*$",
    re.IGNORECASE,
)
_SOCIAL_QUESTION = re.compile(
    r"(?:(?:你好|您好|嗨|哈[喽啰]|早上好|早安|中午好|下午好|晚上好|晚安)"
    r"[呀啊哦呢啦哟~～!！。,.，\s]*)?"
    r"(?:(?:你|您)?(?:最近|今天|这几天)?(?:过得|过的)怎么样|"
    r"(?:你|您)(?:最近|今天|这几天)?怎么样|(?:最近|今天|这几天)怎么样|"
    r"(?:你|您)?(?:最近|今天)?(?:在忙什么|在干嘛|在做什么|忙什么|忙啥|"
    r"吃饭了吗|吃了吗|吃饭了没|吃了没|还好吗|还好吧|好吗))"
    r"[呀啊哦呢啦哟~～!！?？。,.，\s]*",
)
_DEEP = re.compile(
    r"深入(?:分析|推理|思考|研究)|深度(?:分析|推理|思考|研究)|全面(?:分析|论证)|"
    r"严谨(?:证明|推导)|详细推导|最高(?:强度|档位)|彻底分析|"
    r"think deeply|in.depth analysis|rigorous(?:ly)? (?:prove|derive)",
    re.IGNORECASE,
)
_COMPLEX = re.compile(
    r"排(?:查|错)|报错|异常|修复|诊断|调试|堆栈|崩溃|架构|重构|并发|死锁|"
    r"推导|证明|多步|连锁|裁定|伤害计算|最优|最[好佳].*(?:打|方案|选择)|"
    r"怎么打|如何打|展开路线|效果互动|整合包.*(?:冲突|兼容)|"
    r"debug|traceback|stack trace|race condition|deadlock|root cause|optimal|architecture",
    re.IGNORECASE,
)
_TASK = re.compile(
    r"分析|比较|对比|设计|规划|制定|计算|评估|实现|编写|帮我做|解决|推荐|"
    r"analy[sz]e|compare|design|plan|implement|calculate|evaluate",
    re.IGNORECASE,
)
_LOOKUP = re.compile(
    r"查|搜|百科|教程|模组|mod\b|卡牌|资料|下载量|版本|介绍|翻译|总结|"
    r"什么|为何|为什么|怎么|如何|哪|是否|能否|可以|多少|几[个张点岁]|[?？]|"
    r"\b(?:what|why|how|which|where|when|find|search|lookup|explain|summari[sz]e|translate)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EffortDecision:
    """Keep the selected effort and its user-visible explanation together."""

    effort: str
    requested: str
    reason: str
    source: str


def resolve_effort(
    messages,
    *,
    configured="inherit",
    provider_effort="xhigh",
    override=None,
    supported_efforts=None,
):
    """Resolve overrides, classify real user input, and honor the model catalog.

    Args:
        messages: Prepared OpenAI-format messages. System and tool text is ignored.
        configured: Plugin selection, including automatic and inherited modes.
        provider_effort: Provider default used by inherited mode.
        override: Explicit per-model effort; this takes precedence over the plugin.
        supported_efforts: Catalog values, or None if support is unknown.

    Returns:
        An EffortDecision with the actual model-compatible effort and explanation.

    Raises:
        ValueError: An explicit configured effort is invalid.
    """
    selected = override if override is not None else configured
    source = "model_override" if override is not None else "manual"
    if selected == "inherit":
        selected, source = provider_effort, "provider"
    if selected not in CONFIGURED_EFFORTS or selected == "inherit":
        raise ValueError("Invalid local Codex reasoning effort.")
    requested = selected
    if selected == "auto":
        source = "auto"
        turns = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user" or message.get("_no_save"):
                continue
            content = message.get("content", "")
            has_image = False
            if isinstance(content, list):
                has_image = any(
                    part.get("type") in ("image_url", "image", "input_image")
                    for part in content
                    if isinstance(part, dict)
                )
                text = "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in ("text", "input_text")
                )
            else:
                text = content if isinstance(content, str) else ""
            text = _INJECTED_BLOCKS.sub("", text).strip()
            if text.startswith("[系统工具]:"):
                continue
            has_image = has_image or bool(
                re.search(r"\[图片\d*\]|<图片附件>|\[Image Attachment", text)
            )
            text = re.sub(r"<图片附件>.*?</图片附件>", "", text, flags=re.DOTALL)
            text = re.sub(r"\[Image Attachment[^\]]*\]", "", text).strip()
            # AngelHeart prefixes real messages with a sender and a timestamp.
            text = re.sub(
                r"(?m)^\s*\[[^\]\n]*\(ID: [^\]\n]*\)\]"
                r"(?:\s*\([^\n)]*\))?\s*:\s*",
                "",
                text,
            )
            text = re.sub(
                r"(?m)^\[(?:群友\(历史记录\)|历史记录)\]\s*\[内容: 文本\]\s*",
                "",
                text,
            ).strip()
            if text or has_image:
                turns.append((text, has_image))
        text, has_image = turns[-1] if turns else ("", False)
        casual = bool(_CASUAL.fullmatch(text) or _SOCIAL_QUESTION.fullmatch(text))
        inherited = False
        if (
            len(turns) > 1
            and len(text) <= 45
            and not casual
            and _FOLLOW_UP.search(text)
        ):
            # Follow short follow-up chains back to one bounded real task.
            for previous_text, previous_image in reversed(turns[-5:-1]):
                text = previous_text[-3000:] + "\n" + text
                has_image = has_image or previous_image
                if len(previous_text) > 45 or not _FOLLOW_UP.search(previous_text):
                    break
            inherited = True
        if _DEEP.search(text):
            selected, reason = "xhigh", "明确要求深入推理"
        elif (
            _COMPLEX.search(text)
            or (
                _TASK.search(text)
                and (
                    len(text) >= 240
                    or len(re.findall(r"同时|并且|还要|要求|条件|[;；]|\band\b", text))
                    >= 2
                )
            )
            or (len(text) >= 700 and not casual)
        ):
            selected, reason = "high", "多条件分析、排错或游戏推演"
        elif casual and not has_image:
            selected, reason = "low", "日常寒暄，优先响应速度"
        elif has_image or _LOOKUP.search(text) or _TASK.search(text) or len(text) > 120:
            selected, reason = "medium", "图片理解、资料查阅或一般问题"
        else:
            selected, reason = "low", "简短交流，优先响应速度"
        if inherited:
            reason += "（承接上一条问题）"
    else:
        reason = {
            "model_override": "使用模型单独指定的思考强度",
            "provider": "跟随服务商默认思考强度",
            "manual": "使用手动指定的思考强度",
        }[source]
    supported = set(supported_efforts or ()) & set(EFFORT_ORDER)
    if supported and selected not in supported:
        requested_rank = EFFORT_ORDER.index(selected)
        lower = [e for e in EFFORT_ORDER[:requested_rank] if e in supported]
        selected = (
            lower[-1] if lower else next(e for e in EFFORT_ORDER if e in supported)
        )
        reason += f"；当前模型支持范围内改用 {selected}"
    return EffortDecision(selected, requested, reason, source)
