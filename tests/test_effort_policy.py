"""Exercise task routing, injected-context boundaries, and manual overrides."""

import importlib.util
import sys
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "local_codex_effort_policy", Path(__file__).parents[1] / "effort_policy.py"
)
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)
resolve_effort = POLICY.resolve_effort


class EffortPolicyTests(unittest.TestCase):
    def test_everyday_chat_prioritizes_speed(self):
        for text in ("早上好", "谢谢！", "今天有点累，抱抱我", "哈哈哈", "hi"):
            with self.subTest(text=text):
                decision = resolve_effort(
                    [{"role": "user", "content": text}], configured="auto"
                )
                self.assertEqual(decision.effort, "low")
                self.assertEqual(decision.source, "auto")

    def test_ordinary_knowledge_questions_use_medium(self):
        for text in (
            "帮我查一下机械动力这个模组的介绍和下载量",
            "这张游戏王卡的效果是什么？",
            "为什么天空是蓝色的？",
            "Translate this into Chinese: hello world",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    resolve_effort(
                        [{"role": "user", "content": text}], configured="auto"
                    ).effort,
                    "medium",
                )

    def test_short_social_questions_prioritize_speed(self):
        for text in (
            "晚上好呀，今天过得怎么样？",
            "你好，最近过得怎么样？",
            "早上好呀，最近怎么样？",
            "你今天怎么样？",
            "在忙什么呢？",
            "你最近在干嘛？",
            "你在做什么？",
            "吃了吗？",
            "您好，吃饭了吗？",
            "你还好吗？",
            "还好吗？",
        ):
            with self.subTest(text=text):
                decision = resolve_effort(
                    [{"role": "user", "content": text}], configured="auto"
                )
                self.assertEqual(decision.effort, "low")
                self.assertEqual(decision.source, "auto")

    def test_social_question_rule_does_not_hide_real_tasks(self):
        for text, expected in (
            ("晚上好，帮我深入分析这两个缓存方案", "xhigh"),
            ("晚上好，今天过得怎么样？另外帮我深入分析日志", "xhigh"),
            ("你最近在忙什么？帮我排查启动报错", "high"),
            ("你好，机械动力这个模组怎么样？", "medium"),
            ("这个程序怎么样？", "medium"),
            ("你觉得这个设计怎么样？", "medium"),
            ("今天的天气怎么样？", "medium"),
            ("它还好吗？", "medium"),
            ("吃了吗这个梗是什么意思？", "medium"),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    resolve_effort(
                        [{"role": "user", "content": text}], configured="auto"
                    ).effort,
                    expected,
                )

    def test_social_question_resets_complex_followup_classification(self):
        decision = resolve_effort(
            [
                {"role": "user", "content": "深入分析这个并发错误"},
                {"role": "assistant", "content": "详细推理" * 1000},
                {"role": "user", "content": "最近过得怎么样？"},
            ],
            configured="auto",
        )
        self.assertEqual(decision.effort, "low")
        self.assertNotIn("承接", decision.reason)

    def test_game_interactions_and_troubleshooting_use_high(self):
        for text in (
            "帮我看看这个场面我最好怎么打",
            "这两张卡连锁时谁先处理？",
            "这个整合包崩溃了，帮我排查原因",
            "设计一个机器人启动器；要求一键启动；还要托盘控制",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    resolve_effort(
                        [{"role": "user", "content": text}], configured="auto"
                    ).effort,
                    "high",
                )

    def test_explicit_deep_request_uses_xhigh(self):
        decision = resolve_effort(
            [{"role": "user", "content": "请深入分析两种缓存方案的取舍"}],
            configured="auto",
        )
        self.assertEqual(decision.effort, "xhigh")

    def test_system_and_injected_memory_do_not_escalate_chat(self):
        deep_text = "深入分析所有复杂连锁与架构" * 1500
        messages = [
            {"role": "system", "content": deep_text},
            {"role": "developer", "content": deep_text},
            {"role": "user", "content": "你好"},
            {
                "role": "user",
                "content": f"<system_context><memories>{deep_text}</memories></system_context>",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<system_reminder>{deep_text}</system_reminder>",
                    }
                ],
            },
        ]
        self.assertEqual(resolve_effort(messages, configured="auto").effort, "low")

    def test_tool_outputs_do_not_reclassify_original_task(self):
        original = [{"role": "user", "content": "查一下这个模组的介绍"}]
        continuation = original + [
            {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "深入分析并修复所有并发问题" * 500,
            },
        ]
        self.assertEqual(
            resolve_effort(original, configured="auto"),
            resolve_effort(continuation, configured="auto"),
        )

    def test_short_followup_chain_keeps_original_complexity(self):
        messages = [
            {"role": "user", "content": "帮我排查这个整合包的启动报错"},
            {"role": "assistant", "content": "先尝试更新依赖。"},
            {"role": "user", "content": "那换成另一个版本呢"},
            {"role": "assistant", "content": "需要检查加载器兼容性。"},
            {"role": "user", "content": "继续"},
        ]
        decision = resolve_effort(messages, configured="auto")
        self.assertEqual(decision.effort, "high")
        self.assertIn("承接", decision.reason)

    def test_new_greeting_does_not_inherit_previous_complex_task(self):
        decision = resolve_effort(
            [
                {"role": "user", "content": "深入分析这个并发错误"},
                {"role": "assistant", "content": "详细推理" * 1000},
                {"role": "user", "content": "晚安"},
            ],
            configured="auto",
        )
        self.assertEqual(decision.effort, "low")

    def test_sender_name_and_temporary_metadata_do_not_affect_routing(self):
        decision = resolve_effort(
            [
                {
                    "role": "user",
                    "content": "[群友: 深度分析架构师 (ID: 12345)] (2026-10-23 21:30): 你好",
                }
            ],
            configured="auto",
        )
        self.assertEqual(decision.effort, "low")

    def test_image_understanding_has_medium_floor(self):
        decision = resolve_effort(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是谁"},
                        {"type": "image_url", "image_url": {"url": "image.png"}},
                    ],
                }
            ],
            configured="auto",
        )
        self.assertEqual(decision.effort, "medium")

    def test_automatic_image_caption_is_not_user_task_complexity(self):
        decision = resolve_effort(
            [
                {
                    "role": "user",
                    "content": "这是谁 [图片1]\n<图片附件>"
                    + "复杂架构和详细推导" * 1000
                    + "</图片附件>",
                }
            ],
            configured="auto",
        )
        self.assertEqual(decision.effort, "medium")

    def test_manual_and_per_model_choices_take_priority(self):
        messages = [{"role": "user", "content": "深入分析这个架构"}]
        manual = resolve_effort(messages, configured="low")
        model = resolve_effort(messages, configured="auto", override="medium")
        inherited = resolve_effort(
            messages, configured="inherit", provider_effort="xhigh"
        )
        self.assertEqual((manual.effort, manual.source), ("low", "manual"))
        self.assertEqual((model.effort, model.source), ("medium", "model_override"))
        self.assertEqual((inherited.effort, inherited.source), ("xhigh", "provider"))

    def test_catalog_caps_unsupported_effort_without_an_extra_llm_call(self):
        decision = resolve_effort(
            [{"role": "user", "content": "深入思考这个问题"}],
            configured="auto",
            supported_efforts={"low", "medium", "high"},
        )
        self.assertEqual(decision.effort, "high")
        self.assertIn("支持范围", decision.reason)
        fixed = resolve_effort(
            [], configured="max", supported_efforts={"low", "medium", "high", "xhigh"}
        )
        self.assertEqual(fixed.effort, "xhigh")
        lowest = resolve_effort(
            [], configured="low", supported_efforts={"medium", "high"}
        )
        self.assertEqual(lowest.effort, "medium")

    def test_unknown_catalog_keeps_selected_effort(self):
        self.assertEqual(resolve_effort([], configured="xhigh").effort, "xhigh")

    def test_invalid_explicit_configuration_fails_clearly(self):
        for value in ("typo", "", "ultra", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_effort([], configured=value)


if __name__ == "__main__":
    unittest.main()
