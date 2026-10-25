"""Exercise image normalization at the real Codex turn boundary."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

PACKAGE_NAME = "local_codex_image_transport_tests"
PACKAGE_PATH = Path(__file__).parents[1]
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package
for name in ("request_state", "client"):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.{name}", PACKAGE_PATH / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
CLIENT = sys.modules[f"{PACKAGE_NAME}.client"]
INLINE_IMAGE = "data:image/png;base64,aW1hZ2UtZml4dHVyZQ=="


class CapturedTurn(Exception):
    """Stop the isolated transport before any model inference."""


class ImageTransportTests(unittest.IsolatedAsyncioTestCase):
    """Check both preserved images and graceful failures without network I/O."""

    async def capture(self, reference, resolver=None, *, historical=False):
        """Build one real turn and capture the outgoing native parameters.

        Args:
            reference: Image reference in OpenAI message format.
            resolver: Optional asynchronous image converter.
            historical: Include a later text-only user turn.

        Returns:
            Captured native turn parameters.
        """
        client = CLIENT.CodexClient(Path("unused-home"), Path("unused-workspace"))
        client.start = AsyncMock()
        client.finish_thread = AsyncMock()
        captured = {}

        async def request(method, params):
            if method == "thread/start":
                return {"thread": {"id": "fixture-thread"}}
            if method == "turn/start":
                captured.update(params)
                raise CapturedTurn()
            return {}

        client.request = request
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": reference}}],
            }
        ]
        if historical:
            messages.append({"role": "user", "content": "Please continue."})
        with self.assertRaises(CapturedTurn):
            async for _ in client.generate(
                messages, "fixture-model", image_resolver=resolver
            ):
                pass
        self.assertEqual(messages[0]["content"][0]["image_url"]["url"], reference)
        return captured

    async def test_valid_inline_image_is_preserved(self):
        turn = await self.capture(INLINE_IMAGE)
        self.assertEqual(turn["input"][1], {"type": "image", "url": INLINE_IMAGE})

    async def test_historical_local_image_is_converted(self):
        resolver = AsyncMock(return_value=INLINE_IMAGE)
        turn = await self.capture("C:/cache/fixture.webp", resolver, historical=True)
        resolver.assert_awaited_once_with("C:/cache/fixture.webp")
        self.assertEqual(turn["input"][1]["url"], INLINE_IMAGE)

    async def test_remote_image_is_converted(self):
        resolver = AsyncMock(return_value=INLINE_IMAGE)
        turn = await self.capture("https://example.invalid/fixture.png", resolver)
        resolver.assert_awaited_once()
        self.assertEqual(turn["input"][1]["url"], INLINE_IMAGE)

    async def test_unavailable_image_keeps_text_without_raw_references(self):
        for resolver in (
            None,
            AsyncMock(return_value=None),
            AsyncMock(side_effect=OSError("fixture failure")),
        ):
            turn = await self.capture("C:/missing/private-image.webp", resolver)
            self.assertEqual(len(turn["input"]), 1)
            self.assertIn("Image unavailable", turn["input"][0]["text"])
            self.assertNotIn("private-image", turn["input"][0]["text"])

    async def test_invalid_base64_never_reaches_codex(self):
        for reference in (
            "data:image/png;base64,***",
            "data:image/png;base64,",
            "data:image/png,plain",
        ):
            turn = await self.capture(reference)
            self.assertEqual(len(turn["input"]), 1)
            self.assertIn("Image unavailable", turn["input"][0]["text"])
