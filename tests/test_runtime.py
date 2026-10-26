"""Verify PC launch paths, authentication reports and reload boundaries."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from test_request_lifecycle import CLIENT, MAIN

spec = importlib.util.spec_from_file_location(
    "codex_runtime_under_test", Path(__file__).parents[1] / "runtime.py"
)
RUNTIME = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUNTIME)


def test_desktop_native_cli_precedes_npm_shim(tmp_path):
    native = tmp_path / "OpenAI/Codex/bin/1.2.3/codex.exe"
    native.parent.mkdir(parents=True)
    native.touch()
    with (
        patch.dict("os.environ", {"LOCALAPPDATA": str(tmp_path)}),
        patch.object(RUNTIME.shutil, "which", return_value="missing.cmd"),
    ):
        command, _, source = RUNTIME.resolve_runtime(tmp_path)
    assert command == [str(native)]
    assert source == "auto"


def test_custom_npm_shim_uses_node_without_shell(tmp_path):
    shim = tmp_path / "folder with spaces/codex.cmd"
    shim.parent.mkdir()
    shim.touch()
    javascript = shim.parent / "node_modules/@openai/codex/bin/codex.js"
    javascript.parent.mkdir(parents=True)
    javascript.touch()
    settings = tmp_path / "plugin.json"
    settings.write_text(
        json.dumps({"cli_path": str(shim), "codex_home": str(tmp_path)})
    )
    with patch.object(RUNTIME.shutil, "which", return_value="C:/node/node.exe"):
        command, home, _ = RUNTIME.resolve_runtime(tmp_path, settings)
    assert command == ["C:/node/node.exe", str(javascript)]
    assert home == tmp_path


def test_invalid_explicit_cli_does_not_silently_use_another_account(tmp_path):
    settings = tmp_path / "plugin.json"
    settings.write_text(json.dumps({"cli_path": str(tmp_path / "missing.exe")}))
    with pytest.raises(RuntimeError, match="指定"):
        RUNTIME.resolve_runtime(tmp_path, settings)


def test_relative_account_directory_is_rejected(tmp_path):
    settings = tmp_path / "plugin.json"
    settings.write_text('{"codex_home":"relative/path"}')
    with pytest.raises(RuntimeError, match="绝对路径"):
        RUNTIME.resolve_runtime(tmp_path, settings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "account,required,authenticated",
    [
        (None, True, False),
        (None, False, True),
        ({"type": "chatgpt", "email": "private@example.com"}, True, True),
    ],
)
async def test_diagnostics_distinguish_login_and_redact_identity(
    tmp_path, account, required, authenticated
):
    client = CLIENT.CodexClient(tmp_path, tmp_path)
    client.start = AsyncMock()
    client.request = AsyncMock(
        return_value={"account": account, "requiresOpenaiAuth": required}
    )
    client.models = AsyncMock(return_value=["test-model"])
    report = await client.diagnose()
    assert report["authenticated"] is authenticated
    assert report["model_count"] == 1
    assert "private@example.com" not in json.dumps(report)
    client.request.assert_awaited_once_with("account/read", {"refreshToken": False})


@pytest.mark.asyncio
async def test_protocol_failure_does_not_echo_private_rpc_message(tmp_path):
    client = CLIENT.CodexClient(tmp_path, tmp_path)
    client.start = AsyncMock(side_effect=RuntimeError("token-private-data"))
    report = await client.diagnose()
    assert not report["connected"]
    assert "token-private-data" not in json.dumps(report)


@pytest.mark.asyncio
async def test_plugin_unload_closes_codex_without_closing_provider_http(tmp_path):
    from types import SimpleNamespace

    provider = SimpleNamespace(
        codex=SimpleNamespace(close=AsyncMock()), terminate=AsyncMock()
    )
    instance = object.__new__(MAIN.Main)
    with patch.object(MAIN, "INSTANCES", [provider]):
        await instance.terminate()
    provider.codex.close.assert_awaited_once()
    provider.terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_settings_reload_rebuilds_only_local_codex_connections():
    from types import SimpleNamespace

    class Provider:
        pass

    local, other = Provider(), Provider()
    local.provider_config = {"type": "local_codex_openai"}
    local.codex = SimpleNamespace(close=AsyncMock())
    previous = local.codex
    other.provider_config = {"type": "another_provider"}
    other.codex = SimpleNamespace(close=AsyncMock())
    instance = object.__new__(MAIN.Main)
    instance.context = SimpleNamespace(get_all_providers=lambda: [local, other])
    replacement = object()
    with (
        patch.object(MAIN, "CodexClient", return_value=replacement),
        patch.object(MAIN, "INSTANCES", set()),
    ):
        await instance.initialize()
    previous.close.assert_awaited_once()
    other.codex.close.assert_not_awaited()
    assert local.codex is replacement
