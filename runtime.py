"""Resolve local PC launch settings without invoking a shell or reading credentials."""

import json
import os
import shutil
from pathlib import Path


class RuntimeSetupError(RuntimeError):
    """Represent a safe, user-facing local setup error."""


def resolve_runtime(default_home: Path, settings_path: Path | None = None):
    """Find a native CLI or npm entry point and the selected Codex home.

    Args:
        default_home: Fallback account directory, never read for credentials.
        settings_path: Optional AstrBot plugin configuration file.

    Returns:
        A command argument list, account directory and discovery source.

    Raises:
        RuntimeError: Settings or a required executable are unavailable.
    """
    settings = {}
    if settings_path and settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
            if not isinstance(settings, dict):
                raise ValueError("Expected object")
        except (OSError, ValueError) as exc:
            raise RuntimeSetupError(
                "无法读取插件配置，请在插件配置中重新保存。"
            ) from exc
    custom = settings.get("cli_path", "")
    account_home = settings.get("codex_home", "")
    if not isinstance(custom, str) or not isinstance(account_home, str):
        raise RuntimeSetupError("Codex 路径设置必须是文本，请在插件配置中修正。")
    home = Path(
        account_home.strip() or os.environ.get("CODEX_HOME") or default_home
    ).expanduser()
    if not home.is_absolute() or (home.exists() and not home.is_dir()):
        raise RuntimeSetupError("Codex 用户目录必须是文件夹的绝对路径。")
    source = "custom" if custom.strip() else "auto"
    if custom.strip():
        candidates = [Path(custom.strip()).expanduser()]
    else:
        desktop = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
            / "OpenAI/Codex/bin"
        )
        candidates = sorted(
            desktop.glob("*/codex.exe"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for name in ("codex.exe", "codex", "codex.cmd"):
            found = shutil.which(name)
            if found and Path(found) not in candidates:
                candidates.append(Path(found))
    for candidate in candidates:
        if not candidate.is_absolute() or not candidate.is_file():
            continue
        if candidate.suffix.lower() in {".cmd", ".bat", ".ps1"}:
            # npm wrappers are shell scripts. Launch their JavaScript entry directly.
            candidate = candidate.parent / "node_modules/@openai/codex/bin/codex.js"
        if candidate.suffix.lower() == ".js":
            node = shutil.which("node.exe") or shutil.which("node")
            if node and candidate.is_file():
                return [node, str(candidate)], home, source
        elif candidate.is_file() and (
            candidate.suffix.lower() == ".exe" or candidate.name == "codex"
        ):
            return [str(candidate)], home, source
    if custom.strip():
        raise RuntimeSetupError(
            "指定的 Codex CLI 无法运行。请填写 codex.exe 或官方 npm 的 codex.cmd/codex.js 完整路径。"
        )
    raise RuntimeSetupError(
        "未找到 Codex CLI。请先安装并登录，或在插件配置中填写 CLI 完整路径。"
    )
