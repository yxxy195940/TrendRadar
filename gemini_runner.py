# coding=utf-8

import json
import os
import smtplib
import subprocess
from dataclasses import dataclass
from datetime import datetime
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytz
import requests
import yaml


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class GeminiConfig:
    api_key: str
    model: str
    system_prompt: str
    temperature: float
    max_tokens: int


@dataclass
class InputConfig:
    mode: str
    file_path: str
    directory_path: str
    command: str


@dataclass
class OutputConfig:
    save_to_output_dir: bool
    output_prefix: str
    format: str


@dataclass
class PushConfig:
    enabled: bool
    channels: List[str]
    title_template: str


@dataclass
class AppConfig:
    schedule: str
    timezone: str


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _expand_env(data)


def _get_timezone(name: str):
    return pytz.timezone(name)


def _get_now(tz_name: str) -> datetime:
    return datetime.now(_get_timezone(tz_name))


def _parse_multi_accounts(value: str) -> List[str]:
    if not value:
        return []
    items = [item.strip() for item in value.split(";")]
    if all(not item for item in items):
        return []
    return items


def _validate_paired_configs(
    configs: Dict[str, List[str]],
    channel_name: str,
    required_keys: Optional[List[str]] = None,
) -> Tuple[bool, int]:
    non_empty_configs = {k: v for k, v in configs.items() if v}

    if not non_empty_configs:
        return True, 0

    if required_keys:
        for key in required_keys:
            if key not in non_empty_configs or not non_empty_configs[key]:
                return True, 0

    lengths = {k: len(v) for k, v in non_empty_configs.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) > 1:
        print(f"❌ {channel_name} 配置错误：配对配置数量不一致，将跳过该渠道推送")
        for key, length in lengths.items():
            print(f"   - {key}: {length} 个")
        return False, 0

    return True, list(unique_lengths)[0] if unique_lengths else 0


def _limit_accounts(accounts: List[str], max_count: int, channel_name: str) -> List[str]:
    if len(accounts) > max_count:
        print(
            f"⚠️ {channel_name} 配置了 {len(accounts)} 个账号，超过最大限制 {max_count}，只使用前 {max_count} 个"
        )
        print("   ⚠️ 警告：如果您是 fork 用户，过多账号可能导致 GitHub Actions 运行时间过长，存在账号风险")
        return accounts[:max_count]
    return accounts


def _get_account_at_index(accounts: List[str], index: int, default: str = "") -> str:
    if index < len(accounts):
        return accounts[index] if accounts[index] else default
    return default


def _read_latest_file(directory: Path) -> str:
    if not directory.exists():
        raise FileNotFoundError(f"输入目录不存在: {directory}")
    files = [item for item in directory.iterdir() if item.is_file()]
    if not files:
        raise FileNotFoundError(f"输入目录没有文件: {directory}")
    latest = max(files, key=lambda item: item.stat().st_mtime)
    return latest.read_text(encoding="utf-8")


def _load_input(config: InputConfig) -> str:
    mode = config.mode.lower()
    if mode == "file":
        return Path(config.file_path).read_text(encoding="utf-8")
    if mode == "directory":
        return _read_latest_file(Path(config.directory_path))
    if mode == "command":
        if not config.command:
            raise ValueError("command 模式需要配置 input.command")
        result = subprocess.run(
            config.command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    raise ValueError(f"不支持的 input.mode: {config.mode}")


def _build_gemini_payload(prompt: str, config: GeminiConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        },
    }
    if config.system_prompt:
        payload["system_instruction"] = {
            "parts": [{"text": config.system_prompt}],
        }
    return payload


def _call_gemini(prompt: str, config: GeminiConfig) -> Dict[str, Any]:
    if not config.api_key:
        raise ValueError("未配置 GEMINI_API_KEY")
    url = f"{GEMINI_API_URL}/{config.model}:generateContent"
    response = requests.post(
        url,
        params={"key": config.api_key},
        json=_build_gemini_payload(prompt, config),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _extract_response_text(response: Dict[str, Any]) -> str:
    candidates = response.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini 响应中未包含 candidates")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise ValueError("Gemini 响应中未包含内容")
    text = parts[0].get("text", "").strip()
    if not text:
        raise ValueError("Gemini 响应文本为空")
    return text


def _save_output(config: OutputConfig, content: str, raw: Dict[str, Any], tz_name: str) -> Optional[Path]:
    if not config.save_to_output_dir:
        return None
    now = _get_now(tz_name)
    output_dir = Path("output") / "gemini"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = config.format.lower()
    if suffix == "markdown":
        extension = "md"
    elif suffix == "json":
        extension = "json"
    else:
        extension = "txt"
    filename = f"{config.output_prefix}_{timestamp}.{extension}"
    path = output_dir / filename
    if suffix == "json":
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _render_title(template: str, tz_name: str) -> str:
    now = _get_now(tz_name)
    return template.format(date=now.strftime("%Y-%m-%d"), datetime=now.strftime("%Y-%m-%d %H:%M:%S"))


def _send_feishu(webhook_url: str, text: str) -> None:
    payload = {"msg_type": "text", "content": {"text": text}}
    requests.post(webhook_url, json=payload, timeout=30).raise_for_status()


def _send_dingtalk(webhook_url: str, title: str, text: str) -> None:
    payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
    requests.post(webhook_url, json=payload, timeout=30).raise_for_status()


def _send_wework(webhook_url: str, text: str, msg_type: str) -> None:
    if msg_type == "text":
        payload = {"msgtype": "text", "text": {"content": text}}
    else:
        payload = {"msgtype": "markdown", "markdown": {"content": text}}
    requests.post(webhook_url, json=payload, timeout=30).raise_for_status()


def _send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload, timeout=30).raise_for_status()


def _send_email(
    email_from: str,
    email_password: str,
    email_to: str,
    title: str,
    text: str,
    smtp_server: str,
    smtp_port: str,
) -> None:
    if not smtp_server:
        domain = email_from.split("@")[-1]
        smtp_server = f"smtp.{domain}"
    port = int(smtp_port) if smtp_port else 587
    message = MIMEMultipart()
    message["From"] = formataddr((str(Header("TrendRadar Gemini", "utf-8")), email_from))
    message["To"] = email_to
    message["Subject"] = Header(title, "utf-8")
    message.attach(MIMEText(text, "plain", "utf-8"))

    with smtplib.SMTP(smtp_server, port, timeout=30) as server:
        server.starttls()
        server.login(email_from, email_password)
        server.sendmail(email_from, email_to.split(","), message.as_string())


def _send_ntfy(server_url: str, topic: str, token: str, title: str, text: str) -> None:
    url = f"{server_url.rstrip('/')}/{topic}"
    headers = {"Title": title}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    requests.post(url, data=text.encode("utf-8"), headers=headers, timeout=30).raise_for_status()


def _send_bark(bark_url: str, title: str, text: str) -> None:
    from urllib.parse import urlparse

    parsed_url = urlparse(bark_url)
    device_key = parsed_url.path.strip("/").split("/")[0] if parsed_url.path else ""
    if not device_key:
        raise ValueError(f"Bark URL 无法解析 device_key: {bark_url}")
    api_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}/push"
    payload = {
        "title": title,
        "markdown": text,
        "device_key": device_key,
        "sound": "default",
        "group": "TrendRadar-Gemini",
    }
    requests.post(api_endpoint, json=payload, timeout=30).raise_for_status()


def _send_slack(webhook_url: str, text: str) -> None:
    payload = {"text": text}
    requests.post(webhook_url, json=payload, timeout=30).raise_for_status()


def _load_notification_config(config_path: Path) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    notification = config.get("notification", {})
    webhooks = notification.get("webhooks", {})
    return {
        "feishu_url": os.environ.get("FEISHU_WEBHOOK_URL", webhooks.get("feishu_url", "")),
        "dingtalk_url": os.environ.get("DINGTALK_WEBHOOK_URL", webhooks.get("dingtalk_url", "")),
        "wework_url": os.environ.get("WEWORK_WEBHOOK_URL", webhooks.get("wework_url", "")),
        "wework_msg_type": os.environ.get("WEWORK_MSG_TYPE", webhooks.get("wework_msg_type", "markdown")),
        "telegram_bot_token": os.environ.get("TELEGRAM_BOT_TOKEN", webhooks.get("telegram_bot_token", "")),
        "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", webhooks.get("telegram_chat_id", "")),
        "email_from": os.environ.get("EMAIL_FROM", webhooks.get("email_from", "")),
        "email_password": os.environ.get("EMAIL_PASSWORD", webhooks.get("email_password", "")),
        "email_to": os.environ.get("EMAIL_TO", webhooks.get("email_to", "")),
        "email_smtp_server": os.environ.get("EMAIL_SMTP_SERVER", webhooks.get("email_smtp_server", "")),
        "email_smtp_port": os.environ.get("EMAIL_SMTP_PORT", webhooks.get("email_smtp_port", "")),
        "ntfy_server_url": os.environ.get("NTFY_SERVER_URL", webhooks.get("ntfy_server_url", "")),
        "ntfy_topic": os.environ.get("NTFY_TOPIC", webhooks.get("ntfy_topic", "")),
        "ntfy_token": os.environ.get("NTFY_TOKEN", webhooks.get("ntfy_token", "")),
        "bark_url": os.environ.get("BARK_URL", webhooks.get("bark_url", "")),
        "slack_webhook_url": os.environ.get("SLACK_WEBHOOK_URL", webhooks.get("slack_webhook_url", "")),
        "max_accounts_per_channel": int(notification.get("max_accounts_per_channel", 3)),
    }


def _dispatch_notifications(
    config_path: Path,
    push_config: PushConfig,
    title: str,
    body: str,
) -> None:
    notification = _load_notification_config(config_path)
    configured_channels = []
    max_accounts = notification["max_accounts_per_channel"]

    if notification["feishu_url"]:
        configured_channels.append("feishu")
    if notification["dingtalk_url"]:
        configured_channels.append("dingtalk")
    if notification["wework_url"]:
        configured_channels.append("wework")
    if notification["telegram_bot_token"] and notification["telegram_chat_id"]:
        configured_channels.append("telegram")
    if notification["email_from"] and notification["email_password"] and notification["email_to"]:
        configured_channels.append("email")
    if notification["ntfy_server_url"] and notification["ntfy_topic"]:
        configured_channels.append("ntfy")
    if notification["bark_url"]:
        configured_channels.append("bark")
    if notification["slack_webhook_url"]:
        configured_channels.append("slack")

    channels = push_config.channels or configured_channels
    text = f"{title}\n\n{body}"
    print(f"ℹ️ Gemini 推送渠道: {', '.join(channels) if channels else '无'}")
    wework_urls = _parse_multi_accounts(notification["wework_url"])
    wework_count = len([url for url in wework_urls if url])
    if wework_urls or notification["wework_url"]:
        print(f"ℹ️ 企业微信配置检测: 账号数={wework_count}, msg_type={notification['wework_msg_type']}")
    else:
        print("ℹ️ 企业微信配置检测: 未读取到 wework_url")

    for channel in channels:
        if channel == "feishu":
            urls = _parse_multi_accounts(notification["feishu_url"])
            if urls:
                urls = _limit_accounts(urls, max_accounts, "飞书")
                for url in urls:
                    if url:
                        _send_feishu(url, text)
        elif channel == "dingtalk":
            urls = _parse_multi_accounts(notification["dingtalk_url"])
            if urls:
                urls = _limit_accounts(urls, max_accounts, "钉钉")
                for url in urls:
                    if url:
                        _send_dingtalk(url, title, text)
        elif channel == "wework":
            msg_type = notification["wework_msg_type"].lower()
            urls = _parse_multi_accounts(notification["wework_url"])
            if urls:
                urls = _limit_accounts(urls, max_accounts, "企业微信")
                for url in urls:
                    if url:
                        print(f"➡️ 正在推送企业微信: msg_type={msg_type}")
                        _send_wework(url, text, msg_type)
                    else:
                        print("⚠️ 企业微信配置存在空值账号，已跳过")
            else:
                print("ℹ️ 企业微信未配置推送地址")
        elif channel == "telegram":
            tokens = _parse_multi_accounts(notification["telegram_bot_token"])
            chats = _parse_multi_accounts(notification["telegram_chat_id"])
            if tokens and chats:
                valid, count = _validate_paired_configs(
                    {"bot_token": tokens, "chat_id": chats},
                    "Telegram",
                    required_keys=["bot_token", "chat_id"],
                )
                if valid and count > 0:
                    tokens = _limit_accounts(tokens, max_accounts, "Telegram")
                    chats = chats[:len(tokens)]
                    for i in range(len(tokens)):
                        token = tokens[i]
                        chat_id = chats[i]
                        if token and chat_id:
                            _send_telegram(token, chat_id, text)
        elif channel == "email":
            _send_email(
                notification["email_from"],
                notification["email_password"],
                notification["email_to"],
                title,
                text,
                notification["email_smtp_server"],
                notification["email_smtp_port"],
            )
        elif channel == "ntfy":
            server_url = notification["ntfy_server_url"]
            topics = _parse_multi_accounts(notification["ntfy_topic"])
            tokens = _parse_multi_accounts(notification["ntfy_token"])
            if server_url and topics:
                if tokens and len(tokens) != len(topics):
                    print(
                        f"❌ ntfy 配置错误：topic 数量({len(topics)})与 token 数量({len(tokens)})不一致，跳过 ntfy 推送"
                    )
                else:
                    topics = _limit_accounts(topics, max_accounts, "ntfy")
                    if tokens:
                        tokens = tokens[:len(topics)]
                    for index, topic in enumerate(topics):
                        if topic:
                            token = _get_account_at_index(tokens, index, "") if tokens else ""
                            _send_ntfy(server_url, topic, token, title, text)
        elif channel == "bark":
            urls = _parse_multi_accounts(notification["bark_url"])
            if urls:
                urls = _limit_accounts(urls, max_accounts, "Bark")
                for url in urls:
                    if url:
                        _send_bark(url, title, text)
        elif channel == "slack":
            urls = _parse_multi_accounts(notification["slack_webhook_url"])
            if urls:
                urls = _limit_accounts(urls, max_accounts, "Slack")
                for url in urls:
                    if url:
                        _send_slack(url, text)
        else:
            raise ValueError(f"不支持的推送渠道: {channel}")


def _parse_configs(config_path: Path) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    app = config.get("app", {})
    gemini = config.get("gemini", {})
    input_cfg = config.get("input", {})
    output = config.get("output", {})
    push = config.get("push", {})

    app_config = AppConfig(
        schedule=str(app.get("schedule", "")),
        timezone=str(app.get("timezone", "Asia/Shanghai")),
    )
    gemini_config = GeminiConfig(
        api_key=os.environ.get("GEMINI_API_KEY", gemini.get("api_key", "")),
        model=os.environ.get("GEMINI_MODEL", gemini.get("model", "gemini-1.5-pro")),
        system_prompt=os.environ.get("GEMINI_SYSTEM_PROMPT", gemini.get("system_prompt", "")),
        temperature=float(os.environ.get("GEMINI_TEMPERATURE", gemini.get("temperature", 0.2))),
        max_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", gemini.get("max_tokens", 2048))),
    )
    input_config = InputConfig(
        mode=input_cfg.get("mode", "file"),
        file_path=input_cfg.get("file_path", "config/input.txt"),
        directory_path=input_cfg.get("directory_path", "input"),
        command=input_cfg.get("command", ""),
    )
    output_config = OutputConfig(
        save_to_output_dir=bool(output.get("save_to_output_dir", True)),
        output_prefix=str(output.get("output_prefix", "gemini_daily")),
        format=str(output.get("format", "markdown")),
    )
    push_config = PushConfig(
        enabled=bool(push.get("enabled", True)),
        channels=push.get("channels", []),
        title_template=str(push.get("title_template", "[Gemini 日报] {date}")),
    )
    return {
        "app": app_config,
        "gemini": gemini_config,
        "input": input_config,
        "output": output_config,
        "push": push_config,
    }


def main() -> None:
    config_path = Path(os.environ.get("GEMINI_CONFIG_PATH", "config/gemini_runner.yaml"))
    base_config_path = Path(os.environ.get("CONFIG_PATH", "config/config.yaml"))
    configs = _parse_configs(config_path)
    app_config: AppConfig = configs["app"]
    gemini_config: GeminiConfig = configs["gemini"]
    input_config: InputConfig = configs["input"]
    output_config: OutputConfig = configs["output"]
    push_config: PushConfig = configs["push"]

    prompt = _load_input(input_config)
    response = _call_gemini(prompt, gemini_config)
    content = _extract_response_text(response)
    output_path = _save_output(output_config, content, response, app_config.timezone)
    title = _render_title(push_config.title_template, app_config.timezone)

    if output_path:
        print(f"✅ Gemini 输出已保存: {output_path}")
    else:
        print("ℹ️ Gemini 输出未保存（save_to_output_dir=false）")

    if push_config.enabled:
        _dispatch_notifications(base_config_path, push_config, title, content)
        print("✅ Gemini 推送完成")
    else:
        print("ℹ️ Gemini 推送已禁用")


if __name__ == "__main__":
    main()
