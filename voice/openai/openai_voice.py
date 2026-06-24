"""
google voice service
"""
import json

from bridge.reply import Reply, ReplyType
from common.log import logger
from common.proxy import config_proxy_dict, proxy_dict
from config import conf
from voice.voice import Voice
import requests
from common import const
import datetime, random

DEFAULT_ASR_MODEL = "gpt-4o-mini-transcribe"


def _normalize_api_base(value):
    return (value or "").strip().rstrip("/")


def _saved_openai_instances():
    instances = conf().get("voice_to_text_openai_instances") or []
    if not isinstance(instances, list):
        return []
    return [item for item in instances if isinstance(item, dict)]


def _selected_openai_instance(instance_id=""):
    instances = _saved_openai_instances()
    explicit_id = (instance_id or "").strip()
    selected_id = explicit_id or (conf().get("voice_to_text_openai_instance_id") or "").strip()
    if selected_id:
        for item in instances:
            if (item.get("id") or "").strip() == selected_id:
                return item
        if explicit_id:
            raise ValueError(f"ASR OpenAI instance not found: {explicit_id}")
    return instances[0] if instances else None


def _stringify_replace_json(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("replace_json must be a JSON object")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _normalize_asr_options(options):
    if not isinstance(options, dict):
        return {}
    normalized = {}
    for key in ("instance_id", "model", "response_format", "hotwords"):
        value = options.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            normalized[key] = value
    if "replace_json" in options and options.get("replace_json") not in (None, ""):
        normalized["replace_json"] = _stringify_replace_json(options.get("replace_json"))
    return normalized


def _openai_asr_runtime_config(options=None):
    overrides = _normalize_asr_options(options)
    instance = _selected_openai_instance(overrides.get("instance_id", ""))
    if instance:
        api_key = (instance.get("api_key") or "").strip() or conf().get("voice_to_text_api_key") or conf().get("open_ai_api_key") or ""
        api_base = (
            _normalize_api_base(instance.get("api_base"))
            or _normalize_api_base(conf().get("voice_to_text_api_base"))
            or _normalize_api_base(conf().get("open_ai_api_base"))
            or "https://api.openai.com/v1"
        )
        model = overrides.get("model") or (instance.get("model") or "").strip() or conf().get("voice_to_text_model") or DEFAULT_ASR_MODEL
        proxy_value = (instance.get("proxy") or "").strip()
        proxies = proxy_dict(proxy_value) if proxy_value else None
        data = {"model": model}
        response_format = overrides.get("response_format") or (instance.get("response_format") or "").strip()
        hotwords = overrides.get("hotwords") or (instance.get("hotwords") or "").strip()
        replace_json = overrides.get("replace_json") or (instance.get("replace_json") or "").strip()
        if response_format:
            data["response_format"] = response_format
        if hotwords:
            data["hotwords"] = hotwords
        if replace_json:
            data["replace_json"] = replace_json
        return {
            "api_key": api_key,
            "api_base": api_base,
            "data": data,
            "proxies": proxies,
            "instance_id": (instance.get("id") or "").strip(),
        }

    api_key = conf().get("voice_to_text_api_key") or conf().get("open_ai_api_key") or ""
    api_base = (
        _normalize_api_base(conf().get("voice_to_text_api_base"))
        or _normalize_api_base(conf().get("open_ai_api_base"))
        or "https://api.openai.com/v1"
    )
    data = {
        "model": overrides.get("model") or conf().get("voice_to_text_model") or DEFAULT_ASR_MODEL,
    }
    for key in ("response_format", "hotwords", "replace_json"):
        if overrides.get(key):
            data[key] = overrides[key]
    return {
        "api_key": api_key,
        "api_base": api_base,
        "data": data,
        "proxies": config_proxy_dict("voice_to_text_proxy"),
        "instance_id": "",
    }


class OpenaiVoice(Voice):
    def __init__(self):
        # No-op: this implementation calls OpenAI HTTP endpoints directly via
        # `requests`, so it does not need a global SDK to be configured.
        pass

    def voiceToText(self, voice_file, options=None):
        logger.debug("[Openai] voice file name={}".format(voice_file))
        try:
            runtime = _openai_asr_runtime_config(options=options)
            api_key = runtime["api_key"]
            api_base = runtime["api_base"]
            url = f'{api_base}/audio/transcriptions'
            headers = {
                'Authorization': 'Bearer ' + api_key,
                # 'Content-Type': 'multipart/form-data' # 加了会报错，不知道什么原因
            }
            logger.info(
                "[Openai] voiceToText request: url=%s model=%s instance=%s",
                url,
                runtime["data"].get("model"),
                runtime.get("instance_id") or "legacy",
            )
            with open(voice_file, "rb") as file:
                files = {
                    "file": file,
                }
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=runtime["data"],
                    proxies=runtime["proxies"],
                )
            response_data = response.json()
            if response.status_code != 200 or "text" not in response_data:
                logger.error(
                    f"[Openai] voiceToText failed: status={response.status_code}, "
                    f"resp={response_data}"
                )
                reply = Reply(ReplyType.ERROR, "我暂时还无法听清您的语音，请稍后再试吧~")
            else:
                text = response_data["text"]
                reply = Reply(ReplyType.TEXT, text)
                logger.info("[Openai] voiceToText text={} voice file name={}".format(text, voice_file))
        except Exception as e:
            logger.error(f"[Openai] voiceToText exception: {e}", exc_info=True)
            reply = Reply(ReplyType.ERROR, "我暂时还无法听清您的语音，请稍后再试吧~")
        return reply


    def textToVoice(self, text):
        try:
            api_base = conf().get("open_ai_api_base") or "https://api.openai.com/v1"
            url = f'{api_base}/audio/speech'
            headers = {
                'Authorization': 'Bearer ' + conf().get("open_ai_api_key"),
                'Content-Type': 'application/json'
            }
            data = {
                'model': conf().get("text_to_voice_model") or const.TTS_1,
                'input': text,
                'voice': conf().get("tts_voice_id") or "alloy"
            }
            response = requests.post(url, headers=headers, json=data, proxies=config_proxy_dict("proxy"))
            file_name = "tmp/" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + str(random.randint(0, 1000)) + ".mp3"
            logger.debug(f"[OPENAI] text_to_Voice file_name={file_name}, input={text}")
            with open(file_name, 'wb') as f:
                f.write(response.content)
            logger.info(f"[OPENAI] text_to_Voice success")
            reply = Reply(ReplyType.VOICE, file_name)
        except Exception as e:
            logger.error(e)
            reply = Reply(ReplyType.ERROR, "遇到了一点小问题，请稍后再问我吧")
        return reply
