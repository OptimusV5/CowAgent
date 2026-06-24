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


def _selected_openai_instance():
    instances = _saved_openai_instances()
    selected_id = (conf().get("voice_to_text_openai_instance_id") or "").strip()
    if selected_id:
        for item in instances:
            if (item.get("id") or "").strip() == selected_id:
                return item
    return instances[0] if instances else None


def _openai_asr_runtime_config():
    instance = _selected_openai_instance()
    if instance:
        api_key = (instance.get("api_key") or "").strip() or conf().get("voice_to_text_api_key") or conf().get("open_ai_api_key") or ""
        api_base = (
            _normalize_api_base(instance.get("api_base"))
            or _normalize_api_base(conf().get("voice_to_text_api_base"))
            or _normalize_api_base(conf().get("open_ai_api_base"))
            or "https://api.openai.com/v1"
        )
        model = (instance.get("model") or "").strip() or conf().get("voice_to_text_model") or DEFAULT_ASR_MODEL
        proxy_value = (instance.get("proxy") or "").strip()
        proxies = proxy_dict(proxy_value) if proxy_value else None
        data = {"model": model}
        response_format = (instance.get("response_format") or "").strip()
        hotwords = (instance.get("hotwords") or "").strip()
        replace_json = (instance.get("replace_json") or "").strip()
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
    return {
        "api_key": api_key,
        "api_base": api_base,
        "data": {
            "model": conf().get("voice_to_text_model") or DEFAULT_ASR_MODEL,
        },
        "proxies": config_proxy_dict("voice_to_text_proxy"),
        "instance_id": "",
    }


class OpenaiVoice(Voice):
    def __init__(self):
        # No-op: this implementation calls OpenAI HTTP endpoints directly via
        # `requests`, so it does not need a global SDK to be configured.
        pass

    def voiceToText(self, voice_file):
        logger.debug("[Openai] voice file name={}".format(voice_file))
        try:
            runtime = _openai_asr_runtime_config()
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
