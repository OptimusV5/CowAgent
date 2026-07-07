from bridge.context import ContextType
from channel.chat_message import ChatMessage
import json
import os
import requests
from common.log import logger
from common.tmp_dir import TmpDir
from common import utils
from common.utils import expand_path
from config import conf


class FeishuFileTooLargeError(Exception):
    pass


class FeishuDownloadError(Exception):
    pass


def _feishu_error_message(response, fallback: str) -> str:
    try:
        data = response.json()
        code = data.get("code")
        msg = data.get("msg") or fallback
        if code == 234037:
            raise FeishuFileTooLargeError(
                "该聊天附件超过飞书消息资源下载接口 100MB 限制，无法通过 IM 附件接口直接下载。"
            )
        return f"{msg} (code={code})" if code else msg
    except FeishuFileTooLargeError:
        raise
    except Exception:
        return fallback


def _download_response_to_file(response, target_path: str) -> None:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


class FeishuMessage(ChatMessage):
    def __init__(self, event: dict, is_group=False, access_token=None):
        super().__init__(event)
        msg = event.get("message")
        sender = event.get("sender")
        self.access_token = access_token
        self.msg_id = msg.get("message_id")
        self.create_time = msg.get("create_time")
        self.is_group = is_group
        self.chat_id = msg.get("chat_id")
        self.root_id = msg.get("root_id")
        self.parent_id = msg.get("parent_id")
        self.upper_message_id = msg.get("upper_message_id")
        msg_type = msg.get("message_type")
        self.msg_type = msg_type
        self.raw_content = msg.get("content")

        if msg_type == "text":
            self.ctype = ContextType.TEXT
            content = json.loads(msg.get('content'))
            self.content = content.get("text").strip()
        elif msg_type == "image":
            # 单张图片消息：下载并缓存，等待用户提问时一起发送
            self.ctype = ContextType.IMAGE
            content = json.loads(msg.get("content"))
            image_key = content.get("image_key")
            
            # 下载图片到工作空间临时目录
            workspace_root = expand_path(conf().get("agent_workspace", "~/cow"))
            tmp_dir = os.path.join(workspace_root, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            image_path = os.path.join(tmp_dir, f"{image_key}.png")
            
            # 下载图片
            url = f"https://open.feishu.cn/open-apis/im/v1/messages/{msg.get('message_id')}/resources/{image_key}"
            headers = {"Authorization": "Bearer " + access_token}
            params = {"type": "image"}
            response = requests.get(url=url, headers=headers, params=params)
            
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"[FeiShu] Downloaded single image, key={image_key}, path={image_path}")
                self.content = image_path
                self.image_path = image_path  # 保存图片路径
            else:
                logger.error(f"[FeiShu] Failed to download single image, key={image_key}, status={response.status_code}")
                self.content = f"[图片下载失败: {image_key}]"
                self.image_path = None
        elif msg_type == "post":
            # 富文本消息，可能包含图片、文本等多种元素
            content = json.loads(msg.get("content"))
            
            # 飞书富文本消息结构：content 直接包含 title 和 content 数组
            # 不是嵌套在 post 字段下
            title = content.get("title", "")
            content_list = content.get("content", [])
            
            logger.info(f"[FeiShu] Post message - title: '{title}', content_list length: {len(content_list)}")
            
            # 收集所有图片和文本
            image_keys = []
            text_parts = []
            
            if title:
                text_parts.append(title)
            
            for block in content_list:
                logger.debug(f"[FeiShu] Processing block: {block}")
                # block 本身就是元素列表
                if not isinstance(block, list):
                    continue
                    
                for element in block:
                    element_tag = element.get("tag")
                    logger.debug(f"[FeiShu] Element tag: {element_tag}, element: {element}")
                    if element_tag == "img":
                        # 找到图片元素
                        image_key = element.get("image_key")
                        if image_key:
                            image_keys.append(image_key)
                    elif element_tag == "text":
                        # 文本元素
                        text_content = element.get("text", "")
                        if text_content:
                            text_parts.append(text_content)
                    elif element_tag == "a":
                        link_text = element.get("text", "")
                        href = element.get("href") or element.get("url") or ""
                        if link_text and href:
                            text_parts.append(f"{link_text} {href}")
                        elif href:
                            text_parts.append(href)
                        elif link_text:
                            text_parts.append(link_text)
            
            logger.info(f"[FeiShu] Parsed - images: {len(image_keys)}, text_parts: {text_parts}")
            
            # 富文本消息统一作为文本消息处理
            self.ctype = ContextType.TEXT
            
            if image_keys:
                # 如果包含图片，下载并在文本中引用本地路径
                workspace_root = expand_path(conf().get("agent_workspace", "~/cow"))
                tmp_dir = os.path.join(workspace_root, "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                
                # 保存图片路径映射
                self.image_paths = {}
                for image_key in image_keys:
                    image_path = os.path.join(tmp_dir, f"{image_key}.png")
                    self.image_paths[image_key] = image_path
                
                def _download_images():
                    for image_key, image_path in self.image_paths.items():
                        url = f"https://open.feishu.cn/open-apis/im/v1/messages/{self.msg_id}/resources/{image_key}"
                        headers = {"Authorization": "Bearer " + access_token}
                        params = {"type": "image"}
                        response = requests.get(url=url, headers=headers, params=params)
                        if response.status_code == 200:
                            with open(image_path, "wb") as f:
                                f.write(response.content)
                            logger.info(f"[FeiShu] Image downloaded from post message, key={image_key}, path={image_path}")
                        else:
                            logger.error(f"[FeiShu] Failed to download image from post, key={image_key}, status={response.status_code}")
                
                # 立即下载图片，不使用延迟下载
                # 因为 TEXT 类型消息不会调用 prepare()
                _download_images()
                
                # 构建消息内容：文本 + 图片路径
                content_parts = []
                if text_parts:
                    content_parts.append("\n".join(text_parts).strip())
                for image_key, image_path in self.image_paths.items():
                    content_parts.append(f"[图片: {image_path}]")
                
                self.content = "\n".join(content_parts)
                logger.info(f"[FeiShu] Received post message with {len(image_keys)} image(s) and text: {self.content}")
            else:
                # 纯文本富文本消息
                self.content = "\n".join(text_parts).strip() if text_parts else "[富文本消息]"
                logger.info(f"[FeiShu] Received post message (text only): {self.content}")
        elif msg_type == "file":
            self.ctype = ContextType.FILE
            content = json.loads(msg.get("content"))
            file_key = content.get("file_key")
            file_name = content.get("file_name")

            # 落到 agent_workspace/tmp 下（绝对路径），与图片处理一致；
            # 否则相对路径 ./tmp 在 agent 工作区里 read 时会找不到。
            workspace_root = expand_path(conf().get("agent_workspace", "~/cow"))
            tmp_dir = os.path.join(workspace_root, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            self.content = os.path.join(
                tmp_dir, f"{file_key}.{utils.get_path_suffix(file_name)}"
            )

            def _download_file():
                url = f"https://open.feishu.cn/open-apis/im/v1/messages/{self.msg_id}/resources/{file_key}"
                headers = {
                    "Authorization": "Bearer " + access_token,
                }
                params = {
                    "type": "file"
                }
                response = requests.get(url=url, headers=headers, params=params, stream=True, timeout=(5, 120))
                if response.status_code == 200:
                    _download_response_to_file(response, self.content)
                else:
                    error = _feishu_error_message(response, "文件下载失败")
                    logger.info(f"[FeiShu] Failed to download file, key={file_key}, res={response.text}")
                    raise FeishuDownloadError(error)
            self._prepare_fn = _download_file
        elif msg_type == "audio":
            # 飞书用户发送的语音消息类型为 "audio"，文件为 opus 编码格式。
            # 映射为 ContextType.VOICE，交由 chat_channel 的语音转文字（STT）流程处理。
            # 文件通过 _prepare_fn 延迟下载，在 chat_channel 调用 cmsg.prepare() 时才执行。
            self.ctype = ContextType.VOICE
            content = json.loads(msg.get("content"))
            file_key = content.get("file_key")

            # 落到 agent_workspace/tmp 下（绝对路径），保证语音 STT 流程可读到
            workspace_root = expand_path(conf().get("agent_workspace", "~/cow"))
            tmp_dir = os.path.join(workspace_root, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            self.content = os.path.join(tmp_dir, f"{file_key}.opus")
            logger.info(f"[FeiShu] audio message: file_key={file_key}, save_path={self.content}")

            def _download_audio():
                logger.info(f"[FeiShu] downloading audio: file_key={file_key}, msg_id={self.msg_id}")
                url = f"https://open.feishu.cn/open-apis/im/v1/messages/{self.msg_id}/resources/{file_key}"
                headers = {
                    "Authorization": "Bearer " + access_token,
                }
                params = {
                    "type": "file"
                }
                try:
                    response = requests.get(url=url, headers=headers, params=params, stream=True, timeout=(5, 120))
                    logger.info(
                        f"[FeiShu] download audio response: status={response.status_code}, "
                        f"size={response.headers.get('Content-Length', 'unknown')} bytes"
                    )
                    if response.status_code == 200:
                        _download_response_to_file(response, self.content)
                        logger.info(f"[FeiShu] audio saved to: {self.content}")
                    else:
                        _feishu_error_message(response, "语音下载失败")
                        logger.error(f"[FeiShu] Failed to download audio, key={file_key}, status={response.status_code}, res={response.text}")
                except Exception as e:
                    logger.error(f"[FeiShu] Exception downloading audio, key={file_key}: {e}", exc_info=True)
            self._prepare_fn = _download_audio
        elif msg_type == "media":
            # 飞书视频消息类型为 media。这里按文件消息下载并缓存，但标记为 video，
            # 后续文本消息会自动附加为 [视频: path]，交由 Agent/工具层解析。
            self.ctype = ContextType.FILE
            self.file_type = "video"
            content = json.loads(msg.get("content"))
            file_key = content.get("file_key")
            file_name = content.get("file_name") or f"{file_key}.mp4"
            suffix = utils.get_path_suffix(file_name) or "mp4"

            workspace_root = expand_path(conf().get("agent_workspace", "~/cow"))
            tmp_dir = os.path.join(workspace_root, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            self.content = os.path.join(tmp_dir, f"{file_key}.{suffix}")
            logger.info(f"[FeiShu] media message: file_key={file_key}, save_path={self.content}")

            def _download_media():
                logger.info(f"[FeiShu] downloading media: file_key={file_key}, msg_id={self.msg_id}")
                url = f"https://open.feishu.cn/open-apis/im/v1/messages/{self.msg_id}/resources/{file_key}"
                headers = {
                    "Authorization": "Bearer " + access_token,
                }

                last_response = None
                try:
                    for resource_type in ("file", "media"):
                        response = requests.get(
                            url=url,
                            headers=headers,
                            params={"type": resource_type},
                            stream=True,
                            timeout=(5, 120),
                        )
                        last_response = response
                        logger.info(
                            f"[FeiShu] download media response: type={resource_type}, "
                            f"status={response.status_code}, size={response.headers.get('Content-Length', 'unknown')} bytes"
                        )
                        if response.status_code == 200:
                            _download_response_to_file(response, self.content)
                            logger.info(f"[FeiShu] media saved to: {self.content}")
                            return
                        try:
                            _feishu_error_message(response, "视频下载失败")
                        except FeishuFileTooLargeError:
                            raise
                        except Exception:
                            pass

                    res_text = last_response.text if last_response is not None else ""
                    status_code = last_response.status_code if last_response is not None else "unknown"
                    logger.error(
                        f"[FeiShu] Failed to download media, key={file_key}, "
                        f"status={status_code}, res={res_text}"
                    )
                    raise FeishuDownloadError("视频下载失败")
                except FeishuFileTooLargeError:
                    raise
                except Exception as e:
                    logger.error(f"[FeiShu] Exception downloading media, key={file_key}: {e}", exc_info=True)
                    raise FeishuDownloadError(str(e))
            self._prepare_fn = _download_media
        elif msg_type == "interactive":
            self.ctype = ContextType.TEXT
            try:
                content = json.loads(msg.get("content"))
            except Exception:
                content = {}
            text_parts = []
            self._collect_interactive_text(content, text_parts)
            self.content = "\n".join(part for part in text_parts if part).strip() or "[卡片消息]"
        else:
            self.ctype = ContextType.TEXT
            try:
                content = json.loads(msg.get("content") or "{}")
            except Exception:
                content = {}
            text_parts = []
            self._collect_interactive_text(content, text_parts)
            fallback = "\n".join(part for part in text_parts if part).strip()
            self.content = fallback or f"[{msg_type} 消息]"

        self.from_user_id = sender.get("sender_id").get("open_id")
        self.to_user_id = event.get("app_id")
        if is_group:
            # 群聊
            self.other_user_id = msg.get("chat_id")
            self.actual_user_id = self.from_user_id
            self.content = self.content.replace("@_user_1", "").strip()
            self.actual_user_nickname = ""
        else:
            # 私聊
            self.other_user_id = self.from_user_id
            self.actual_user_id = self.from_user_id

    def _collect_interactive_text(self, node, out: list):
        if isinstance(node, dict):
            tag = node.get("tag")
            if tag == "a":
                href = node.get("href") or node.get("url") or node.get("link") or ""
                text = node.get("text") or node.get("content") or ""
                if text and href:
                    out.append(f"{text} {href}")
                elif href:
                    out.append(str(href))
                elif text:
                    out.append(str(text))
                return
            url = node.get("url") or node.get("href") or node.get("link")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                out.append(url)
            for key in ("title", "subtitle", "content", "text", "elements", "body", "header", "columns", "items", "fields", "actions"):
                if key in node:
                    self._collect_interactive_text(node.get(key), out)
        elif isinstance(node, list):
            for item in node:
                self._collect_interactive_text(item, out)
        elif isinstance(node, str):
            cleaned = node.strip()
            if cleaned:
                out.append(cleaned)
