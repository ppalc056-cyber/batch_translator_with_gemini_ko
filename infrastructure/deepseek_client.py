import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

try:
    from infrastructure.logger_config import setup_logger
    from infrastructure.OpenAICompatibleClient import OpenAICompatibleClient
    from infrastructure.gemini_client import (
        GeminiApiException,
        GeminiInvalidRequestException,
        GeminiContentSafetyException,
        GeminiAllApiKeysExhaustedException,
    )
except ImportError:
    from .logger_config import setup_logger
    from .OpenAICompatibleClient import OpenAICompatibleClient
    from .gemini_client import (
        GeminiApiException,
        GeminiInvalidRequestException,
        GeminiContentSafetyException,
        GeminiAllApiKeysExhaustedException,
    )

logger = setup_logger(__name__)


class DeepSeekClient:
    """GeminiClient 호환 시그니처를 제공하는 DeepSeek 전용 클라이언트."""

    def __init__(
        self,
        auth_credentials: Optional[Union[str, List[str]]] = None,
        base_url: str = "https://api.deepseek.com",
        available_models: Optional[List[str]] = None,
        requests_per_minute: Optional[float] = None,
        api_timeout: float = 500.0,
        **_: Any,
    ) -> None:
        api_key = ""
        if isinstance(auth_credentials, list):
            api_key = next((k.strip() for k in auth_credentials if isinstance(k, str) and k.strip()), "")
        elif isinstance(auth_credentials, str):
            api_key = auth_credentials.strip()

        if not api_key:
            raise GeminiInvalidRequestException("DeepSeek API 키가 제공되지 않았습니다.")

        timeout_sec = max(1, int(api_timeout))
        self.base_url = self._normalize_base_url(base_url)
        self.available_models = available_models or ["deepseek-v4-flash", "deepseek-v4-pro"]
        self.client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=self.base_url,
            requests_per_minute=requests_per_minute,
            request_timeout=timeout_sec,
        )

    @staticmethod
    def _normalize_base_url(raw_url: Optional[str]) -> str:
        url = (raw_url or "").strip().rstrip("/")
        if not url:
            return "https://api.deepseek.com/chat/completions"

        if url.endswith("/chat/completions") or url.endswith("/v1/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/chat/completions"

    @staticmethod
    def _normalize_reasoning_effort(raw: Optional[str]) -> str:
        level = (raw or "high").strip().lower()
        if level in ("low", "medium"):
            return "high"
        if level == "xhigh":
            return "max"
        if level not in ("high", "max"):
            return "high"
        return level

    @staticmethod
    def _extract_content_text(parts: Any) -> str:
        chunks: List[str] = []
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, str):
                    chunks.append(p)
                elif isinstance(p, dict) and isinstance(p.get("text"), str):
                    chunks.append(p["text"])
                elif hasattr(p, "text") and isinstance(getattr(p, "text"), str):
                    chunks.append(getattr(p, "text"))
        return "\n".join(c for c in chunks if c)

    def _convert_prompt_to_messages(self, prompt: Union[str, List[Any]]) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]

        messages: List[Dict[str, str]] = []
        if not isinstance(prompt, list):
            raise GeminiInvalidRequestException("프롬프트는 문자열 또는 리스트여야 합니다.")

        for item in prompt:
            if isinstance(item, dict) and "role" in item and "content" in item:
                role = str(item.get("role", "user")).lower()
                role = "assistant" if role == "model" else role
                messages.append({"role": role, "content": str(item.get("content", ""))})
                continue

            role = getattr(item, "role", "user")
            role = "assistant" if role == "model" else str(role)
            parts = getattr(item, "parts", None)
            text = self._extract_content_text(parts)
            if text.strip():
                messages.append({"role": role, "content": text})

        if not messages:
            raise GeminiInvalidRequestException("유효한 프롬프트 메시지가 없습니다.")
        return messages

    async def list_models_async(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": model,
                "short_name": model,
                "display_name": model,
                "description": "DeepSeek model",
            }
            for model in self.available_models
        ]

    async def generate_text_async(
        self,
        prompt: Union[str, List[Any]],
        model_name: str,
        generation_config_dict: Optional[Dict[str, Any]] = None,
        safety_settings_list_of_dicts: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = None,
        system_instruction_text: Optional[str] = None,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        stream: bool = False,
    ) -> Optional[Union[str, Any]]:
        del safety_settings_list_of_dicts, thinking_budget

        messages = self._convert_prompt_to_messages(prompt)
        cfg = dict(generation_config_dict or {})

        expect_json = cfg.get("response_mime_type") == "application/json"
        cfg.pop("response_schema", None)
        cfg.pop("response_mime_type", None)

        thinking_enabled = bool(cfg.pop("deepseek_thinking_enabled", True))
        effort_input = cfg.pop("deepseek_reasoning_effort", None) or cfg.pop("thinking_level", None)
        effort = self._normalize_reasoning_effort(effort_input)

        cfg["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
        if thinking_enabled:
            cfg["reasoning_effort"] = effort

        try:
            response = await asyncio.to_thread(
                self.client.generate_text,
                prompt=messages,
                model_name=model_name,
                generation_config=cfg,
                system_instruction_text=system_instruction_text,
                stream=stream,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
            )
        except Exception as e:
            logger.error(f"DeepSeek API 호출 실패: {e}")
            raise GeminiApiException(f"DeepSeek API 호출 실패: {e}") from e

        if stream:
            return response

        if response is None:
            raise GeminiContentSafetyException("DeepSeek API로부터 응답을 받지 못했습니다.")

        if expect_json and isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                raise GeminiApiException(f"JSON 파싱 실패: {e}") from e

        return response
