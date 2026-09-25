# -*- coding: utf-8 -*-
"""
LLM Rotator Module for KOINONIA Assistant
Provides transparent, thread-safe Key Rotation and Model Rotation for Groq API calls.
Prevents HTTP 429 RateLimitError across rapid multi-step RAG pipelines.
"""

import os
import time
import logging
import threading
from typing import List, Optional, Any, Dict
from langchain_groq import ChatGroq

logger = logging.getLogger("koinonia.llm_rotator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LLMRotator:
    """
    Intelligent Rotator for Groq LLMs.
    - Round-robin rotates through all available API keys.
    - Tracks rate-limited keys and places them on cooldown.
    - Automatically cascades to fallback models if primary model is saturated.
    """
    def __init__(self, temperature: float = 0.0, default_model: Optional[str] = None):
        self.temperature = temperature
        self._lock = threading.Lock()
        self._key_index = 0
        self._key_cooldowns: Dict[str, float] = {}

        self.models: List[str] = self._load_models(default_model)
        self.keys: List[str] = self._load_keys()
        logger.info(f"[LLMRotator] Initialized with {len(self.keys)} Groq keys and models: {self.models}")

    def _load_keys(self) -> List[str]:
        keys = []
        try:
            import frappe
            if hasattr(frappe, "conf") and frappe.conf:
                conf_keys = frappe.conf.get("groq_api_keys") or []
                if isinstance(conf_keys, list):
                    keys.extend(conf_keys)
                elif isinstance(conf_keys, str):
                    keys.extend([k.strip() for k in conf_keys.split(",") if k.strip()])
                single_key = frappe.conf.get("groq_api_key")
                if single_key and single_key not in keys:
                    keys.insert(0, single_key)
        except Exception:
            pass

        # Environment fallback
        env_keys = os.getenv("GROQ_API_KEYS", "")
        if env_keys:
            keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])
        single_env = os.getenv("GROQ_API_KEY", "")
        if single_env and single_env not in keys:
            keys.append(single_env)

        # Deduplicate preserving order
        unique_keys = []
        for k in keys:
            k = k.strip()
            if k and k not in unique_keys and not k.startswith("placeholder"):
                unique_keys.append(k)

        return unique_keys or ["placeholder_key"]

    def _load_models(self, default_model: Optional[str] = None) -> List[str]:
        default_cascade = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b"
        ]
        models = []
        if default_model:
            models.append(default_model.strip())

        try:
            import frappe
            if hasattr(frappe, "conf") and frappe.conf:
                conf_models = frappe.conf.get("groq_models")
                if isinstance(conf_models, list):
                    models.extend(conf_models)
                conf_single = frappe.conf.get("groq_model")
                if conf_single and conf_single not in models:
                    models.insert(0, conf_single.strip())
        except Exception:
            pass

        env_model = os.getenv("GROQ_MODEL", "")
        if env_model and env_model not in models:
            models.insert(0, env_model.strip())

        for m in default_cascade:
            if m not in models:
                models.append(m)

        cleaned = []
        for m in models:
            m = m.strip()
            if m in ["gpt-oss-20b", "openai-gpt-oss-20b"]:
                m = "openai/gpt-oss-20b"
            elif m in ["gpt-oss-120b", "openai-gpt-oss-120b"]:
                m = "openai/gpt-oss-120b"
            if m not in cleaned:
                cleaned.append(m)
        return cleaned

    def _get_next_key(self) -> str:
        with self._lock:
            now = time.time()
            total_keys = len(self.keys)

            for _ in range(total_keys):
                candidate = self.keys[self._key_index % total_keys]
                self._key_index = (self._key_index + 1) % total_keys

                cooldown_until = self._key_cooldowns.get(candidate, 0.0)
                if now >= cooldown_until:
                    return candidate

            # If all keys on cooldown, pick the one with earliest expiry
            earliest_key = min(self.keys, key=lambda k: self._key_cooldowns.get(k, 0.0))
            wait_time = max(0.0, self._key_cooldowns.get(earliest_key, 0.0) - now)
            if 0 < wait_time < 4.0:
                logger.info(f"[LLMRotator] All keys on cooldown. Waiting {wait_time:.1f}s for key...")
                time.sleep(wait_time)
            return earliest_key

    def _mark_key_cooldown(self, key: str, duration: float = 60.0):
        with self._lock:
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else key
            self._key_cooldowns[key] = time.time() + duration
            logger.warning(f"[LLMRotator] Key {masked} rate limited. Cooldown set for {duration}s.")

    def _clear_key_cooldown(self, key: str):
        with self._lock:
            self._key_cooldowns.pop(key, None)

    def invoke(self, input_data: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        if len(self.keys) <= 1:
            refreshed = self._load_keys()
            if len(refreshed) > len(self.keys):
                self.keys = refreshed

        max_attempts = max(len(self.keys) * len(self.models), 10)
        last_error = None

        for attempt in range(max_attempts):
            key = self._get_next_key()
            model_idx = (attempt // len(self.keys)) % len(self.models)
            active_model = self.models[model_idx]
            masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else key

            try:
                chat = ChatGroq(
                    model=active_model,
                    temperature=self.temperature,
                    groq_api_key=key,
                    max_retries=1,
                    timeout=kwargs.get("timeout", 45)
                )

                if config:
                    response = chat.invoke(input_data, config=config, **kwargs)
                else:
                    response = chat.invoke(input_data, **kwargs)

                self._clear_key_cooldown(key)
                return response

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "rate_limit" in err_str or
                    "429" in err_str or
                    "tpm" in err_str or
                    "rpm" in err_str or
                    "too many requests" in err_str or
                    "quota" in err_str
                )

                if is_rate_limit:
                    logger.warning(
                        f"[LLMRotator] Attempt {attempt+1}/{max_attempts}: Key {masked_key} hit rate limit on model '{active_model}'. "
                        f"Rotating to next key..."
                    )
                    self._mark_key_cooldown(key, duration=60.0, error_str=err_str)
                    last_error = e
                    time.sleep(0.3)  # brief pause before next attempt
                    continue
                else:
                    logger.error(f"[LLMRotator] Error invoking {active_model} with key {masked_key}: {e}")
                    last_error = e
                    if any(c in err_str for c in ["500", "502", "503", "504", "timeout", "connection"]):
                        time.sleep(1.0)
                        continue
                    raise e

        logger.critical(f"[LLMRotator] All {max_attempts} rotation attempts exhausted across all keys and models!")
        raise last_error or RuntimeError("LLM Rotator exhausted all attempts.")

    def bind(self, **kwargs):
        return BoundLLMRotator(self, **kwargs)

    def with_config(self, **kwargs):
        return self


class BoundLLMRotator:
    def __init__(self, rotator: LLMRotator, **bind_kwargs):
        self.rotator = rotator
        self.bind_kwargs = bind_kwargs

    def invoke(self, input_data: Any, config: Optional[Dict] = None, **kwargs):
        merged_kwargs = {**self.bind_kwargs, **kwargs}
        return self.rotator.invoke(input_data, config=config, **merged_kwargs)


_llm_rotator_instance = None

def get_llm_rotator(temperature: float = 0.0, default_model: Optional[str] = None) -> LLMRotator:
    global _llm_rotator_instance
    if _llm_rotator_instance is None:
        _llm_rotator_instance = LLMRotator(temperature=temperature, default_model=default_model)
    return _llm_rotator_instance