import logging
from src.Config import get_settings
import time, random

class AgentBase:
    """
    Base class for all agent nodes. Handles LLM invocation, retry/backoff, config, and logging.
    """
    def __init__(self, model_config_key, temp_config_key, prompt):
        self.cfg = get_settings()
        self.model_name = getattr(self.cfg, model_config_key)
        self.temperature = getattr(self.cfg, temp_config_key)
        self.prompt = prompt
        self.logger = logging.getLogger(self.__class__.__name__)

    def _llm_invoke_with_retries(self, llm_callable, messages, max_retries=5, backoff_base=2.0):
        for attempt in range(1, max_retries + 1):
            try:
                return llm_callable.invoke(messages)
            except Exception as e:
                err_text = str(e)
                is_rate = (
                    "429" in err_text
                    or "rate_limit" in err_text.lower()
                    or "rate_limit_exceeded" in err_text.lower()
                    or "tokens per minute" in err_text.lower()
                )
                if is_rate and attempt < max_retries:
                    wait = min(60.0, backoff_base * (2 ** (attempt - 1))) + random.random()
                    self.logger.warning(f"Rate-limited on attempt {attempt}/{max_retries} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise

    def log(self, msg, level="info"):
        getattr(self.logger, level)(msg)
