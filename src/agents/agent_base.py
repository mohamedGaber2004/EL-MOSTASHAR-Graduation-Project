import logging
import time
import random
from typing import Any, Callable, List
from src.Config import get_settings
from src.Graph.graph_helpers import _now


class AgentBase:
    def __init__(self, model_config_key: str, temp_config_key: str, prompt: str) -> None:
        self.cfg = get_settings()
        self.model_name: str = getattr(self.cfg, model_config_key)
        self.temperature: float = getattr(self.cfg, temp_config_key)
        self.prompt: str = prompt
        self.logger = logging.getLogger(self.__class__.__name__)

    def _llm_invoke_with_retries(
        self,
        llm_callable: Callable,
        messages: List[Any],
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> Any:
        for attempt in range(1, max_retries + 1):
            try:
                return llm_callable.invoke(messages)
            except Exception as e:
                err = str(e).lower()
                is_rate = any(k in err for k in ("429", "rate_limit", "rate_limit_exceeded", "tokens per minute", "please try again"))
                if is_rate and attempt < max_retries:
                    wait = min(60.0, backoff_base * (2 ** (attempt - 1))) + random.random()
                    self.log(f"Rate-limited ({attempt}/{max_retries}) — إعادة المحاولة بعد {wait:.1f}s", "warning")
                    time.sleep(wait)
                    continue
                raise

    def _empty_update(self, state: Any, agent_key: str, output: Any) -> Any:
        return state.model_copy(update={
            "agent_outputs":    {**state.agent_outputs, agent_key: output},
            "completed_agents": state.completed_agents + [agent_key],
            "current_agent":    agent_key,
            "last_updated":     _now(),
        })

    def log(self, msg: str, level: str = "info") -> None:
        getattr(self.logger, level)(msg)