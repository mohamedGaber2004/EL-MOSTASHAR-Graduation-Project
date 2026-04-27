import logging
import time
import random
from typing import Any, Callable, List

from src.Config import get_settings
from src.Utils.agents_enums import AgentsEnums


class AgentBase:
    def __init__(self, model_config_key: str, temp_config_key: str, prompt: str) -> None:
        self.cfg = get_settings()
        self.model_name: str = getattr(self.cfg, model_config_key)
        self.temperature: float = getattr(self.cfg, temp_config_key)
        self.prompt: str = prompt

    def _llm_invoke_with_retries(self,llm_callable: Callable,messages: List[Any],max_retries: int = None,backoff_base: float = 2.0,) -> Any:
        if max_retries is None:
            max_retries = self.cfg.INVOKATION_MAX_RETRIES
        for attempt in range(1, max_retries + 1):
            try:
                return llm_callable.invoke(messages)
            except Exception as e:
                err = str(e).lower()
                is_rate = any(k in err for k in AgentsEnums.AGENT_INVOKATION_ERRORS.value)
                if is_rate and attempt < max_retries:
                    wait = min(60.0, backoff_base * (2 ** (attempt - 1))) + random.random()
                    time.sleep(wait)
                    continue
                raise

    def _empty_update(self, state: Any, agent_key: str, output: Any, inter_agent_delay: float = get_settings().INTER_AGENTS_DELAY) -> Any:
        time.sleep(inter_agent_delay) 
        return state.model_copy(update={
            "agent_outputs":    {**state.agent_outputs, agent_key: output},
            "completed_agents": state.completed_agents + [agent_key],
            "current_agent":    agent_key
        })