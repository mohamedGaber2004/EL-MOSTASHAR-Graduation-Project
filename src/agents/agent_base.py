import logging , time , random
from typing import Any, Callable, List

from src.Config import get_settings
from src.Utils.agents_enums import AgentsEnums
from src.LLMs.GPT_OSS_Model import get_llm_oss
from src.LLMs.Qwen_Model import get_llm_qwn
from src.LLMs.Llama_Model import get_llm_llama

logger = logging.getLogger(__name__)

class AgentBase:
    def __init__(self, model_config_key: str, temp_config_key: str, prompt: str) -> None:
        self.cfg = get_settings()
        self.model_name: str = getattr(self.cfg, model_config_key)
        self.temperature: float = getattr(self.cfg, temp_config_key)
        self.prompt: str = prompt
        self.oss_llm = get_llm_oss(self.model_name,self.temperature)
        self.qwen_llm = get_llm_qwn(self.model_name,self.temperature)
        self.lama_llm = get_llm_llama(self.model_name,self.temperature)
        logger.debug("Initialized %s with model: %s, temp: %s", self.__class__.__name__, self.model_name, self.temperature)

    def _llm_invoke_with_retries(self, llm_callable: Callable, messages: List[Any], max_retries: int = None, backoff_base: float = 2.0,) -> Any:
        if max_retries is None:
            max_retries = self.cfg.INVOKATION_MAX_RETRIES
        logger.debug("Invoking LLM for %s (max retries: %d)", self.__class__.__name__, max_retries)
        for attempt in range(1, max_retries + 1):
            try:
                result = llm_callable.invoke(messages)
                if attempt > 1:
                    logger.info("LLM invocation succeeded on attempt %d", attempt)
                else:
                    logger.debug("LLM invocation succeeded on first attempt")
                return result
            except Exception as e:
                err = str(e).lower()
                is_rate = any(k in err for k in AgentsEnums.AGENT_INVOKATION_ERRORS.value)
                if is_rate and attempt < max_retries:
                    wait = min(60.0, backoff_base * (2 ** (attempt - 1))) + random.random()
                    logger.warning("LLM rate limit/timeout on attempt %d. Waiting %.2fs before retry.", attempt, wait)
                    time.sleep(wait)
                    continue
                logger.error("LLM invocation failed after %d attempts. Error: %s", attempt, e)
                raise

    def _empty_update(self, state: Any, agent_key: str, output: Any, inter_agent_delay: float = get_settings().INTER_AGENTS_DELAY) -> Any:
        logger.info("Agent %s completed. Updating state...", agent_key)
        logger.debug("Applying inter-agent delay of %.2fs", inter_agent_delay)
        time.sleep(inter_agent_delay) 
        return state.model_copy(update={
            "agent_outputs":    {**state.agent_outputs, agent_key: output},
            "completed_agents": state.completed_agents + [agent_key],
            "current_agent":    agent_key
        })