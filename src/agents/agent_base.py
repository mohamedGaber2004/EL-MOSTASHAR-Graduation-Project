import logging
import time
import random
from typing import Any, Callable, List, Literal

from src.Config import get_settings
from src.Utils.agents_enums import AgentsEnums
from src.Graph.graph_helpers import _now

logger = logging.getLogger(__name__)

LLMProvider = Literal["oss", "qwen", "llama"]


class AgentBase:
    """Base class for all court-simulation agents."""

    def __init__(self,model_config_key: str,temp_config_key: str,prompt: str,llm_provider: LLMProvider = "llama") -> None:
        self.cfg        = get_settings()
        self.model_name: str   = getattr(self.cfg, model_config_key)
        self.temperature: float = getattr(self.cfg, temp_config_key)
        self.prompt: str       = prompt
        self._llm_provider     = llm_provider
        self._llm              = self._init_llm(llm_provider)

        logger.debug(
            "Initialized %s | model=%s | temp=%s | provider=%s",
            self.__class__.__name__,
            self.model_name,
            self.temperature,
            llm_provider,
        )

    # ── LLM factory ───────────────────────────────────────────────────

    def _init_llm(self, provider: LLMProvider):
        """Instantiate only the requested LLM provider."""
        if provider == "oss":
            from src.LLMs.GPT_OSS_Model import get_llm_oss
            return get_llm_oss(self.model_name, self.temperature)
        if provider == "qwen":
            from src.LLMs.Qwen_Model import get_llm_qwn
            return get_llm_qwn(self.model_name, self.temperature)
        if provider == "llama":
            from src.LLMs.Llama_Model import get_llm_llama
            return get_llm_llama(self.model_name, self.temperature)
        raise ValueError(f"Unknown LLM provider: {provider!r}")

    # ── retry wrapper ─────────────────────────────────────────────────

    def _llm_invoke_with_retries(self,llm_callable: Callable,messages: List[Any],max_retries: int = None,backoff_base: float = 2.0) -> Any:
        if max_retries is None:
            max_retries = self.cfg.INVOKATION_MAX_RETRIES

        logger.debug(
            "Invoking LLM for %s (max_retries=%d)",
            self.__class__.__name__,
            max_retries,
        )

        for attempt in range(1, max_retries + 1):
            try:
                result = llm_callable.invoke(messages)
                if attempt > 1:
                    logger.info("LLM succeeded on attempt %d", attempt)
                return result
            except Exception as e:
                err = str(e).lower()
                is_rate = any(
                    k in err for k in AgentsEnums.AGENT_INVOKATION_ERRORS.value
                )
                if is_rate and attempt < max_retries:
                    wait = (
                        min(60.0, backoff_base * (2 ** (attempt - 1)))
                        + random.random()
                    )
                    logger.warning(
                        "Rate-limit / timeout on attempt %d — waiting %.2fs",
                        attempt,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "LLM invocation failed after %d attempt(s): %s", attempt, e
                )
                raise

    # ── state update helper ───────────────────────────────────────────

    def _empty_update(self,state: Any,agent_key: str,output: Any,inter_agent_delay: float = None) -> Any:
        if inter_agent_delay is None:
            inter_agent_delay = self.cfg.INTER_AGENTS_DELAY

        logger.info("Agent [%s] complete — updating state.", agent_key)
        if inter_agent_delay > 0:
            logger.debug("Inter-agent delay: %.2fs", inter_agent_delay)
            time.sleep(inter_agent_delay)

        return state.model_copy(
            update={
                "agent_outputs":    {**state.agent_outputs, agent_key: output},
                "completed_agents": state.completed_agents + [agent_key],
                "current_agent":    agent_key,
                "last_updated":     _now(),          # was missing in original
            }
        )