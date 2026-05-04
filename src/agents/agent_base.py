import logging
import time
import random
from typing import Any, Callable, List

from src.Config import get_settings
from src.Utils.agents_enums import AgentsEnums
from src.Graph.graph_helpers import _now

logger = logging.getLogger(__name__)

class AgentBase:
    """Base class for all court-simulation agents."""

    def __init__(self,model_config_key: str,temp_config_key: str,prompt: str) -> None:
        self.cfg = get_settings()
        self.agent: str = model_config_key
        self.model_name: str   = getattr(self.cfg, model_config_key)
        self.temperature: float = getattr(self.cfg, temp_config_key)
        self.prompt: str | List[str] = prompt
        self._llm = self._init_llm(self.agent)

        logger.debug(
            "Initialized %s | model=%s | temp=%s | provider=%s",
            self.__class__.__name__,
            self.model_name,
            self.temperature,
        )

    # ── LLM factory ───────────────────────────────────────────────────

    def _init_llm(self, agent_name:str):
        """Instantiate only the requested LLM provider."""
        if agent_name == "DATA_INGESTION_MODEL":
            from src.LLMs.DATA_INGESTION_MODEL import get_ingesion_model
            return get_ingesion_model()['open_router_llm']
        if agent_name == "PROCEDURAL_AUDITOR_MODEL":
            from src.LLMs.PROCEDURAL_AUDITOR_MODEL import get_procedural_model
            return get_procedural_model()
        if agent_name == "LEGAL_RESEARCHER_MODEL":
            from src.LLMs.LEGAL_RESEARCHER_MODEL import get_legal_reasearcher_model
            return get_legal_reasearcher_model()
        if agent_name == "EVIDENCE_SCORING_MODEL":
            from src.LLMs.EVIDENCE_SCORING_MODEL import get_evidence_scoring_model
            return get_evidence_scoring_model()
        if agent_name == "DEFENSE_AGENT_MODEL":
            from src.LLMs.DEFENSE_AGENT_MODEL import get_defense_model
            return get_defense_model()
        if agent_name == "JUDJICAL_PRINCIPLE_AGENT":
            from src.LLMs.JUDJICAL_PRINCIPLE_AGENT import get_judjical_principle_model
            return get_judjical_principle_model()
        if agent_name == "JUDGE_MODEL":
            from src.LLMs.JUDGE_MODEL import get_judge_model
            return get_judge_model()
        raise ValueError(f"Unknown LLM provider: {agent_name!r}")

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
                "completed_agents": state.completed_agents + [agent_key],
                "current_agent":    agent_key,
                "last_updated":     _now(),          # was missing in original
            }
        )