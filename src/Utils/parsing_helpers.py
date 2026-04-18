from typing import Any, Type, Tuple
import logging

logger = logging.getLogger(__name__)


def coerce_bool(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"): 
            return True
        if v in ("false", "no", "0"): 
            return False
    return value


def coerce_str(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return value


def safe_parse(model_cls: Type, data: Any, context: str = None) -> Tuple[Any, bool]:
    """Attempt to parse `data` into `model_cls` using Pydantic v2 .model_validate.
    Returns a tuple (instance_or_none, success_flag). On failure logs a warning and returns (None, False).
    """
    try:
        # allow dict-like or raw objects
        if hasattr(model_cls, "model_validate"):
            inst = model_cls.model_validate(data)
        else:
            inst = model_cls(**(data or {}))
        return inst, True
    except Exception as e:
        ctx = f" for {context}" if context else ""
        logger.warning("Failed to parse %s%s: %s", getattr(model_cls, "__name__", str(model_cls)), ctx, exc_info=e)
        return None, False
