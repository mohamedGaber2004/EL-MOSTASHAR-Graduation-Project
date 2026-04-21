"""
Utility functions for cleaning and validating agent outputs.
"""
from typing import Any, Dict

def remove_nulls(obj: Any) -> Any:
    """
    Recursively remove keys with None/null values from dicts and lists.
    """
    if isinstance(obj, dict):
        return {k: remove_nulls(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [remove_nulls(v) for v in obj if v is not None]
    else:
        return obj


def clean_agent_output(output: Dict) -> Dict:
    """
    Clean agent output dict by removing nulls and empty fields.
    """
    cleaned = remove_nulls(output)
    if isinstance(cleaned, dict):
        return {k: v for k, v in cleaned.items() if v not in (None, [], {}, "")}
    return cleaned
