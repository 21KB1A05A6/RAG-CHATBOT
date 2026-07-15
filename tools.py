"""
Custom LangChain tool that calculates total learning hours for a
sequence of recommended courses, plus a plain helper used internally
by the RAG engine to get a numeric total.
"""

from typing import List
from langchain_core.tools import tool
from courses_data import COURSES

_DURATION_BY_NAME = {c["course_name"].lower(): c["duration_hours"] for c in COURSES}


@tool
def calculate_total_learning_hours(course_names: List[str]) -> str:
    """Calculate the total number of learning hours required to complete a
    given list of course names, using the course catalog's duration data.

    Args:
        course_names: A list of course names (should match catalog names).

    Returns:
        A human-readable summary with the total hours and any course names
        that could not be matched in the catalog.
    """
    total = 0
    matched = []
    unmatched = []

    for name in course_names:
        hours = _DURATION_BY_NAME.get(name.strip().lower())
        if hours is None:
            unmatched.append(name)
        else:
            total += hours
            matched.append((name, hours))

    summary = f"Total learning hours: {total}"
    if matched:
        breakdown = "; ".join(f"{name}: {hours}h" for name, hours in matched)
        summary += f" ({breakdown})"
    if unmatched:
        summary += f" | Not found in catalog: {', '.join(unmatched)}"
    return summary


def calculate_total_hours_raw(course_names: List[str]) -> float:
    """Plain numeric version used internally by the RAG engine so the
    Pydantic response can carry a float rather than a string."""
    total = 0.0
    for name in course_names:
        hours = _DURATION_BY_NAME.get(name.strip().lower())
        if hours:
            total += hours
    return total
