"""
Pydantic models describing the structured output returned by the
Course Recommendation Assistant.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    """Traceability metadata linking a recommendation back to the catalog."""

    course_id: str = Field(description="Unique ID of the course in the catalog")
    course_name: str
    source: str = Field(description="Origin identifier for this course document")
    experience_level: str
    duration: str


class RecommendedCourse(BaseModel):
    course_name: str = Field(description="Exact course name as it appears in the catalog")
    skills_covered: List[str] = Field(description="Key skills this course teaches")
    duration: str = Field(description="Course duration, e.g. '20 hours'")
    experience_level: str = Field(description="Beginner / Intermediate / Advanced")
    match_reason: str = Field(description="Why this specific course matches the learner's request")
    source: SourceMetadata = Field(
        default_factory=lambda: SourceMetadata(
            course_id="unknown",
            course_name="unknown",
            source="unknown",
            experience_level="unknown",
            duration="unknown",
        ),
        description="Source metadata for this recommendation",
    )


class RecommendationResponse(BaseModel):
    """Top-level structured response returned for every user query."""

    recommended_courses: List[RecommendedCourse] = Field(
        description="The specific course(s) recommended to the learner"
    )
    reason: str = Field(description="Overall reasoning behind the recommendation")
    prerequisites: List[str] = Field(
        description="Prerequisites the learner still needs before/alongside the recommendation"
    )
    learning_sequence: List[str] = Field(
        description="Ordered list of course names forming the full learning path"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model's confidence in this recommendation, 0-1"
    )
    total_learning_hours: Optional[float] = Field(
        default=None,
        description="Total hours to complete the learning_sequence, computed by the calculate_total_learning_hours tool",
    )
    tool_output: Optional[str] = Field(
        default=None,
        description="Raw output string from the custom learning-hours tool, kept for transparency",
    )
