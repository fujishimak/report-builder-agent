from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

ALLOWED_GRAPH_SCHEMAS = []


def register_schema(cls):
    """Automatically logs theclass for LangGraph msgpack serialisation"""
    ALLOWED_GRAPH_SCHEMAS.append((cls.__module__, cls.__name__))
    return cls


@register_schema
class DocumentChunk(BaseModel):
    """Represents a chunk of document content"""

    doc_id: str = Field(description="Document identifier")
    content: str = Field(description="The actual text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    relevance_score: float = Field(
        default=0.0, description="Relevance score for retrieval"
    )


# TODO: Implement the AnswerResponse schema for structured Q&A responses.
# This schema should include fields for the question, answer, sources, confidence, and timestamp.
# Refer to README.md Task 1.1 for detailed field requirements.
@register_schema
class AnswerResponse(BaseModel):
    """Structured response for Q&A tasks"""

    question: str = Field(description="The original user question")
    answer: str = Field(description="The generated answer")
    sources: list[str] = Field(
        default_factory=list, description="List of source document IDs used"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the response was generated"
    )


@register_schema
class SummarizationResponse(BaseModel):
    """Structured response for summarization tasks"""

    original_length: int = Field(description="Length of original text")
    summary: str = Field(description="The generated summary")
    key_points: list[str] = Field(description="List of key points extracted")
    document_ids: list[str] = Field(
        default_factory=list, description="Documents summarized"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the response was generated"
    )


@register_schema
class CalculationResponse(BaseModel):
    """Structured response for calculation tasks"""

    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculated result")
    explanation: str = Field(description="Step-by-step explanation")
    units: str | None = Field(default=None, description="Units if applicable")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the calculation was performed"
    )


@register_schema
class UpdateMemoryResponse(BaseModel):
    """Response after updating memory"""

    summary: str = Field(description="Summary of the conversation up to this point")
    document_ids: list[str] = Field(
        default_factory=list,
        description="List of documents ids that are relevant to the users last message",
    )


# TODO: Implement the UserIntent schema for intent classification.
# This schema should include fields for intent_type, confidence, and reasoning.
# Refer to README.md Task 1.2 for detailed field requirements.
@register_schema
class UserIntent(BaseModel):
    """User intent classification - TO BE IMPLEMENTED"""

    intent_type: Literal["qa", "summarization", "calculation", "unknown"] = Field(
        description="The classified intent"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in classification"
    )
    reasoning: str = Field(description="Explanation for the classification")


@register_schema
class SessionState(BaseModel):
    """Session state"""

    session_id: str
    user_id: str
    conversation_history: list[TypedDict] = Field(default_factory=list)
    document_context: list[str] = Field(
        default_factory=list, description="Active document IDs"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the session was created"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="When the session was last updated"
    )
