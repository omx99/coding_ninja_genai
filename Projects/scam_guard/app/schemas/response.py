from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ScamLabel(str, Enum):
    SCAM = "Scam"
    NOT_SCAM = "Not Scam"
    UNCERTAIN = "Uncertain"


class IntentType(str, Enum):
    OTP_FRAUD = "OTP Fraud"
    PHISHING = "Phishing"
    ACCOUNT_SUSPENSION = "Account Suspension"
    REWARD_MANIPULATION = "Reward Manipulation"
    FEAR_TACTICS = "Fear Tactics"
    FAKE_AUTHORITY = "Fake Authority"
    LOAN_SCAM = "Loan Scam"
    URGENCY = "Urgency"
    SERVICE_REMINDER = "Service Reminder"
    INFORMATIONAL_ALERT = "Informational Alert"
    TRANSACTIONAL_NOTIFICATION = "Transactional Notification"
    ORDER_CONFIRMATION = "Order Confirmation"
    ACCOUNT_UPDATE = "Account Update"
    MARKETING_MESSAGE = "Marketing Message"
    UNKNOWN = "Unknown"


class ScamResult(BaseModel):
    label: ScamLabel = Field(
        ...,
        description="Classification result: Scam, Not Scam, or Uncertain"
    )
    intent_type: IntentType = Field(
        ...,
        description="The underlying intent or manipulation type detected"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of classification between 0.0 and 1.0"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Brief explanation of why this was classified as such"
    )
    prompt_version_used: Optional[str] = Field(
        default=None,
        description="Which prompt version produced this result"
    )


class MessageResult(BaseModel):
    """Single row result — original message + LLM result combined."""
    message: str
    actual_label: str                # from dataset
    predicted_label: str             # from LLM
    intent_type: str
    confidence_score: float
    reasoning: str
    is_correct: bool                 # actual vs predicted match
    prompt_version_used: str


class BatchSummary(BaseModel):
    """Aggregate stats across all processed messages."""
    total_processed: int
    correct_predictions: int
    accuracy: float                  # correct / total
    scam_detected: int
    not_scam_detected: int
    uncertain_detected: int
    prompt_version_used: str


class BatchResult(BaseModel):
    """Full batch response — summary + per message results."""
    summary: BatchSummary
    results: List[MessageResult]