"""Pydantic output models for the daily analysis crew tasks.

Passed as `output_pydantic` to each CrewAI Task. CrewAI injects the JSON
schema into the prompt — field descriptions function as LLM instructions.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RiskFlag(BaseModel):
    description: str = Field(
        description="Concise risk description, max 20 words. Only materially significant findings."
    )
    severity: Literal["low", "moderate", "high"] = Field(
        description="low=informational, moderate=warrants attention, high=action required"
    )


class WorkoutOutput(BaseModel):
    activity_id: int | None = Field(default=None, description="Numeric activity_id if present")
    session_name: str = Field(description="Athlete's session name, or sport + date if absent")
    is_commute: bool = Field(description="True if this is a commute/transport activity")
    is_erg_mode: bool = Field(
        description="True if ERG mode detected. When True, do NOT comment on power stability or pacing evenness."
    )
    executive_summary: str = Field(
        description="2-3 sentence coach-facing takeaway, max 50 words. Omit for commutes."
    )
    quantitative_summary: str = Field(
        description=(
            "Key metrics with comparisons: TSS, IF, power vs FTP, HR response, zone distribution. "
            "Cycling cadence=RPM, running cadence=SPM — never compare across sports. "
            "Running power from GPS/foot pod is not cycling power. "
            "Ignore isolated abnormal values at activity start."
        )
    )
    qualitative_assessment: str = Field(
        description=(
            "Execution quality, pacing, zone adherence. "
            "For ERG rides: focus on cadence patterns and HR response only."
        )
    )
    progress_indicators: str = Field(
        description="How this session fits overall training progression, 1-2 sentences."
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list,
        description="Empty list if no significant concerns. No ERG power flags.",
    )
    coach_recommendations: list[str] = Field(
        default_factory=list,
        description="Specific actionable adjustments, max 30 words each. Empty for commutes.",
    )
    commute_note: str = Field(
        default="",
        description="One sentence for commute activities. Empty for non-commutes.",
    )
    data_gaps: list[str] = Field(
        default_factory=list,
        description="Missing data that limits analysis, e.g. ['No HR data']. Empty if complete.",
    )


class WorkoutAnalysisOutput(BaseModel):
    """Output for the workout_performance_analyst task."""

    daily_summary: str = Field(
        description="Key highlight, most important risk/achievement, primary coach recommendation. Max 150 words."
    )
    workouts: list[WorkoutOutput] = Field(
        default_factory=list,
        description="Key sessions first, commutes last. Empty if no workout data.",
    )
    no_data: bool = Field(default=False, description="True if no workout data available for this date.")


class RecoveryBaseline(BaseModel):
    hrv_typical: float | None = Field(default=None, description="Typical HRV over analysis window")
    resting_hr_typical: float | None = Field(default=None, description="Typical resting HR (bpm)")
    sleep_hours_typical_weekday: float | None = Field(
        default=None,
        description=(
            "Athlete's typical sleep duration on weekdays. "
            "Compare weekday sleep against this — not a universal baseline."
        ),
    )
    sleep_hours_typical_weekend: float | None = Field(
        default=None,
        description=(
            "Athlete's typical sleep duration on weekends. "
            "Compare weekend sleep against this — not a universal baseline."
        ),
    )
    sleep_quality_typical: float | None = Field(
        default=None,
        description="Typical self-reported sleep quality score over the analysis window",
    )
    readiness_typical: float | None = Field(default=None, description="Typical self-reported readiness")


class RestitutionAnalysisOutput(BaseModel):
    """Output for the restitution_analyst task."""

    data_quality_note: str = Field(
        default="",
        description="Prominently state if fewer than 5 days have restitution data. Empty otherwise.",
    )
    recovery_baseline: RecoveryBaseline = Field(
        description="Athlete's typical recovery metrics derived from the analysis window"
    )
    trend_analysis: str = Field(
        description=(
            "Direction and magnitude of HRV, resting HR, sleep (hours and quality), and readiness "
            "across the analysis period. Focus on multi-day patterns, not daily noise. "
            "Max 100 words."
        )
    )
    load_recovery_correlation: str = Field(
        description=(
            "How HRV, resting HR, and readiness respond to TSS/IF patterns. "
            "Apply the 1-day temporal lag: HRV and resting HR on day N reflect training from day N-1. "
            "Distinguish acute fatigue (short spikes, recover within 1-2 days) from chronic "
            "accumulation (sustained suppression across multiple days). "
            "Max 100 words."
        )
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list,
        description=(
            "Genuine multi-day patterns only: sustained HRV suppression, rising resting HR trend, "
            "persistent low readiness relative to load, progressive sleep decline. "
            "Empty list if no significant concerns."
        ),
    )
    overall_recovery_quality: Literal["good", "adequate", "concerning"] = Field(
        description="good=well-recovered, adequate=manageable, concerning=action required"
    )
    coach_recommendations: list[str] = Field(
        default_factory=list,
        description="1-2 specific, actionable recovery-focused adjustments.",
    )


class MemoryDraft(BaseModel):
    """A new memory observation to be stored."""

    category: Literal["recovery", "habit", "performance", "risk", "goal"] = Field(
        description="Category this memory belongs to"
    )
    scope: Literal["recent", "long_term"] = Field(
        description="recent=30-day relevance, long_term=multi-month pattern"
    )
    content: str = Field(description="1-3 sentence natural language observation about the athlete")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident you are in this observation (0.0-1.0)")
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How much this should shape coaching (0.0-1.0), independent of confidence. "
        "Risks, active goals, and strong recurring habits are high; incidental observations are low.",
    )
    evidence_dates: list[str] = Field(
        default_factory=list,
        description="YYYY-MM-DD dates from the context that support this observation",
    )


class MemoryUpdate(BaseModel):
    """An update to an existing memory."""

    memory_id: str = Field(description="ID of the memory to update")
    content: str = Field(description="Updated 1-3 sentence observation")
    confidence: float = Field(ge=0.0, le=1.0, description="Updated confidence score")
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Updated standing significance to coaching (0.0-1.0), independent of confidence",
    )
    evidence_dates: list[str] = Field(default_factory=list, description="Updated supporting dates")


class MemoryExtractionOutput(BaseModel):
    """Output for the memory_extractor task run after each daily analysis."""

    new_memories: list[MemoryDraft] = Field(
        default_factory=list,
        description="Brand-new observations not covered by existing memories",
    )
    updated_memories: list[MemoryUpdate] = Field(
        default_factory=list,
        description="Refinements or corrections to existing memories",
    )
    deactivated_memory_ids: list[str] = Field(
        default_factory=list,
        description="IDs of memories that are now contradicted or no longer relevant",
    )


class MemoryConsolidationOutput(BaseModel):
    """Output for the memory_consolidation task (weekly background job)."""

    updates: list[MemoryUpdate] = Field(
        default_factory=list,
        description="Updated content for memories whose pattern has changed",
    )
    promotions: list[str] = Field(
        default_factory=list,
        description="memory_ids to promote from recent to long_term scope",
    )
    deactivations: list[str] = Field(
        default_factory=list,
        description="memory_ids that are obsolete or contradicted",
    )
    new_long_term: list[MemoryDraft] = Field(
        default_factory=list,
        description="New long-term pattern observations identified across the review window",
    )


class CoachingOutput(BaseModel):
    """Output for the daily_coach task."""

    todays_recap: str = Field(
        description="Coach-facing: execution vs plan, key metrics in plain language. Max 60 words."
    )
    key_takeaway: str = Field(
        description="Coach-facing: single most important observation. Max 40 words."
    )
    looking_ahead: str = Field(
        description="Coach-facing: one practical suggestion for next session or recovery. Max 40 words."
    )
    coach_notes: str = Field(
        default="",
        description="Internal flags/caveats NOT for athlete: HRV concerns, load monitoring notes. Max 80 words.",
    )
    athlete_message: str = Field(
        description=(
            "Athlete-facing: casual conversational message. NO headers or bullets. Max 150 words. "
            "Do NOT praise ERG power stability (automatic). "
            "Sleep: compare to athlete's own weekday/weekend pattern, not a fixed baseline. "
            "No false continuity ('again', 'as usual') unless in the data. "
            "Empathetic if sessions missed — no rigid enforcement language. "
            "Verify all claims against provided context."
        )
    )
