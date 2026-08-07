from pydantic import BaseModel


class FitnessPoint(BaseModel):
    """One day of fitness/fatigue/form, sourced from Intervals.icu's wellness data.

    Not persisted — a plain response shape, not a Mongo document. TSB is derived
    locally (ctl - atl); it isn't an Intervals.icu-specific field.
    """

    date: str  # YYYY-MM-DD
    ctl: float | None = None
    atl: float | None = None
    tsb: float | None = None
    ramp_rate: float | None = None
