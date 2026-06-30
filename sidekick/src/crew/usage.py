"""Collect per-run and per-agent LLM token usage and cost from a finished crew.

CrewAI's event bus does not carry token counts on individual LLM calls, but each
agent accumulates usage in a private `_token_process` over the run (populated by
litellm's success callback). After `crew.kickoff()` returns, we read those
counters per agent and price each agent by its own model, since agents in a crew
may use different LLMs.
"""

import logging
from typing import TYPE_CHECKING, Any

from crew.llm_pricing import estimate_cost_usd
from models.prompt_log import AgentUsage, RunUsage

if TYPE_CHECKING:
    from crewai import Crew

logger = logging.getLogger(__name__)


def _agent_model(agent: Any) -> str | None:
    llm = getattr(agent, "llm", None)
    if llm is None:
        return None
    if isinstance(llm, str):
        return llm
    return getattr(llm, "model", None)


def _sum_costs(per_agent: list[AgentUsage]) -> float | None:
    known = [a.cost_usd for a in per_agent if a.cost_usd is not None]
    return round(sum(known), 6) if known else None


def collect_run_usage(
    crew: "Crew", athlete_id: int, crew_name: str, run_id: str
) -> RunUsage:
    """Build a RunUsage from the crew's per-agent token counters (call after kickoff).

    Usage telemetry must never break the analysis it measures, so any failure here
    falls back to a zeroed RunUsage rather than propagating.
    """
    try:
        return _collect_run_usage(crew, athlete_id, crew_name, run_id)
    except Exception:
        logger.exception("Failed to collect run usage for run %s", run_id)
        return RunUsage(run_id=run_id, athlete_id=athlete_id, crew_name=crew_name)


def _collect_run_usage(
    crew: "Crew", athlete_id: int, crew_name: str, run_id: str
) -> RunUsage:
    per_agent: list[AgentUsage] = []
    for agent in crew.agents:
        token_process = getattr(agent, "_token_process", None)
        if token_process is None:
            continue
        try:
            summary = token_process.get_summary()
        except Exception:
            logger.exception("Failed to read token usage for an agent in run %s", run_id)
            continue
        model = _agent_model(agent)
        per_agent.append(
            AgentUsage(
                agent_role=getattr(agent, "role", None),
                model=model,
                prompt_tokens=summary.prompt_tokens,
                cached_prompt_tokens=summary.cached_prompt_tokens,
                completion_tokens=summary.completion_tokens,
                total_tokens=summary.total_tokens,
                successful_requests=summary.successful_requests,
                cost_usd=estimate_cost_usd(
                    model,
                    summary.prompt_tokens,
                    summary.cached_prompt_tokens,
                    summary.completion_tokens,
                ),
            )
        )

    return RunUsage(
        run_id=run_id,
        athlete_id=athlete_id,
        crew_name=crew_name,
        prompt_tokens=sum(a.prompt_tokens for a in per_agent),
        cached_prompt_tokens=sum(a.cached_prompt_tokens for a in per_agent),
        completion_tokens=sum(a.completion_tokens for a in per_agent),
        total_tokens=sum(a.total_tokens for a in per_agent),
        successful_requests=sum(a.successful_requests for a in per_agent),
        cost_usd=_sum_costs(per_agent),
        per_agent=per_agent,
    )
