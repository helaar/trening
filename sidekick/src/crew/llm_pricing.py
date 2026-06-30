"""Token pricing for the Inspect pane's cost estimates.

Rates are USD per million tokens, from Anthropic's public pricing. The
cache-read rate is ~0.1x the input rate (Anthropic prompt-caching pricing).

CrewAI's UsageMetrics exposes cached prompt *read* tokens but not cache-*write*
(creation) tokens, so cache writes are billed here at the standard input rate —
a small underestimate of the cache-write premium, made explicit rather than
guessed. Update _RATES when models or prices change.
"""

from dataclasses import dataclass

_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelRates:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float


# Keyed by the bare model id (provider prefix and date suffix are stripped before
# lookup, so "claude-haiku-4-5" matches "anthropic/claude-haiku-4-5-20251001").
_RATES: dict[str, ModelRates] = {
    "claude-sonnet-4-6": ModelRates(input_per_mtok=3.0, output_per_mtok=15.0, cache_read_per_mtok=0.30),
    "claude-haiku-4-5": ModelRates(input_per_mtok=1.0, output_per_mtok=5.0, cache_read_per_mtok=0.10),
}


def get_rates(model: str | None) -> ModelRates | None:
    """Return rates for a model id, or None if it is not in the pricing table."""
    bare = (model or "").split("/", 1)[-1]
    for key, rates in _RATES.items():
        if bare.startswith(key):
            return rates
    return None


def estimate_cost_usd(
    model: str | None,
    prompt_tokens: int,
    cached_prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate USD cost from token counts, or None if the model price is unknown.

    `prompt_tokens` is the full input count (including cached reads); cached reads
    are re-priced at the cheaper cache-read rate.
    """
    rates = get_rates(model)
    if rates is None:
        return None
    non_cached_input = max(prompt_tokens - cached_prompt_tokens, 0)
    cost = (
        non_cached_input * rates.input_per_mtok
        + cached_prompt_tokens * rates.cache_read_per_mtok
        + completion_tokens * rates.output_per_mtok
    ) / _MILLION
    return round(cost, 6)
