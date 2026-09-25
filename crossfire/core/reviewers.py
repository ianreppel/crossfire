"""Reviewer-to-candidate assignment algorithm."""

from __future__ import annotations


def assign_reviewers(
    *,
    reviewers: list[str] | tuple[str, ...],
    num_candidates: int,
    num_reviewers_per_candidate: int,
    round_num: int,
    models_used_this_round: set[str],
) -> dict[int, list[str]] | None:
    """Assigns reviewer models to candidates with each reviewer appearing at most once per round.

    Reviewers are listed cheapest-first, with the models that both gateways serve at the head. The window slides one
    place per round, so early rounds stay inside that cheap, portable head and only long runs reach the pricier or
    gateway-specific tail. A single round never repeats a reviewer; models may recur across rounds.
    """
    available = [name for name in reviewers if name not in models_used_this_round]
    required = num_candidates * num_reviewers_per_candidate
    if required == 0:
        return {}
    if len(available) < required:
        return None

    first = (round_num - 1) % len(available)
    selected = [available[(first + offset) % len(available)] for offset in range(required)]

    assignments: dict[int, list[str]] = {}
    for candidate_index in range(num_candidates):
        start = candidate_index * num_reviewers_per_candidate
        assignments[candidate_index] = selected[start : start + num_reviewers_per_candidate]

    return assignments
