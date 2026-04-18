"""Reviewer-to-candidate assignment algorithm."""

from __future__ import annotations

import random


def assign_reviewers(
    *,
    reviewers: list[str] | tuple[str, ...],
    num_candidates: int,
    num_reviewers_per_candidate: int,
    round_num: int,
    models_used_this_round: set[str],
) -> dict[int, list[str]] | None:
    """Assigns reviewer models to candidates with each reviewer appearing at most once per round."""
    available = [name for name in reviewers if name not in models_used_this_round]
    required = num_candidates * num_reviewers_per_candidate
    if len(available) < required:
        return None

    pool = list(available)
    random.Random(round_num).shuffle(pool)

    assignments: dict[int, list[str]] = {}
    for candidate_index in range(num_candidates):
        start = candidate_index * num_reviewers_per_candidate
        group = pool[start : start + num_reviewers_per_candidate]
        if len(group) < num_reviewers_per_candidate:
            return None
        assignments[candidate_index] = group

    return assignments
