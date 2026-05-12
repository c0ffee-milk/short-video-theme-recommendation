#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    PROJECT_ROOT / "data_clean" / "interactions_clean.csv",
    PROJECT_ROOT / "data_clean" / "video_catalog.csv",
    PROJECT_ROOT / "data_clean" / "user_profile.csv",
    PROJECT_ROOT / "output" / "tables" / "theme_behavior_summary.csv",
    PROJECT_ROOT / "output" / "tables" / "user_group_theme_summary.csv",
    PROJECT_ROOT / "output" / "tables" / "user_theme_preference.csv",
    PROJECT_ROOT / "output" / "rules" / "theme_behavior_rules.csv",
    PROJECT_ROOT / "output" / "recommendations" / "baseline_recommendations.csv",
    PROJECT_ROOT / "output" / "recommendations" / "rule_enhanced_recommendations.csv",
    PROJECT_ROOT / "output" / "tables" / "evaluation_summary.csv",
]

REQUIRED_FIGURES = [
    PROJECT_ROOT / "output" / "figures" / "top_theme_watch_time.png",
    PROJECT_ROOT / "output" / "figures" / "theme_interaction_rates.png",
    PROJECT_ROOT / "output" / "figures" / "recommendation_evaluation.png",
]


def require_non_empty(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"empty file: {path}")


def main() -> None:
    for path in REQUIRED_FILES + REQUIRED_FIGURES:
        require_non_empty(path)

    interactions = pd.read_csv(PROJECT_ROOT / "data_clean" / "interactions_clean.csv")
    themes = pd.read_csv(PROJECT_ROOT / "output" / "tables" / "theme_behavior_summary.csv")
    rules = pd.read_csv(PROJECT_ROOT / "output" / "rules" / "theme_behavior_rules.csv")
    evaluation = pd.read_csv(PROJECT_ROOT / "output" / "tables" / "evaluation_summary.csv")
    baseline = pd.read_csv(PROJECT_ROOT / "output" / "recommendations" / "baseline_recommendations.csv")
    enhanced = pd.read_csv(PROJECT_ROOT / "output" / "recommendations" / "rule_enhanced_recommendations.csv")

    print("Output validation passed")
    print(f"clean interactions: {len(interactions):,}")
    print(f"users: {interactions['user_id'].nunique():,}")
    print(f"videos: {interactions['pid'].nunique():,}")
    print(f"themes: {themes['root_id'].nunique():,}")
    print(f"rules: {len(rules):,}")
    print(f"baseline recommendations: {len(baseline):,}")
    print(f"rule-enhanced recommendations: {len(enhanced):,}")
    print("\nTop 5 themes by exposure:")
    print(themes.head(5)[["root_id", "root_name_cn", "exposure_count", "avg_watch_ratio", "effective_view_rate"]].to_string(index=False))
    print("\nEvaluation:")
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
