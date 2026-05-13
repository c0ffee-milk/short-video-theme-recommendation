#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data_raw" / "shortvideo_tiny"
CLEAN_DIR = PROJECT_ROOT / "data_clean"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
RULE_DIR = OUTPUT_DIR / "rules"
REC_DIR = OUTPUT_DIR / "recommendations"
FIG_DIR = OUTPUT_DIR / "figures"

BOOL_COLUMNS = ["cvm_like", "click", "comment", "follow", "collect", "forward", "hate"]
BEHAVIOR_COLUMNS = ["effective_view", "cvm_like", "collect", "comment", "forward", "hate"]
TOP_K = 10


def ensure_dirs() -> None:
    for path in [CLEAN_DIR, TABLE_DIR, RULE_DIR, REC_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
        .astype(bool)
    )


def unique_join(values: pd.Series) -> str:
    items = [str(v).strip() for v in values.dropna() if str(v).strip() and str(v).strip() != "nan"]
    return "|".join(dict.fromkeys(items))


def load_category_map() -> pd.DataFrame:
    categories = pd.read_csv(RAW_DIR / "categories_cn_en.csv")
    root_map = (
        categories[["category_id", "category_name_cn", "category_name_en"]]
        .drop_duplicates("category_id")
        .rename(
            columns={
                "category_id": "root_id",
                "category_name_cn": "root_name_cn",
                "category_name_en": "root_name_en",
            }
        )
    )
    return root_map


def clean_interactions() -> pd.DataFrame:
    interactions = pd.read_csv(RAW_DIR / "interaction_sampled.csv", low_memory=False)

    for column in BOOL_COLUMNS:
        interactions[column] = to_bool(interactions[column])

    interactions["watch_time"] = pd.to_numeric(interactions["watch_time"], errors="coerce").fillna(0)
    interactions["duration"] = pd.to_numeric(interactions["duration"], errors="coerce")
    safe_duration = interactions["duration"].where(interactions["duration"] > 0)
    interactions["effective_view"] = interactions["watch_time"] > 3
    interactions["watch_ratio"] = (interactions["watch_time"] / safe_duration).replace([np.inf, -np.inf], np.nan).fillna(0)
    interactions["watch_ratio"] = interactions["watch_ratio"].clip(lower=0, upper=1)

    interactions["age"] = pd.to_numeric(interactions["age"], errors="coerce")
    interactions["mod_price"] = pd.to_numeric(interactions["mod_price"], errors="coerce")
    interactions["age_group"] = pd.cut(
        interactions["age"],
        bins=[0, 18, 25, 35, 45, 60, 120],
        labels=["<=18", "19-25", "26-35", "36-45", "46-60", "60+"],
        include_lowest=True,
    ).astype(str)
    interactions.loc[interactions["age"].isna(), "age_group"] = "unknown"
    interactions["price_group"] = pd.cut(
        interactions["mod_price"],
        bins=[-1, 1000, 2000, 4000, 7000, np.inf],
        labels=["<=1000", "1001-2000", "2001-4000", "4001-7000", "7000+"],
    ).astype(str)
    interactions.loc[interactions["mod_price"].isna(), "price_group"] = "unknown"

    root_map = load_category_map()
    interactions = interactions.merge(root_map, on="root_id", how="left")
    interactions["root_name_cn"] = interactions["root_name_cn"].fillna("主题" + interactions["root_id"].astype(str))
    interactions["root_name_en"] = interactions["root_name_en"].fillna("theme_" + interactions["root_id"].astype(str))

    group_keys = ["user_id", "pid", "exposed_time"]
    first_columns = [
        "author_id",
        "category_id",
        "category_level",
        "parent_id",
        "root_id",
        "author_fans_count",
        "watch_time",
        "duration",
        "title",
        "p_hour",
        "p_date",
        "gender",
        "age",
        "age_group",
        "mod_price",
        "price_group",
        "fre_city",
        "fre_community_type",
        "fre_city_level",
        "root_name_cn",
        "root_name_en",
        "watch_ratio",
    ]
    agg = {column: "first" for column in first_columns}
    agg.update({column: "max" for column in BOOL_COLUMNS + ["effective_view"]})
    agg["tag_name"] = unique_join

    cleaned = interactions.groupby(group_keys, as_index=False).agg(agg)
    cleaned["interaction_score"] = (
        0.45 * cleaned["watch_ratio"]
        + 0.20 * cleaned["effective_view"].astype(float)
        + 0.15 * cleaned["cvm_like"].astype(float)
        + 0.10 * cleaned["collect"].astype(float)
        + 0.05 * cleaned["comment"].astype(float)
        + 0.08 * cleaned["forward"].astype(float)
        - 0.20 * cleaned["hate"].astype(float)
    ).clip(lower=0)
    return cleaned


def save_clean_outputs(cleaned: pd.DataFrame) -> None:
    cleaned.to_csv(CLEAN_DIR / "interactions_clean.csv", index=False)

    video_columns = [
        "pid",
        "author_id",
        "category_id",
        "category_level",
        "parent_id",
        "root_id",
        "root_name_cn",
        "root_name_en",
        "duration",
        "author_fans_count",
        "title",
        "tag_name",
    ]
    cleaned[video_columns].drop_duplicates("pid").to_csv(CLEAN_DIR / "video_catalog.csv", index=False)

    user_columns = ["user_id", "gender", "age", "age_group", "mod_price", "price_group", "fre_city", "fre_community_type", "fre_city_level"]
    cleaned[user_columns].drop_duplicates("user_id").to_csv(CLEAN_DIR / "user_profile.csv", index=False)


def summarize_theme_behavior(cleaned: pd.DataFrame) -> pd.DataFrame:
    grouped = cleaned.groupby(["root_id", "root_name_cn", "root_name_en"], dropna=False)
    summary = grouped.agg(
        exposure_count=("pid", "size"),
        user_count=("user_id", "nunique"),
        video_count=("pid", "nunique"),
        avg_watch_time=("watch_time", "mean"),
        avg_watch_ratio=("watch_ratio", "mean"),
        effective_view_rate=("effective_view", "mean"),
        like_rate=("cvm_like", "mean"),
        collect_rate=("collect", "mean"),
        comment_rate=("comment", "mean"),
        forward_rate=("forward", "mean"),
        hate_rate=("hate", "mean"),
    ).reset_index()
    summary = summary.sort_values(["exposure_count", "avg_watch_ratio"], ascending=[False, False])
    summary.to_csv(TABLE_DIR / "theme_behavior_summary.csv", index=False)
    return summary


def summarize_user_groups(cleaned: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dimension in ["gender", "age_group", "fre_city_level"]:
        group_summary = (
            cleaned.groupby([dimension, "root_id", "root_name_cn"], dropna=False)
            .agg(
                exposure_count=("pid", "size"),
                avg_preference_score=("interaction_score", "mean"),
                avg_watch_ratio=("watch_ratio", "mean"),
                like_rate=("cvm_like", "mean"),
                collect_rate=("collect", "mean"),
                forward_rate=("forward", "mean"),
            )
            .reset_index()
            .rename(columns={dimension: "group_value"})
        )
        group_summary.insert(0, "group_dimension", dimension)
        frames.append(group_summary)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(TABLE_DIR / "user_group_theme_summary.csv", index=False)
    return result


def build_user_theme_preference(cleaned: pd.DataFrame) -> pd.DataFrame:
    preference = (
        cleaned.groupby(["user_id", "root_id", "root_name_cn", "root_name_en"], dropna=False)
        .agg(
            exposure_count=("pid", "size"),
            video_count=("pid", "nunique"),
            avg_watch_ratio=("watch_ratio", "mean"),
            effective_view_rate=("effective_view", "mean"),
            like_count=("cvm_like", "sum"),
            collect_count=("collect", "sum"),
            comment_count=("comment", "sum"),
            forward_count=("forward", "sum"),
            hate_count=("hate", "sum"),
            preference_score=("interaction_score", "sum"),
        )
        .reset_index()
    )
    preference = preference.sort_values(["user_id", "preference_score"], ascending=[True, False])
    preference.to_csv(TABLE_DIR / "user_theme_preference.csv", index=False)
    return preference


def build_transactions(cleaned: pd.DataFrame) -> list[set[str]]:
    transactions: list[set[str]] = []
    for row in cleaned.itertuples(index=False):
        items = {
            f"theme={row.root_id}",
            f"gender={row.gender}",
            f"age_group={row.age_group}",
            f"city_level={row.fre_city_level}",
        }
        for tag in str(row.tag_name).split("|"):
            tag = tag.strip()
            if tag and tag != "nan":
                items.add(f"tag={tag}")
        if row.effective_view:
            items.add("behavior=effective_view")
        if row.cvm_like:
            items.add("behavior=like")
        if row.collect:
            items.add("behavior=collect")
        if row.comment:
            items.add("behavior=comment")
        if row.forward:
            items.add("behavior=forward")
        if row.hate:
            items.add("behavior=hate")
        transactions.append(items)
    return transactions


def mine_rules(cleaned: pd.DataFrame, min_support_count: int = 20, min_confidence: float = 0.05) -> pd.DataFrame:
    transactions = build_transactions(cleaned)
    total = len(transactions)
    item_count: Counter[str] = Counter()
    pair_count: Counter[tuple[str, str]] = Counter()
    triple_count: Counter[tuple[str, str, str]] = Counter()

    for items in transactions:
        for item in items:
            item_count[item] += 1
        for left, right in combinations(sorted(items), 2):
            pair_count[(left, right)] += 1
        selected_left = sorted(i for i in items if i.startswith(("theme=", "tag=", "gender=", "age_group=", "city_level=")))
        behaviors = sorted(i for i in items if i.startswith("behavior="))
        for a, b in combinations(selected_left, 2):
            for behavior in behaviors:
                triple_count[(a, b, behavior)] += 1

    behavior_items = {item for item in item_count if item.startswith("behavior=")}
    rows: list[dict[str, object]] = []

    for (left, right), count in pair_count.items():
        if count < min_support_count:
            continue
        if right in behavior_items and left.startswith(("theme=", "tag=")):
            antecedent = (left,)
            consequent = right
        elif left in behavior_items and right.startswith(("theme=", "tag=")):
            antecedent = (right,)
            consequent = left
        else:
            continue
        confidence = count / item_count[antecedent[0]]
        if confidence < min_confidence:
            continue
        support = count / total
        lift = confidence / (item_count[consequent] / total) if item_count[consequent] else 0
        rows.append(
            {
                "antecedent": " & ".join(antecedent),
                "consequent": consequent,
                "support_count": count,
                "support": support,
                "confidence": confidence,
                "lift": lift,
            }
        )

    for (a, b, behavior), count in triple_count.items():
        if count < min_support_count:
            continue
        antecedent_count = sum(1 for items in transactions if a in items and b in items)
        if antecedent_count == 0:
            continue
        confidence = count / antecedent_count
        if confidence < min_confidence:
            continue
        support = count / total
        lift = confidence / (item_count[behavior] / total) if item_count[behavior] else 0
        rows.append(
            {
                "antecedent": f"{a} & {b}",
                "consequent": behavior,
                "support_count": count,
                "support": support,
                "confidence": confidence,
                "lift": lift,
            }
        )

    rules = pd.DataFrame(rows)
    if rules.empty:
        rules = pd.DataFrame(columns=["antecedent", "consequent", "support_count", "support", "confidence", "lift"])
    else:
        rules = rules.sort_values(["lift", "confidence", "support_count"], ascending=[False, False, False])
    rules.to_csv(RULE_DIR / "theme_behavior_rules.csv", index=False)
    return rules


def split_train_test(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = cleaned.sort_values(["user_id", "exposed_time", "pid"])
    train_parts = []
    test_parts = []
    for _, group in sorted_df.groupby("user_id"):
        if len(group) < 3:
            train_parts.append(group)
            continue
        cut = max(1, int(len(group) * 0.8))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else sorted_df.iloc[:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else sorted_df.iloc[:0].copy()
    return train, test


def normalize_mapping(values: dict[object, float]) -> dict[object, float]:
    if not values:
        return {}
    clean_values = {key: max(float(value), 0.0) for key, value in values.items()}
    max_value = max(clean_values.values())
    if max_value <= 0:
        return {key: 0.0 for key in clean_values}
    return {key: value / max_value for key, value in clean_values.items()}


def split_tags(tag_string: object) -> list[str]:
    return [tag.strip() for tag in str(tag_string).split("|") if tag.strip() and tag.strip() != "nan"]


def build_rule_boosts(rules: pd.DataFrame) -> dict[str, dict[object, float]]:
    features: dict[str, dict[object, float]] = {
        "theme_rule_quality": defaultdict(float),
        "tag_rule_quality": defaultdict(float),
        "group_theme_rule_quality": defaultdict(float),
        "theme_hate_risk": defaultdict(float),
    }
    if rules.empty:
        return {key: {} for key in features}

    behavior_weights = {
        "behavior=effective_view": 0.45,
        "behavior=like": 1.00,
        "behavior=collect": 1.20,
        "behavior=comment": 0.80,
        "behavior=forward": 1.10,
        "behavior=hate": -1.00,
    }
    for row in rules.itertuples(index=False):
        consequent = str(row.consequent)
        if consequent not in behavior_weights:
            continue
        if int(row.support_count) < 20 or float(row.lift) <= 1.02:
            continue
        parts = [part.strip() for part in str(row.antecedent).split(" & ")]
        score = abs(behavior_weights[consequent]) * float(row.confidence) * np.log1p(float(row.support_count)) * np.log(float(row.lift))
        themes = [int(part.split("=", 1)[1]) for part in parts if part.startswith("theme=")]
        tags = [part.split("=", 1)[1] for part in parts if part.startswith("tag=")]
        groups = [part for part in parts if part.startswith(("gender=", "age_group=", "city_level="))]

        if consequent == "behavior=hate":
            for theme in themes:
                features["theme_hate_risk"][theme] += score
            continue

        for theme in themes:
            features["theme_rule_quality"][theme] += score
            for group in groups:
                dimension, value = group.split("=", 1)
                features["group_theme_rule_quality"][(dimension, value, theme)] += score
        for tag in tags:
            features["tag_rule_quality"][tag] += score

    return {key: normalize_mapping(dict(value)) for key, value in features.items()}


def build_group_theme_success(train: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    global_success = (
        train.groupby("root_id")
        .agg(
            exposure_count=("pid", "size"),
            avg_interaction_score=("interaction_score", "mean"),
            effective_view_rate=("effective_view", "mean"),
            like_rate=("cvm_like", "mean"),
            collect_rate=("collect", "mean"),
            forward_rate=("forward", "mean"),
            hate_rate=("hate", "mean"),
        )
        .reset_index()
    )
    global_success["success"] = (
        0.40 * global_success["avg_interaction_score"]
        + 0.25 * global_success["effective_view_rate"]
        + 0.12 * global_success["like_rate"]
        + 0.15 * global_success["collect_rate"]
        + 0.12 * global_success["forward_rate"]
        - 0.20 * global_success["hate_rate"]
    ).clip(lower=0)
    global_map = dict(zip(global_success["root_id"], global_success["success"]))

    result: dict[tuple[str, str, int], float] = {}
    alpha = 50
    for dimension in ["gender", "age_group", "fre_city_level"]:
        grouped = (
            train.groupby([dimension, "root_id"], dropna=False)
            .agg(
                exposure_count=("pid", "size"),
                avg_interaction_score=("interaction_score", "mean"),
                effective_view_rate=("effective_view", "mean"),
                like_rate=("cvm_like", "mean"),
                collect_rate=("collect", "mean"),
                forward_rate=("forward", "mean"),
                hate_rate=("hate", "mean"),
            )
            .reset_index()
        )
        grouped["raw_success"] = (
            0.40 * grouped["avg_interaction_score"]
            + 0.25 * grouped["effective_view_rate"]
            + 0.12 * grouped["like_rate"]
            + 0.15 * grouped["collect_rate"]
            + 0.12 * grouped["forward_rate"]
            - 0.20 * grouped["hate_rate"]
        ).clip(lower=0)
        for row in grouped.itertuples(index=False):
            root_id = int(row.root_id)
            prior = global_map.get(root_id, 0.0)
            smoothed = (row.exposure_count * row.raw_success + alpha * prior) / (row.exposure_count + alpha)
            result[(dimension, str(getattr(row, dimension)), root_id)] = smoothed

    return normalize_mapping(result)


def build_theme_similarity(preference: pd.DataFrame) -> dict[tuple[int, int], float]:
    theme_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    for _, group in preference.sort_values("preference_score", ascending=False).groupby("user_id"):
        top_themes = [int(theme) for theme in group.head(5)["root_id"].tolist()]
        for theme in top_themes:
            theme_counts[theme] += 1
        for left, right in combinations(sorted(set(top_themes)), 2):
            pair_counts[(left, right)] += 1

    similarity: dict[tuple[int, int], float] = {}
    for (left, right), count in pair_counts.items():
        denom = np.sqrt(theme_counts[left] * theme_counts[right])
        if denom > 0:
            value = count / denom
            similarity[(left, right)] = value
            similarity[(right, left)] = value
    return similarity


def build_user_positive_tags(train: pd.DataFrame) -> dict[int, set[str]]:
    positive = train[(train["effective_view"] | train["cvm_like"] | train["collect"] | train["comment"] | train["forward"]) & (~train["hate"])]
    tags: dict[int, set[str]] = defaultdict(set)
    for row in positive[["user_id", "tag_name"]].itertuples(index=False):
        tags[int(row.user_id)].update(split_tags(row.tag_name))
    return tags


def recommend(train: pd.DataFrame, test: pd.DataFrame, preference: pd.DataFrame, rules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = train.drop_duplicates("pid")[["pid", "root_id", "root_name_cn", "title", "duration", "author_fans_count", "tag_name"]]
    popularity = (
        train.groupby("pid")
        .agg(popularity_score=("interaction_score", "sum"), train_exposure=("pid", "size"))
        .reset_index()
    )
    catalog = catalog.merge(popularity, on="pid", how="left").fillna({"popularity_score": 0, "train_exposure": 0})
    catalog["item_popularity_score"] = np.log1p(catalog["popularity_score"])
    max_item_popularity = catalog["item_popularity_score"].max() or 1
    catalog["item_popularity_score"] = catalog["item_popularity_score"] / max_item_popularity

    seen = train.groupby("user_id")["pid"].apply(set).to_dict()
    user_preferences = preference.groupby("user_id")
    rule_features = build_rule_boosts(rules)
    group_theme_success = build_group_theme_success(train)
    theme_similarity = build_theme_similarity(preference)
    user_positive_tags = build_user_positive_tags(train)
    user_hate = train.groupby(["user_id", "root_id"])["hate"].mean().to_dict()

    baseline_rows = []
    enhanced_rows = []
    for user_id in sorted(train["user_id"].unique()):
        if user_id in user_preferences.groups:
            pref = user_preferences.get_group(user_id)[["root_id", "preference_score"]].copy()
            max_pref = pref["preference_score"].max() or 1
            pref["personal_theme_score"] = pref["preference_score"] / max_pref
            pref_map = dict(zip(pref["root_id"].astype(int), pref["personal_theme_score"]))
        else:
            pref = pd.DataFrame(columns=["root_id", "personal_theme_score"])
            pref_map = {}

        candidates = catalog[~catalog["pid"].isin(seen.get(user_id, set()))].copy()
        if candidates.empty:
            continue
        candidates = candidates.merge(pref[["root_id", "personal_theme_score"]], on="root_id", how="left").fillna({"personal_theme_score": 0})

        user_row = train[train["user_id"] == user_id].iloc[-1]
        gender = str(user_row.gender)
        age_group = str(user_row.age_group)
        city_level = str(user_row.fre_city_level)
        positive_tags = user_positive_tags.get(int(user_id), set())

        def score_group_theme(root_id: int) -> float:
            return (
                0.40 * group_theme_success.get(("gender", gender, root_id), 0)
                + 0.35 * group_theme_success.get(("age_group", age_group, root_id), 0)
                + 0.25 * group_theme_success.get(("fre_city_level", city_level, root_id), 0)
            )

        def score_similarity(root_id: int) -> float:
            if not pref_map:
                return 0.0
            return min(sum(score * theme_similarity.get((int(theme), root_id), 0) for theme, score in pref_map.items()), 1.0)

        def score_tag_match(tag_name: object) -> float:
            candidate_tags = set(split_tags(tag_name))
            if not candidate_tags or not positive_tags:
                return 0.0
            return min(sum(rule_features["tag_rule_quality"].get(tag, 0) for tag in candidate_tags & positive_tags), 1.0)

        candidates["group_theme_success"] = candidates["root_id"].astype(int).map(score_group_theme)
        candidates["theme_rule_quality"] = candidates["root_id"].astype(int).map(rule_features["theme_rule_quality"]).fillna(0)
        candidates["similar_theme_score"] = candidates["root_id"].astype(int).map(score_similarity)
        candidates["tag_match_score"] = candidates["tag_name"].map(score_tag_match)
        candidates["theme_hate_risk"] = candidates["root_id"].astype(int).map(rule_features["theme_hate_risk"]).fillna(0)
        candidates["user_hate_risk"] = candidates["root_id"].astype(int).map(lambda root_id: user_hate.get((user_id, root_id), 0))
        candidates["hate_risk_penalty"] = 0.55 * candidates["user_hate_risk"] + 0.45 * candidates["theme_hate_risk"]

        candidates["baseline_score"] = 0.82 * candidates["personal_theme_score"] + 0.18 * candidates["item_popularity_score"]
        candidates["rule_boost"] = (
            0.35 * candidates["group_theme_success"]
            + 0.30 * candidates["theme_rule_quality"]
            + 0.20 * candidates["similar_theme_score"]
            + 0.15 * candidates["tag_match_score"]
        )
        candidates["enhanced_score"] = (
            0.58 * candidates["baseline_score"]
            + 0.16 * candidates["group_theme_success"]
            + 0.11 * candidates["theme_rule_quality"]
            + 0.08 * candidates["similar_theme_score"]
            + 0.09 * candidates["item_popularity_score"]
            + 0.05 * candidates["tag_match_score"]
            - 0.07 * candidates["hate_risk_penalty"]
        )

        base_top = candidates.sort_values(["baseline_score", "item_popularity_score"], ascending=False).head(TOP_K)
        enhanced_top = candidates.sort_values(["enhanced_score", "item_popularity_score"], ascending=False).head(TOP_K)
        for rank, row in enumerate(base_top.itertuples(index=False), start=1):
            baseline_rows.append(
                {
                    "user_id": user_id,
                    "rank": rank,
                    "pid": row.pid,
                    "root_id": row.root_id,
                    "root_name_cn": row.root_name_cn,
                    "score": row.baseline_score,
                    "title": row.title,
                }
            )
        for rank, row in enumerate(enhanced_top.itertuples(index=False), start=1):
            enhanced_rows.append(
                {
                    "user_id": user_id,
                    "rank": rank,
                    "pid": row.pid,
                    "root_id": row.root_id,
                    "root_name_cn": row.root_name_cn,
                    "score": row.enhanced_score,
                    "rule_boost": row.rule_boost,
                    "title": row.title,
                }
            )

    baseline = pd.DataFrame(baseline_rows)
    enhanced = pd.DataFrame(enhanced_rows)
    baseline.to_csv(REC_DIR / "baseline_recommendations.csv", index=False)
    enhanced.to_csv(REC_DIR / "rule_enhanced_recommendations.csv", index=False)
    evaluation = evaluate_recommendations(test, baseline, enhanced)
    evaluation.to_csv(TABLE_DIR / "evaluation_summary.csv", index=False)
    return baseline, enhanced, evaluation


def evaluate_recommendations(test: pd.DataFrame, baseline: pd.DataFrame, enhanced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive_test = test[(test["effective_view"] | test["cvm_like"] | test["collect"] | test["comment"] | test["forward"]) & (~test["hate"])]
    test_items = positive_test.groupby("user_id")["pid"].apply(set).to_dict() if not positive_test.empty else {}
    test_themes = positive_test.groupby("user_id")["root_id"].apply(set).to_dict() if not positive_test.empty else {}

    for name, recs in [("baseline", baseline), ("rule_enhanced", enhanced)]:
        hit_values = []
        precision_values = []
        recall_values = []
        theme_match_values = []
        for user_id, true_items in test_items.items():
            group = recs[recs["user_id"] == user_id]
            if group.empty:
                continue
            recommended_items = set(group["pid"])
            recommended_themes = set(group["root_id"])
            true_themes = test_themes.get(user_id, set())
            hits = recommended_items & true_items
            hit_values.append(1.0 if hits else 0.0)
            precision_values.append(len(hits) / max(len(recommended_items), 1))
            recall_values.append(len(hits) / len(true_items))
            theme_match_values.append(len(recommended_themes & true_themes) / max(len(recommended_themes), 1))
        rows.append(
            {
                "method": name,
                "user_count": len(test_items),
                "HitRate@10": np.mean(hit_values) if hit_values else 0,
                "Precision@10": np.mean(precision_values) if precision_values else 0,
                "Recall@10": np.mean(recall_values) if recall_values else 0,
                "ThemeMatch@10": np.mean(theme_match_values) if theme_match_values else 0,
            }
        )
    return pd.DataFrame(rows)


def configure_plot_fonts() -> None:
    preferred_fonts = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC"]
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_outputs(theme_summary: pd.DataFrame, group_summary: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    configure_plot_fonts()

    top_watch = theme_summary.sort_values("avg_watch_time", ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    plt.bar(top_watch["root_name_cn"].astype(str), top_watch["avg_watch_time"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average watch time")
    plt.title("Top themes by average watch time")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_theme_watch_time.png", dpi=160)
    plt.close()

    top_interaction = theme_summary.sort_values("exposure_count", ascending=False).head(10)
    x = np.arange(len(top_interaction))
    width = 0.25
    plt.figure(figsize=(11, 5))
    plt.bar(x - width, top_interaction["like_rate"], width, label="like")
    plt.bar(x, top_interaction["collect_rate"], width, label="collect")
    plt.bar(x + width, top_interaction["forward_rate"], width, label="forward")
    plt.xticks(x, top_interaction["root_name_cn"].astype(str), rotation=45, ha="right")
    plt.ylabel("Rate")
    plt.title("Interaction rates of high-exposure themes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "theme_interaction_rates.png", dpi=160)
    plt.close()

    gender_summary = group_summary[group_summary["group_dimension"] == "gender"].copy()
    pivot = gender_summary.pivot_table(index="group_value", columns="root_name_cn", values="avg_preference_score", aggfunc="mean", fill_value=0)
    top_cols = theme_summary.head(12)["root_name_cn"].tolist()
    pivot = pivot[[c for c in top_cols if c in pivot.columns]]
    if not pivot.empty:
        plt.figure(figsize=(12, 4))
        plt.imshow(pivot.values, aspect="auto")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.colorbar(label="Preference score")
        plt.title("Gender-theme preference heatmap")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "gender_theme_preference_heatmap.png", dpi=160)
        plt.close()

    if not evaluation.empty:
        metrics = ["HitRate@10", "Precision@10", "Recall@10", "ThemeMatch@10"]
        eval_plot = evaluation.set_index("method")[metrics]
        eval_plot.T.plot(kind="bar", figsize=(9, 5))
        plt.ylabel("Score")
        plt.title("Recommendation evaluation")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "recommendation_evaluation.png", dpi=160)
        plt.close()


def main() -> None:
    ensure_dirs()
    cleaned = clean_interactions()
    save_clean_outputs(cleaned)
    train, test = split_train_test(cleaned)
    theme_summary = summarize_theme_behavior(train)
    group_summary = summarize_user_groups(train)
    preference = build_user_theme_preference(train)
    rules = mine_rules(train)
    _, _, evaluation = recommend(train, test, preference, rules)
    plot_outputs(theme_summary, group_summary, evaluation)

    print("Local pipeline finished")
    print(f"clean_interactions={len(cleaned):,}")
    print(f"train_interactions={len(train):,}")
    print(f"test_interactions={len(test):,}")
    print(f"themes={theme_summary['root_id'].nunique():,}")
    print(f"rules={len(rules):,}")
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
