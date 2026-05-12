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


def build_rule_boosts(rules: pd.DataFrame) -> dict[str, float]:
    boosts: dict[str, float] = defaultdict(float)
    if rules.empty:
        return boosts
    positive_behaviors = {"behavior=effective_view", "behavior=like", "behavior=collect", "behavior=comment", "behavior=forward"}
    for row in rules.itertuples(index=False):
        if row.consequent not in positive_behaviors:
            continue
        for part in str(row.antecedent).split(" & "):
            if part.startswith("theme="):
                theme = part.split("=", 1)[1]
                boosts[theme] += float(row.confidence) * max(float(row.lift), 0)
    if boosts:
        max_boost = max(boosts.values())
        boosts = {theme: value / max_boost for theme, value in boosts.items()}
    return boosts


def recommend(cleaned: pd.DataFrame, preference: pd.DataFrame, rules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = split_train_test(cleaned)
    catalog = cleaned.drop_duplicates("pid")[["pid", "root_id", "root_name_cn", "title", "duration", "author_fans_count"]]
    popularity = (
        train.groupby("pid")
        .agg(popularity_score=("interaction_score", "sum"), train_exposure=("pid", "size"))
        .reset_index()
    )
    catalog = catalog.merge(popularity, on="pid", how="left").fillna({"popularity_score": 0, "train_exposure": 0})
    seen = train.groupby("user_id")["pid"].apply(set).to_dict()
    user_preferences = preference.groupby("user_id")
    rule_boosts = build_rule_boosts(rules)

    baseline_rows = []
    enhanced_rows = []
    for user_id in sorted(cleaned["user_id"].unique()):
        if user_id in user_preferences.groups:
            pref = user_preferences.get_group(user_id)[["root_id", "preference_score"]].copy()
        else:
            pref = pd.DataFrame(columns=["root_id", "preference_score"])
        candidates = catalog[~catalog["pid"].isin(seen.get(user_id, set()))].copy()
        if candidates.empty:
            continue
        candidates = candidates.merge(pref, on="root_id", how="left").fillna({"preference_score": 0})
        max_pop = candidates["popularity_score"].max() or 1
        candidates["baseline_score"] = candidates["preference_score"] + 0.15 * candidates["popularity_score"] / max_pop
        candidates["rule_boost"] = candidates["root_id"].astype(str).map(rule_boosts).fillna(0)
        candidates["enhanced_score"] = candidates["baseline_score"] * (1 + 0.25 * candidates["rule_boost"])

        base_top = candidates.sort_values(["baseline_score", "popularity_score"], ascending=False).head(TOP_K)
        enhanced_top = candidates.sort_values(["enhanced_score", "popularity_score"], ascending=False).head(TOP_K)
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
    evaluation = evaluate_recommendations(test, preference, baseline, enhanced)
    evaluation.to_csv(TABLE_DIR / "evaluation_summary.csv", index=False)
    return baseline, enhanced, evaluation


def evaluate_recommendations(test: pd.DataFrame, preference: pd.DataFrame, baseline: pd.DataFrame, enhanced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    high_pref_theme = preference.sort_values("preference_score", ascending=False).groupby("user_id")["root_id"].apply(lambda s: set(s.head(3))).to_dict()
    test_items = test.groupby("user_id")["pid"].apply(set).to_dict() if not test.empty else {}
    test_themes = test.groupby("user_id")["root_id"].apply(set).to_dict() if not test.empty else {}

    for name, recs in [("baseline", baseline), ("rule_enhanced", enhanced)]:
        hit_values = []
        precision_values = []
        recall_values = []
        theme_match_values = []
        for user_id, group in recs.groupby("user_id"):
            recommended_items = set(group["pid"])
            recommended_themes = set(group["root_id"])
            true_items = test_items.get(user_id, set())
            true_themes = test_themes.get(user_id, set())
            preferred_themes = high_pref_theme.get(user_id, set())
            if true_items:
                hits = recommended_items & true_items
                hit_values.append(1.0 if hits else 0.0)
                precision_values.append(len(hits) / max(len(recommended_items), 1))
                recall_values.append(len(hits) / len(true_items))
            if preferred_themes:
                theme_match_values.append(len(recommended_themes & (preferred_themes | true_themes)) / max(len(recommended_themes), 1))
        rows.append(
            {
                "method": name,
                "user_count": recs["user_id"].nunique() if not recs.empty else 0,
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
    theme_summary = summarize_theme_behavior(cleaned)
    group_summary = summarize_user_groups(cleaned)
    preference = build_user_theme_preference(cleaned)
    rules = mine_rules(cleaned)
    _, _, evaluation = recommend(cleaned, preference, rules)
    plot_outputs(theme_summary, group_summary, evaluation)

    print("Local pipeline finished")
    print(f"clean_interactions={len(cleaned):,}")
    print(f"themes={theme_summary['root_id'].nunique():,}")
    print(f"rules={len(rules):,}")
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
