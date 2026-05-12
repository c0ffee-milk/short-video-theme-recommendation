#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data_raw" / "shortvideo_tiny"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main() -> None:
    spark = (
        SparkSession.builder.appName("ShortVideoThemePreference")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    interactions = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW_DIR / "interaction_sampled.csv"))
    categories = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW_DIR / "categories_cn_en.csv"))

    root_map = (
        categories.select(
            F.col("category_id").alias("root_id"),
            F.col("category_name_cn").alias("root_name_cn"),
            F.col("category_name_en").alias("root_name_en"),
        )
        .dropDuplicates(["root_id"])
    )

    bool_cols = ["cvm_like", "click", "comment", "follow", "collect", "forward", "hate"]
    for col in bool_cols:
        interactions = interactions.withColumn(col, F.col(col).cast("boolean"))

    cleaned = (
        interactions.withColumn("effective_view", F.col("watch_time") > F.lit(3))
        .withColumn(
            "watch_ratio_raw",
            F.when(F.col("duration") > 0, F.col("watch_time") / F.col("duration")).otherwise(F.lit(0.0)),
        )
        .withColumn("watch_ratio", F.least(F.greatest(F.col("watch_ratio_raw"), F.lit(0.0)), F.lit(1.0)))
        .join(root_map, on="root_id", how="left")
    )

    deduped = cleaned.groupBy("user_id", "pid", "exposed_time").agg(
        F.first("root_id", ignorenulls=True).alias("root_id"),
        F.first("root_name_cn", ignorenulls=True).alias("root_name_cn"),
        F.first("gender", ignorenulls=True).alias("gender"),
        F.first("age", ignorenulls=True).alias("age"),
        F.first("fre_city_level", ignorenulls=True).alias("fre_city_level"),
        F.first("watch_time", ignorenulls=True).alias("watch_time"),
        F.first("duration", ignorenulls=True).alias("duration"),
        F.first("watch_ratio", ignorenulls=True).alias("watch_ratio"),
        F.max(F.col("effective_view").cast("int")).cast("boolean").alias("effective_view"),
        *[F.max(F.col(c).cast("int")).cast("boolean").alias(c) for c in bool_cols],
        F.concat_ws("|", F.collect_set("tag_name")).alias("tag_name"),
    )

    theme_summary = deduped.groupBy("root_id", "root_name_cn").agg(
        F.count("pid").alias("exposure_count"),
        F.countDistinct("user_id").alias("user_count"),
        F.countDistinct("pid").alias("video_count"),
        F.avg("watch_time").alias("avg_watch_time"),
        F.avg("watch_ratio").alias("avg_watch_ratio"),
        F.avg(F.col("effective_view").cast("double")).alias("effective_view_rate"),
        F.avg(F.col("cvm_like").cast("double")).alias("like_rate"),
        F.avg(F.col("collect").cast("double")).alias("collect_rate"),
        F.avg(F.col("forward").cast("double")).alias("forward_rate"),
    )
    theme_summary.write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "tables" / "spark_theme_behavior_summary"))

    transactions = deduped.select(
        "user_id",
        "pid",
        F.array_distinct(
            F.array_remove(
                F.array(
                    F.concat(F.lit("theme="), F.col("root_id").cast("string")),
                    F.concat(F.lit("gender="), F.coalesce(F.col("gender").cast("string"), F.lit("unknown"))),
                    F.concat(F.lit("city_level="), F.coalesce(F.col("fre_city_level").cast("string"), F.lit("unknown"))),
                    F.when(F.col("effective_view"), F.lit("behavior=effective_view")),
                    F.when(F.col("cvm_like"), F.lit("behavior=like")),
                    F.when(F.col("collect"), F.lit("behavior=collect")),
                    F.when(F.col("comment"), F.lit("behavior=comment")),
                    F.when(F.col("forward"), F.lit("behavior=forward")),
                    F.when(F.col("hate"), F.lit("behavior=hate")),
                ),
                None,
            )
        ).alias("items"),
    )

    model = FPGrowth(itemsCol="items", minSupport=0.001, minConfidence=0.05).fit(transactions)
    model.associationRules.write.mode("overwrite").option("header", True).csv(str(OUTPUT_DIR / "rules" / "spark_fpgrowth_rules"))

    spark.stop()


if __name__ == "__main__":
    main()
