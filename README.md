# 短视频主题偏好分析与推荐策略优化

Short Video Theme Preference Analysis and Recommendation Strategy Optimization

本项目是“大数据处理”课程期末项目，基于 Tsinghua ShortVideo Dataset 的抽样数据，围绕短视频平台中的用户行为、视频主题和用户属性，完成主题偏好分析、关联规则挖掘和推荐策略优化实验。

## 项目目标

项目关注的问题是：如何从短视频用户的观看、点赞、收藏、转发等行为中识别用户对不同视频主题的偏好，并将主题偏好和群体行为规则用于推荐策略优化。

实验流程包括：

1. 数据清洗与字段构建
2. 主题行为统计分析
3. 用户-主题偏好画像构建
4. 主题与行为之间的关联规则挖掘
5. 基础推荐与规则增强推荐对比
6. 离线评估与可视化展示

## 数据说明

原始数据来自 Tsinghua ShortVideo Dataset。完整原始视频数据体量很大，本项目只在本地使用小规模替代版完成实验。

本仓库不上传原始视频、原始行为大表或下载日志，`data_raw/` 已被 `.gitignore` 排除。仓库中只保留：

- 清洗后的轻量数据表
- 统计分析结果
- 关联规则结果
- 推荐结果
- 可视化图表
- 本地、Spark 和 Hive 分析代码

如需重新运行完整流水线，需要自行将原始数据放到：

```text
data_raw/shortvideo_tiny/
  interaction_sampled.csv
  categories_cn_en.csv
```

## 目录结构

```text
.
├── README.md
├── .gitignore
├── scripts/
│   ├── run_local_pipeline.py       # 本地一键实验流程
│   └── check_outputs.py            # 输出文件校验与摘要打印
├── spark/
│   └── theme_preference_spark.py   # Spark 版本主题统计与 FP-Growth
├── hive/
│   └── shortvideo_analysis.sql     # Hive 建表、清洗视图和聚合 SQL
├── data_clean/
│   ├── interactions_clean.csv      # 清洗后的用户-视频交互数据
│   ├── user_profile.csv            # 用户画像基础表
│   └── video_catalog.csv           # 视频目录基础表
└── output/
    ├── tables/                     # 主题统计、用户偏好、评估结果
    ├── rules/                      # 关联规则结果
    ├── recommendations/            # 推荐结果
    └── figures/                    # 可视化图表
```

## 环境依赖

本地流程主要依赖：

- Python 3.9+
- pandas
- numpy
- matplotlib

Spark 版本需要：

- PySpark 或可用的 `spark-submit`

Hive SQL 需要：

- Hive 或兼容 Hive SQL 的离线数仓环境

## 本地运行

在仓库根目录运行：

```bash
python3 scripts/run_local_pipeline.py
python3 scripts/check_outputs.py
```

`run_local_pipeline.py` 会完成：

- 读取原始抽样 CSV
- 派生 `effective_view`、`watch_ratio`、年龄段和价格段
- 聚合同一用户-视频曝光的多标签重复行
- 输出清洗表
- 按时间顺序切分训练集和测试集
- 基于训练集生成主题行为统计、用户-主题偏好和关联规则
- 生成基础推荐与规则增强推荐
- 只使用测试集进行离线评估
- 输出评估指标和图表

`check_outputs.py` 会检查关键产物是否存在并打印摘要。

## Spark 运行

如果本机或集群环境已安装 Spark，可运行：

```bash
spark-submit spark/theme_preference_spark.py
```

Spark 脚本会输出：

```text
output/tables/spark_theme_behavior_summary/
output/rules/spark_fpgrowth_rules/
```

## Hive SQL

Hive 脚本位于：

```text
hive/shortvideo_analysis.sql
```

其中包含：

- 原始交互表 DDL
- 类别映射表 DDL
- 清洗视图
- 主题行为聚合表
- 用户-主题偏好表
- 用户分组主题统计表

实际使用时需要根据集群 HDFS 路径调整 SQL 中的 `LOCATION`。

## 推荐策略设计

本项目设置两组推荐方法进行对比：

- **Baseline**：仅使用训练集中的用户历史主题偏好和视频热门度排序，作为朴素个性化推荐方案。
- **Rule-enhanced**：在 baseline 的基础上，引入训练集内可解释的规则增强特征，包括用户分群主题成功率、强关联规则主题质量、相似主题迁移、标签级规则匹配和负反馈风险惩罚。

为避免测试集泄露，用户偏好、关联规则、候选视频、热门度和分群统计均只由训练集构建；测试集只用于计算离线评估指标。

## 当前实验结果摘要

本地流程已在小规模替代数据上完成验证。

```text
clean interactions: 129,483
train interactions: 101,690
test interactions: 27,793
users: 6,654
videos: 31,496
themes: 36
rules: 7,658
baseline recommendations: 66,540
rule-enhanced recommendations: 66,540
```

高曝光主题示例：

| 主题 | 曝光数 | 平均观看比例 | 有效观看率 |
| --- | ---: | ---: | ---: |
| 影视和短剧 | 18,469 | 0.3704 | 0.7283 |
| 搞笑 | 11,623 | 0.5114 | 0.7607 |
| 美食 | 7,298 | 0.4290 | 0.6905 |
| 生活 | 6,298 | 0.5239 | 0.7191 |
| 亲子 | 4,656 | 0.5629 | 0.7448 |

推荐评估结果：

| 方法 | HitRate@10 | Precision@10 | Recall@10 | ThemeMatch@10 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.023013 | 0.002398 | 0.008712 | 0.412038 |
| rule_enhanced | 0.026107 | 0.002649 | 0.009914 | 0.423806 |

相对 baseline，rule-enhanced 在四项指标上均获得提升，说明训练集中的群体主题规律和关联规则可以为个体主题偏好推荐提供有效补充。

## 主要输出

- `output/tables/theme_behavior_summary.csv`：不同主题下的观看、点赞、收藏、转发等行为统计
- `output/tables/user_group_theme_summary.csv`：不同用户群体下的主题偏好统计
- `output/tables/user_theme_preference.csv`：用户-主题偏好画像
- `output/rules/theme_behavior_rules.csv`：主题/标签与行为之间的关联规则
- `output/recommendations/baseline_recommendations.csv`：基础推荐结果
- `output/recommendations/rule_enhanced_recommendations.csv`：规则增强推荐结果
- `output/tables/evaluation_summary.csv`：推荐效果评估
- `output/figures/`：可视化图表

## 图表

仓库中包含以下可视化结果：

- `output/figures/top_theme_watch_time.png`
- `output/figures/theme_interaction_rates.png`
- `output/figures/gender_theme_preference_heatmap.png`
- `output/figures/recommendation_evaluation.png`

## 说明

本仓库用于课程学习与实验展示，不包含原始大体量视频数据。若复现实验，请先根据数据集说明自行准备原始抽样数据，并放置到 `data_raw/shortvideo_tiny/`。该目录不会被 Git 追踪。
