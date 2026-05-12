CREATE DATABASE IF NOT EXISTS shortvideo_project;
USE shortvideo_project;

DROP TABLE IF EXISTS interaction_raw;
CREATE EXTERNAL TABLE interaction_raw (
  user_id BIGINT,
  pid BIGINT,
  author_id BIGINT,
  category_id INT,
  category_level INT,
  parent_id INT,
  root_id INT,
  exposed_time BIGINT,
  author_fans_count BIGINT,
  watch_time DOUBLE,
  duration DOUBLE,
  cvm_like BOOLEAN,
  click BOOLEAN,
  comment BOOLEAN,
  follow BOOLEAN,
  collect BOOLEAN,
  forward BOOLEAN,
  hate BOOLEAN,
  tag_name STRING,
  title STRING,
  p_hour INT,
  p_date INT,
  gender STRING,
  age INT,
  mod_price DOUBLE,
  fre_city STRING,
  fre_community_type STRING,
  fre_city_level STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\'
)
STORED AS TEXTFILE
LOCATION '/shortvideo/raw/interaction_sampled'
TBLPROPERTIES ('skip.header.line.count'='1');

DROP TABLE IF EXISTS category_map;
CREATE EXTERNAL TABLE category_map (
  category_level INT,
  category_id INT,
  category_name_cn STRING,
  parent_id INT,
  root_id INT,
  category_name_en STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\'
)
STORED AS TEXTFILE
LOCATION '/shortvideo/raw/categories_cn_en'
TBLPROPERTIES ('skip.header.line.count'='1');

CREATE OR REPLACE VIEW interaction_clean_view AS
SELECT
  user_id,
  pid,
  author_id,
  category_id,
  category_level,
  parent_id,
  root_id,
  exposed_time,
  author_fans_count,
  watch_time,
  duration,
  CASE WHEN watch_time > 3 THEN true ELSE false END AS effective_view,
  CASE WHEN duration > 0 THEN LEAST(GREATEST(watch_time / duration, 0), 1) ELSE 0 END AS watch_ratio,
  cvm_like,
  click,
  comment,
  follow,
  collect,
  forward,
  hate,
  tag_name,
  title,
  p_hour,
  p_date,
  gender,
  age,
  CASE
    WHEN age <= 18 THEN '<=18'
    WHEN age <= 25 THEN '19-25'
    WHEN age <= 35 THEN '26-35'
    WHEN age <= 45 THEN '36-45'
    WHEN age <= 60 THEN '46-60'
    WHEN age > 60 THEN '60+'
    ELSE 'unknown'
  END AS age_group,
  mod_price,
  fre_city,
  fre_community_type,
  fre_city_level
FROM interaction_raw;

DROP TABLE IF EXISTS theme_behavior_summary;
CREATE TABLE theme_behavior_summary AS
SELECT
  i.root_id,
  COALESCE(m.category_name_cn, CONCAT('主题', CAST(i.root_id AS STRING))) AS root_name_cn,
  COUNT(*) AS exposure_count,
  COUNT(DISTINCT i.user_id) AS user_count,
  COUNT(DISTINCT i.pid) AS video_count,
  AVG(i.watch_time) AS avg_watch_time,
  AVG(i.watch_ratio) AS avg_watch_ratio,
  AVG(CASE WHEN i.effective_view THEN 1 ELSE 0 END) AS effective_view_rate,
  AVG(CASE WHEN i.cvm_like THEN 1 ELSE 0 END) AS like_rate,
  AVG(CASE WHEN i.collect THEN 1 ELSE 0 END) AS collect_rate,
  AVG(CASE WHEN i.comment THEN 1 ELSE 0 END) AS comment_rate,
  AVG(CASE WHEN i.forward THEN 1 ELSE 0 END) AS forward_rate,
  AVG(CASE WHEN i.hate THEN 1 ELSE 0 END) AS hate_rate
FROM interaction_clean_view i
LEFT JOIN category_map m
  ON i.root_id = m.category_id
GROUP BY i.root_id, COALESCE(m.category_name_cn, CONCAT('主题', CAST(i.root_id AS STRING)));

DROP TABLE IF EXISTS user_theme_preference;
CREATE TABLE user_theme_preference AS
SELECT
  user_id,
  root_id,
  COUNT(*) AS exposure_count,
  COUNT(DISTINCT pid) AS video_count,
  AVG(watch_ratio) AS avg_watch_ratio,
  SUM(
    0.45 * watch_ratio
    + 0.20 * CASE WHEN effective_view THEN 1 ELSE 0 END
    + 0.15 * CASE WHEN cvm_like THEN 1 ELSE 0 END
    + 0.10 * CASE WHEN collect THEN 1 ELSE 0 END
    + 0.05 * CASE WHEN comment THEN 1 ELSE 0 END
    + 0.08 * CASE WHEN forward THEN 1 ELSE 0 END
    - 0.20 * CASE WHEN hate THEN 1 ELSE 0 END
  ) AS preference_score
FROM interaction_clean_view
GROUP BY user_id, root_id;

DROP TABLE IF EXISTS user_group_theme_summary;
CREATE TABLE user_group_theme_summary AS
SELECT
  'gender' AS group_dimension,
  gender AS group_value,
  root_id,
  COUNT(*) AS exposure_count,
  AVG(watch_ratio) AS avg_watch_ratio,
  AVG(CASE WHEN cvm_like THEN 1 ELSE 0 END) AS like_rate,
  AVG(CASE WHEN collect THEN 1 ELSE 0 END) AS collect_rate,
  AVG(CASE WHEN forward THEN 1 ELSE 0 END) AS forward_rate
FROM interaction_clean_view
GROUP BY gender, root_id
UNION ALL
SELECT
  'age_group' AS group_dimension,
  age_group AS group_value,
  root_id,
  COUNT(*) AS exposure_count,
  AVG(watch_ratio) AS avg_watch_ratio,
  AVG(CASE WHEN cvm_like THEN 1 ELSE 0 END) AS like_rate,
  AVG(CASE WHEN collect THEN 1 ELSE 0 END) AS collect_rate,
  AVG(CASE WHEN forward THEN 1 ELSE 0 END) AS forward_rate
FROM interaction_clean_view
GROUP BY age_group, root_id
UNION ALL
SELECT
  'fre_city_level' AS group_dimension,
  fre_city_level AS group_value,
  root_id,
  COUNT(*) AS exposure_count,
  AVG(watch_ratio) AS avg_watch_ratio,
  AVG(CASE WHEN cvm_like THEN 1 ELSE 0 END) AS like_rate,
  AVG(CASE WHEN collect THEN 1 ELSE 0 END) AS collect_rate,
  AVG(CASE WHEN forward THEN 1 ELSE 0 END) AS forward_rate
FROM interaction_clean_view
GROUP BY fre_city_level, root_id;
