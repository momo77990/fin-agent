# 将研报 / 财报 / 招股书等 PDF 文件放到本目录，Agent 的 `search_filings` 工具
# 会自动索引（按文件 mtime 缓存）。
#
# 例如：
#   data/nvidia_2024_annual_report.pdf
#   data/moutai_2023_q4.pdf
#
# 注意：嵌入模型由「API 设置」中的 Embedding Model 字段决定，
# 默认 text-embedding-3-small。
