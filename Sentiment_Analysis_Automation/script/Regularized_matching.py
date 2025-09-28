import pandas as pd
import re
import os

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))

input_file = os.path.join(project_dir, "Demand_analysis_data", "fumo", "Dataset", "jiugong_without.csv")
output_file = os.path.join(project_dir, "Demand_analysis_data", "fumo", "Dataset", "new_jiugong_3.csv")

# 读取 csv
df = pd.read_csv(input_file)

# 定义正则模式
# pattern = re.compile(r"(副本|赋能|流派|祝福)")
# pattern = re.compile(r"(玩法|体验|好玩|感觉|有意思|难|玩不懂|玩不明白)")
pattern = re.compile(r"(一样|比)|(?=.*七星)(?=.*九宫)|(?=.*星海)(?=.*九宫)")

# 保留 chat_content 列里包含 “副本” 或者 "九宫" 或者 "赋能" 或者 ”流派“ 或者 "祝福" 的行
mask = df['chat_content'].astype(str).apply(lambda x: bool(pattern.search(x)))

# 保留匹配的行
df_filtered = df[mask]

# 保存结果
df_filtered.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"筛选完成，新文件已保存为 {output_file}")