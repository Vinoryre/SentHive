import pandas as pd
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))

# 输入和输出文件路径
input_file = os.path.join(project_dir, "Demand_analysis_data", "fumo", "Dataset", "jiugong.csv")
output_file = os.path.join(project_dir, "Demand_analysis_data", "fumo", "Dataset", "jiugong_without.csv")

# 读取 csv
df = pd.read_csv(input_file)

# 假设要检查所有列中是否包含"带"
# 如果只需要检查特定列， 比如 'name' 列， 把 df.astype(str).apply(... 里的 df 换成 df['name']
mask = df['chat_content'].astype(str).apply(lambda x: bool(re.search("带|次数", x)))
df_filtered = df[~mask]  # 去掉含有 "带" 的行

# 保存新的 csv
df_filtered.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"过滤完成，新文件已保存为 {output_file}")
