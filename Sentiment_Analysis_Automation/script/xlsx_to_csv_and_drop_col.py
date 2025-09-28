import pandas as pd
import os


def xlsx_to_csv_remove_columns(input_xlsx, output_csv, columns_to_remove):
    # 读取xlsx
    df = pd.read_excel(input_xlsx, engine="openpyxl")

    # 删除列
    df.drop(columns=columns_to_remove, inplace=True, errors="ignore")

    # 保存为csv
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"文件已保存到： {output_csv}")


if __name__ == "__main__":
    current_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_root))

    input_file = os.path.join(project_root, "Demand_analysis_data", "yuanxing", "Dataset", "tuanzhan.xlsx")
    output_file = os.path.join(project_root, "Demand_analysis_data", "yuanxing", "Dataset", "tuanzhan.csv")
    cols_to_remove = ["server_name"]

    xlsx_to_csv_remove_columns(input_xlsx=input_file, output_csv=output_file, columns_to_remove=cols_to_remove)
