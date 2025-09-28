import pandas as pd
from logger_utils import log_time


def sentiment_statistics(df: pd.DataFrame, label_col: str, text_column: str, id_column: str) -> pd.DataFrame:
    """
    根据指定的标签列统计情感分布(数量，占比，人数)
    :param df: 已经包含标签的DataFrame
    :param label_col: 用哪个列的标签进行统计，例如 'model_tag2'
    :param text_column: 文本列名字
    :param id_column: 用户id列名字
    :return: 统计结果 DataFrame
    """
    start_time = log_time(None, "Sentiment statistics")

    if label_col not in df.columns:
        raise ValueError(f"指定的列 {label_col} 不存在于 DataFrame 中!")

    # 聊天总数
    chat_num_total = df[text_column].count()

    # 总人数
    player_num_total = df[id_column].nunique()

    # 按指定标签列分组
    grouped = df.groupby(label_col).agg({
        text_column: 'count',  # 每个情感的信息数量
        id_column: 'nunique'    # 每个情感的唯一人数
    }).reset_index()

    # 计算占比
    grouped['数据量占比'] = round(grouped[text_column] / chat_num_total, 4)
    grouped['人数占比'] = round(grouped[id_column] / player_num_total, 4)

    # 调整列顺序
    grouped = grouped[[label_col, text_column, '数据量占比', id_column, '人数占比']]

    log_time(start_time, "Sentiment_statistics")

    return grouped