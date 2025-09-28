import os
import pandas as pd
import yaml
from Labeler import label_data  # Labeler.py
from SentimentStats import sentiment_statistics  # 新写的统计模块
from keyword_cluster import generate_keyword_stats  # 相似度聚类,生成关键字典模块
from ai_summary import ai_summary_pipeline  # ai总结模块


# 加载yaml配置函数
def load_config(yaml_path):
    """
    :param yaml_path: yaml文件路径
    :return: config 配置
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(csv_path=None,
                 labeler_config=None,
                 analysis_config=None,
                 keyword_cluster_config=None,
                 batch_size=None,
                 save_csv=False,
                 output_dir=None,
                 project_dir=None,
                 ):
    """
    主流程 pipeline
    :param csv_path: 原始聊天数据路径
    :param labeler_config: 标注模块整个配置信息
    :param analysis_config: 分析模块整个配置信息
    :param keyword_cluster_config: 聚合模块整个配置信息
    :param batch_size: 批量预测大小
    :param save_csv: 是否保存结果
    :param output_dir: 保存路径
    :param project_dir: 项目根目录
    :return: (df_labeled, global_dist, player_dist)
    """
    # 调用 Labeler 进行标注
    df_labeled = label_data(csv_path,
                            batch_size=batch_size,
                            labeler_config=labeler_config,
                            project_dir=project_dir,
                            )

    # 读取目标标签列
    label_col = analysis_config.get('label_col', '')
    if not label_col:
        raise ValueError("analysis_config 缺少 label_col")

    # 读取文本列
    text_column = labeler_config.get('data', {}).get('text_column')
    if not text_column:
        raise ValueError("labeler_config 缺少 data.text_column")

    # 读取id列
    id_column = labeler_config.get('data', {}).get('id_column')
    if not id_column:
        raise ValueError("labeler_config 缺少 data.id_column")

    # 调用 SentimentsStats 统计情感
    grouped = sentiment_statistics(df_labeled, label_col, text_column, id_column)

    # 打印结果
    print(f"=== 按 {label_col} 的情感分布 ===")
    print(grouped)

    # 可选保存
    if save_csv:
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        os.makedirs(output_dir, exist_ok=True)
        df_labeled.to_csv(os.path.join(output_dir, "labeled_chats.csv"), index=False)
        grouped.to_csv(os.path.join(output_dir, f"sentiment_{label_col}.csv"), index=False)

    type_value = analysis_config['type_value']
    # 1. 聚合关键词得到 df_result
    df_result = generate_keyword_stats(df_labeled,
                                       label_col=label_col,
                                       type_value=type_value,
                                       text_column=text_column,
                                       id_column=id_column,
                                       keyword_cluster_config=keyword_cluster_config,
                                       project_dir=project_dir,
                                      )
    # print(f"returned df_result: {type(df_result)}")

    # 打印 df_result
    print(df_result.to_csv(index=False, sep='\t'))

    # 2. 调用 AI 生成总结
    response = ai_summary_pipeline(df_result,
                                   analysis_config=analysis_config,
                                   )

    print(response)


if __name__ == "__main__":
    # 获取当前项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))

    # 读取pipeline.yaml路径
    yaml_path = os.path.join(project_dir, "Sentiment_Analysis_Automation", "configs", "pipeline.yaml")
    config = load_config(yaml_path)

    # 主函数
    run_pipeline(csv_path=os.path.join(project_dir, config['dataset']['csv_path']),
                 labeler_config=config['labeler'],
                 analysis_config=config['analysis'],
                 keyword_cluster_config=config['keyword_cluster'],
                 batch_size=config['pipeline']['batch_size'],
                 project_dir=project_dir,
                 )
