import os

import pandas as pd
import jieba
from textrank4zh import TextRank4Keyword
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
import torch
from logger_utils import log_time, logger


# 文本向量化函数
def get_embeddings(sentences, model_path, device="cuda"):
    """
    用于将文本格式转换为向量
    :param sentences: 需要转换的句子列表
    :param model_path: 使用的向量化模型
    :param device: 模型部署的设备
    :return: 转换后的向量列表
    """
    model = SentenceTransformer(model_path, device=device)
    embeddings = model.encode(sentences, convert_to_tensor=True, device=device)
    return embeddings  # torch.Tensor


# 相似度聚类函数
def cluster_by_similarity_flexible(word_count_dict, model_path, device="cpu", block_size=2000):
    """
    可支持 cpu/cuda 的文本聚类，O(n^2) 时间复杂度
    :param word_count_dict: 输入待处理的字典
    :param model_path: 对句子向量化的模型
    :param device: 向量化模型部署的设备、相似度聚类实现的版本
    :param block_size: GPU分支处理的分块大小
    :return: 聚类后的关键句-人数字典
    """
    start_time = log_time(None, f"Clustering {len(word_count_dict)} items on {device}")

    sorted_items = list(word_count_dict.items())
    sentences = [word for word, _ in sorted_items]
    counts = np.array([count for _, count in sorted_items], dtype=np.int32)
    N = len(sentences)

    if device == "cpu":
        embeddings = get_embeddings(sentences, model_path)
        lengths = np.array([len(w) for w in sentences])
        processed = np.zeros(N, dtype=bool)
        neighbors_list = [[] for _ in range(N)]

        for i in tqdm(range(N), desc="Building sparse neighbors"):
            if processed[i]:
                continue
            vec_i = embeddings[i:i+1]
            sim_row = util.cos_sim(vec_i, embeddings)[0].cpu().numpy()
            threshold_row = np.where((lengths[i] <= 4) | (lengths <= 4), 0.9, 0.85)
            neighbors = np.where(sim_row >= threshold_row)[0]
            neighbors_list[i] = neighbors

        clustered_dict = {}
        for i in tqdm(range(N), desc="Merging clusters"):
            if not processed[i]:
                to_merge = neighbors_list[i][~processed[neighbors_list[i]]]
                clustered_dict[sentences[i]] = counts[to_merge].sum()
                processed[to_merge] = True
    else:
        embeddings = get_embeddings(sentences, model_path, device=device)
        lengths = np.array([len(w) for w in sentences])
        processed = np.zeros(N, dtype=bool)
        clustered_dict = {}

        for i in tqdm(range(0, N, block_size), desc="Merging clusters"):
            i_end = min(i + block_size, N)
            emb_i = embeddings[i:i_end]
            len_i_block = lengths[i:i_end][:, None]
            len_j_all = lengths[None, :]
            threshold_block = np.where((len_i_block <= 4) | (len_j_all <=4 ), 0.9, 0.85)
            threshold_block = torch.tensor(threshold_block, device=device)

            sim_block = util.cos_sim(emb_i, embeddings)
            merge_mask_block = (sim_block >= threshold_block).cpu().numpy()

            for local_idx, global_idx in enumerate(range(i, i_end)):
                if not processed[global_idx]:
                    to_merge = np.where(merge_mask_block[local_idx] & ~processed)[0]
                    clustered_dict[sentences[global_idx]] = counts[to_merge].sum()
                    processed[to_merge] = True

            # 每个 block 处理完，输出日志
            log_time(None, f"[GPU block] Processed items {i}-{i_end}")

    log_time(start_time, f"Clustering {len(word_count_dict)} items on {device}")
    return clustered_dict


# TextRank 提取关键词
def extract_keywords_textrank(df, text_column=None, stop_words_file=None):
    """
    TextRank清洗函数
    :param df: 待清洗的文本
    :param text_column: 需要清洗的列名字
    :param stop_words_file: 停用词表
    :return: 返回经过清洗后的数据
    """
    start_time = log_time(None, f"[Cleaning] Start keyword cleaning for {len(df)} rows")

    tr4w = TextRank4Keyword(stop_words_file=stop_words_file,
                            allow_speech_tags=['an', 'a', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng', 'm','q'])

    def process_row(row):
        """
        清洗后生成衍生列，用于后续字典生成
        :param row: 文本行
        """
        text = row[text_column]
        tr4w.analyze(text=text, lower=True)
        words_no_stop = tr4w.words_no_stop_words
        words_all_filters = tr4w.words_all_filters
        row['keywords_no_stop_str'] = '/'.join(words_no_stop[0]) if words_no_stop else ''
        row['keywords_filtered_str'] = '/'.join(words_all_filters[0]) if words_all_filters else ''
        row['keywords_no_stop_joined'] = row['keywords_no_stop_str'].replace('/', '')
        row['keywords_filtered_joined'] = row['keywords_filtered_str'].replace('/', '')
        return row

    df = df.apply(process_row, axis=1)

    log_time(start_time, f"[Cleaning] Finished keyword cleaning for {len(df)} rows")
    return df


# 生成关键词清洗，统计 + 聚类
def generate_keyword_stats(df=None,
                           label_col=None,
                           type_value=None,
                           text_column=None,
                           id_column=None,
                           keyword_cluster_config=None,
                           project_dir=None
                           ):
    """
    :param df: 经过标注后的文本df
    :param label_col: 目标标签列
    :param type_value: 分析的type类型内容
    :param text_column: 文本列名字
    :param id_column: 用户id列名字
    :param keyword_cluster_config: 聚类模块整个配置信息
    :param project_dir: 项目根目录
    :return: 关键句-用户 df格式
    """
    if type_value:
        df_sub = df[df[label_col].isin(type_value)].copy()
    else:
        df_sub = df.copy()

    # 解析配置
    if keyword_cluster_config is None:
        raise ValueError("必须传入 keyword_cluster_config")

    jieba_user_dict = keyword_cluster_config.get('jieba_user_dict', None)
    stop_words_file = keyword_cluster_config.get('stop_words_file', None)
    similarity_device = keyword_cluster_config.get('similarity_device', 'cpu')
    special_words = keyword_cluster_config.get('special_words', [])

    embedding_model_path = keyword_cluster_config.get('sentence_transformer_model', None)

    if not embedding_model_path:
        raise ValueError("向量化模型没有配置，请在 keyword_cluster_config 中设置 'sentence_transformer_model'")
    else:
        # 拼接 embedding_model_path
        if not os.path.isabs(embedding_model_path):
            embedding_model_path = os.path.join(project_dir, embedding_model_path)

    if jieba_user_dict:
        if not os.path.isabs(jieba_user_dict):
            jieba_user_dict = os.path.join(project_dir, jieba_user_dict)
        if os.path.exists(jieba_user_dict):
            jieba.load_userdict(jieba_user_dict)
        else:
            logger.warning(f"用户词典 {jieba} 不存在，跳过加载")

    if stop_words_file:
        if not os.path.isabs(stop_words_file):
            stop_words_file = os.path.join(project_dir, stop_words_file)

    df_sub = extract_keywords_textrank(df_sub, text_column=text_column, stop_words_file=stop_words_file)

    # TODO: 这里涉及分组统计和相似度聚类，可能会比较耗时，后续可以优化（如并行处理或者缓存中间结果）
    item_list = (df_sub[df_sub['keywords_filtered_joined'].str.len() > 4]
                 .groupby('keywords_no_stop_joined')
                 .agg({id_column: 'nunique', 'model_score_mean': 'mean'})
                 .reset_index()
                )

    if 'model_score_mean' in item_list.columns:
        item_list = item_list.drop(columns=['model_score_mean'])

    word_count_dict = {row['keywords_no_stop_joined']: row[id_column]
                       for _, row in item_list.iterrows()
                       if row['keywords_no_stop_joined'] in special_words or len(row['keywords_no_stop_joined']) > 2}
    # print(f"word_count_dict: {type(word_count_dict), len(word_count_dict)}")

    clustered_dict = cluster_by_similarity_flexible(word_count_dict, embedding_model_path, device=similarity_device)
    df_result = pd.DataFrame(list(clustered_dict.items()), columns=['Value', 'num']).sort_values(by='num', ascending=False)
    # print(f"df_result: {type(df_result)}")
    # print(f"clustered_dict: {type(clustered_dict)}")
    return df_result
