import torch
import pandas as pd
import numpy as np
import yaml
from snownlp import SnowNLP
import os
from transformers import BertTokenizer, BertForSequenceClassification
import time
from logger_utils import log_time


# 解耦生成 models_dict 函数
def load_models_from_info(models_info=None, device="cuda", project_dir=None):
    """
    根据 models_info 列表生成 models_dict
    :param models_info: list of dict, 每个 dict 包含 name/model_path/tokenizer_path
    :param device: 模型加载设备
    :return: dict, key=model_name, value=(tokenizer, model) 或 None
    """
    models_dict = {}
    for m in models_info:
        name = m.get("name")
        model_path = m.get("model_path")
        tokenizer_path = m.get("tokenizer_path")

        if not name or not model_path or not tokenizer_path:
            raise ValueError(f"模型配置错误: {m}")

        # 路径拼接 (如果不是绝对路径, 就拼接到 project_dir )
        if not os.path.isabs(model_path):
            if project_dir is None:
                raise ValueError("使用相对路径时必须传入 project_dir")
            model_path = os.path.join(project_dir, model_path)
        if not os.path.isabs(tokenizer_path):
            if project_dir is None:
                raise ValueError("使用相对路径时必须传入 project_dir")
        tokenizer_path = os.path.join(project_dir, tokenizer_path)
        # 对 ShowNLP 的特殊处理
        if name == "SnowNLP":
            models_dict[name] = None
        else:
            model = BertForSequenceClassification.from_pretrained(model_path).to(device)
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            models_dict[name] = (model, tokenizer)

    return models_dict


# 模型评估函数
def get_sentiment(model, tokenizer, texts, device):
    """

    :param model: 情绪评估模型路径
    :param tokenizer: 情绪评估模型
    :param texts: df 文本
    :param device: 模式部署的设备 cpu/cuda
    :return: 情绪概率分布
    """
    model.eval()
    encoding = tokenizer.batch_encode_plus(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_tensors = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_tensors, attention_mask=attention_mask)
        sentiments = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return sentiments


# 工具函数
# 情感分类阈值 (闭包写法) 构造函数工厂，返回特定阈值下的数据预处理函数
def make_sentiment_category(thresholds):
    """
    :param thresholds: 情感标签分类阈值
    """
    def sentiment_category(score):
        """
        :param score: 标签分数
        """
        if score is None:
            return "中性"
        return "正向" if score > thresholds["sentiment"]["positive"] else ("负向" if score < thresholds["sentiment"]["negative"] else "中性")
    return sentiment_category


def make_mean_category(thresholds):
    def mean_category(score):
        if score is None:
            return "中性"
        return "正向" if score > thresholds["mean"]["positive"] else ("负向" if score < thresholds["mean"]["negative"] else "中性")
    return mean_category


# 投票表决
def vote_sentiment(row):
    """
    数据预处理函数，即将对整批文本逐行处理
    :param row: 文本每一行
    """
    value_counts = row.value_counts()
    if value_counts.max() > 1:
        return value_counts.idxmax()
    else:
        return "中性"


# 一致性标签定义
def consensus_tag(row):
    """
    判断多模型预测是否一致，并处理
    :param row: 一行数据
    """
    tags = row.tolist()  # row 已经包含 tag_cols 了
    # 全部正向
    if all(tag == '正向' for tag in tags):
        return '正向'
    # 全部负向
    elif all(tag == '负向' for tag in tags):
        return '负向'
    # 中性
    else:
        return '中性'


# 核心函数
def label_data(csv_path=None,
               batch_size=None,
               labeler_config=None,
               project_dir=None,
               ):
    """
    模型标数据函数，可对应模型对应列生成，集成多模型评估
    :param csv_path: 数据集路径
    :param batch_size: 模型标注时使用的batch_size
    :param labeler_config: yaml文件里面labeler字段
    :param project_dir: 当前根目录
    :return: 标注后的 df
    """
    if labeler_config is None:
        raise ValueError("必须传入 labeler_config")

    # 读取文本列
    text_column = labeler_config.get("data", {}).get("text_column")
    if not text_column:
        raise ValueError("labeler_config 缺少 data.text_column")

    # 加载模型信息
    models_info = labeler_config.get("model", [])
    if not models_info:
        raise ValueError("labeler_config 中必须至少配置一个模型")

    # 加载阈值
    thresholds = labeler_config.get("thresholds", None)
    if thresholds is None:
        raise ValueError("labeler_config 缺少 thresholds 配置")
    if not all(k in thresholds for k in ["sentiment", "mean"]):
        raise ValueError("thresholds 配置必须包含 'sentiment' 和 'mean' 两个键")

    keywords = labeler_config.get("keywords", {"positive": [], "negative": [], "neutral": []})
    device = labeler_config.get("device", "cuda")

    # 读取 CSV 并且去重
    df = pd.read_csv(csv_path).drop_duplicates()

    # 清理文本列，删除空值和非字符串类型
    df = df.dropna(subset=[text_column])  # 删除空值行
    df = df[df[text_column].map(lambda x: isinstance(x, str))].copy()  # 只保留字符串

    # 加载模型字典
    models_dict = load_models_from_info(models_info, device=device, project_dir=project_dir)

    # 构建阈值分类函数
    sentiment_category = make_sentiment_category(thresholds)
    mean_category = make_mean_category(thresholds)

    score_cols = []
    tag_cols = []

    for model_name, model_obj in models_dict.items():
        # 开始计时
        start_model_time = log_time(None, f"模型标注阶段 {model_name}")

        score_col = f"{model_name}_score"
        tag_col = f"{model_name}_tag"
        score_cols.append(score_col)
        tag_cols.append(tag_col)

        if model_name == "SnowNLP":  # SnowNLP逐行标注
            df[score_col] = df[text_column].apply(lambda x: SnowNLP(x).sentiments)
            df[tag_col] = df[score_col].apply(sentiment_category)
        else:
            model, tokenizer = model_obj
            scores = []
            for i in range(0, len(df), batch_size):
                # batch_start = time.time()
                batch_texts = df[text_column][i:i+batch_size].tolist()
                try:
                    sentiments = get_sentiment(model, tokenizer, batch_texts, device)
                    sentiments = sentiments.squeeze(1)
                    scores.extend(sentiments[:, 1].tolist())
                except Exception as e:
                    print(f"{model_name} 批处理出错： {e}")
                    scores.extend([None]*len(batch_texts))
                # print(f" 批次 {i}:{i+batch_size} 耗时 {time.time() - batch_start:.2f}s")

            df[score_col] = scores
            df[tag_col] = df[score_col].apply(lambda x: sentiment_category(x) if x is not None else '中性')

        log_time(start_model_time, f"模型标注阶段 {model_name}")

    # 优化后的衍生列生成逻辑，如果只有一个模型评分，那么后续不用生成衍生列，直接用他的标签列即可
    if len(score_cols) == 1:
        # 只有一个模型评分， 直接用对应 tag 列作为 result,并且使用唯一模型的分数
        df['result'] = df[tag_cols[0]]
        df['model_score_mean'] = df[score_cols[0]]
    else:
        # 多模型评分，执行原来的衍生列逻辑
        # 平均分
        df['model_score_mean'] = df[score_cols].mean(axis=1)

        # 平均tag
        df['平均tag'] = df[score_cols].mean(axis=1).apply(mean_category)

        # 投票
        df['投票tag'] = df[tag_cols].apply(vote_sentiment, axis=1)

        # 模型一致tag
        df['模型一致tag'] = df[tag_cols].apply(consensus_tag, axis=1)

        # 最终结果默认用投票
        df['result'] = df[['平均tag', '投票tag', '模型一致tag']].apply(vote_sentiment, axis=1)

    # 人工干预关键词
    for kw in keywords.get("positive", []):
        df['result'] = np.where(df[text_column].str.contains(kw), "正向", df['result'])
    for kw in keywords.get("negative", []):
        df['result'] = np.where(df[text_column].str.contains(kw), "负向", df['result'])
    for kw in keywords.get("neutral", []):
        df['result'] = np.where(df[text_column].str.contains(kw), "中性", df['result'])

    return df
