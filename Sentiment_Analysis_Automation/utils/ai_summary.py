import requests
from logger_utils import log_time

# 文本裁剪函数
def truncate_query_string(query_str, len_limit=8192):
    """
    裁剪文本，使其控制在指定的长度之内
    :param query_str: 待裁剪的文本
    :param len_limit: 需要控制的长度
    """
    if len(query_str) > len_limit:
        for index in range(min(len_limit, len(query_str))-1, -1, -1):
            if query_str[index] == '\n':
                query_str = query_str[:index]
                break
        else:
            query_str = query_str[:len_limit]
    return query_str


# AI 调用函数
def request_to_model(query_str,
                     analysis_config=None,
                     is_classify=False,
                     message=None,
                     temperature=0.5,
                     top_p=0.5):
    """
    AI API接口  # TODO: 你应该重写该接口来实现你自己的 LLM 模型
    """
    # 读取配置
    model_name = analysis_config['ai_name']

    if not model_name:
        raise ValueError("analysis_config 中缺少 'ai_name' 配置")

    api_url = analysis_config['ai_url']
    if not api_url:
        raise ValueError("analysis_config 中缺少 'api_url' 配置")

    if message is not None:
        fromdata = {"model": model_name, "messages": message,
                    "temperature": temperature, "top_p": top_p, "is_classify": is_classify}
    else:
        fromdata = {"model": model_name, "messages": query_str,
                    "temperature": temperature, "top_p": top_p, "is_classify": is_classify}

    req = requests.post(api_url, json=fromdata)
    x = req.json()
    return x.get('choices')[0]['message']['content']


# 新版 df_to_query, 直接接收 analysis_config
def df_to_query(df_result, analysis_config):
    """
    根据模板生成一个询问格式
    :param df_result: 待分析的关键句-人数 df
    :param analysis_config: 分析模块整个配置信息
    :return: 向 LLM 询问的内容
    """
    # 拼接 df_result 内容
    input = "\n".join(df_result.apply(lambda row: f"{row['Value']},{row['num']}", axis=1))

    # 从配置读取模板
    template_str = analysis_config.get('ai_summary_template', '')
    if not template_str:
        # 如果模板为空,则抛出异常
        raise ValueError("你应该在 pipeline.yaml 文件里面配置你的 LLM 询问模板")

    # 使用 str.format 替换变量
    query_str = template_str.format(
        analyze_content=analysis_config.get('analyze_content', ''),
        type_value=analysis_config.get('type_value', []),
        input=input,
    )

    # 裁剪长度
    query_str = truncate_query_string(query_str)
    return query_str


# ai_summary 调用
def ai_summary_pipeline(df_result, analysis_config):
    """
    :param df_result: 待分析的关键句-人数 df
    :param analysis_config: 分析模块整个配置信息
    :return: LLM 的答复
    """
    start_time = log_time(None, "ai_summary_pipeline")

    if analysis_config is None:
        raise ValueError("必须传入 analysis_config 配置")

    # 生成 LLM 询问 内容
    query_str = df_to_query(df_result, analysis_config)
    # print(f"询问内容：{query_str}")

    # 返回答复
    response = request_to_model(query_str, analysis_config)

    log_time(start_time, "ai_summary_pipeline")

    return response
