import logging
import time
import os

# 创建 logs 文件夹 (如果不存在)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))

os.makedirs(os.path.join(project_dir, "Sentiment_Analysis_Automation", "logs"), exist_ok=True)
log_file_path = os.path.join(project_dir, "Sentiment_Analysis_Automation", "logs", "pipeline.log")

# 配置日志 (覆盖式）
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 获取 logger
logger = logging.getLogger(__name__)


def log_time(start_time=None, message="Elapsed time"):
    """
    复用时间日志函数, 兼容原来的 Labeler.py 风格
    :param start_time: 为 None 时，返回当前时间戳，用作开始时间 / 为某个时间戳时，计算耗时并且写入日志
    :param message: 日志信息
    :return: start_time 或 elapsed
    """
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    if start_time is None:
        # 开始阶段
        logger.info(f"[{now_str}] {message} 开始")
        return time.time()
    else:
        # 结束阶段
        elapsed = time.time() - start_time
        logger.info(f"[{now_str}] {message} 耗时: {elapsed:.2f}s")
        return elapsed