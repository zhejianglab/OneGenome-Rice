from benchmarks.schedule import scheduler
import argparse
from benchmarks.initation import Config


if __name__ == "__main__":

    # 只保留配置路径参数
    parser = argparse.ArgumentParser(description="序列基准测试工具")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径 (默认: config.yaml)")

    args = parser.parse_args()

    # 加载配置文件
    config = Config(args)

    # 启动调度器
    scheduler(config)
