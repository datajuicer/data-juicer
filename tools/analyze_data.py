from loguru import logger

from data_juicer.config import init_configs


@logger.catch(reraise=True)
def main():
    cfg = init_configs(allow_auto=True)
    if cfg.executor_type in ("ray", "ray_partitioned"):
        from data_juicer.core import RayAnalyzer

        analyzer = RayAnalyzer(cfg)
    else:
        from data_juicer.core import Analyzer

        analyzer = Analyzer(cfg)
    analyzer.run()


if __name__ == "__main__":
    main()
