from loguru import logger

from data_juicer.core import Analyzer


@logger.catch(reraise=True)
def main():
    analyzer = Analyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
