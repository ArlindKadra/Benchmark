import unittest

from src.result_extractor import ResultExtractor


class TestBenchMark(unittest.TestCase):

    def test_flow_identifiers(self):

        benchmark = ResultExtractor('weka.RandomForest_5', min_task_flow=20)
        benchmark.get_interesting_tasks()

if __name__ == '__main__':
    unittest.main()
