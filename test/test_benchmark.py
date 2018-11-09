import unittest
from src.benchmark import ResultExtracter


class TestBenchMark(unittest.TestCase):

    def test_flow_identifiers(self):

        benchmark = ResultExtracter('weka.RandomForest_5', min_task_flow=20)
        benchmark.get_interesting_tasks()

if __name__ == '__main__':
    unittest.main()
