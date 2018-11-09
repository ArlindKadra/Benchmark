import unittest
from pprint import pprint

from src.benchmark import Benchmark


class TestBenchMark(unittest.TestCase):

    def test_flow_identifiers(self):

        benchmark = Benchmark('weka.RandomForest', min_task_flow=5)
        result = benchmark.get_interesting_tasks()
        pprint(result.sort_values('accuracy', axis=0))


if __name__ == '__main__':
    unittest.main()
