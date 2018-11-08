import unittest
from src.benchmark import Benchmark
from pprint import pprint

class TestBenchMark(unittest.TestCase):

    def test_flow_identifiers(self):

        benchmark = Benchmark('weka.RandomForest')
        result = benchmark.get_interesting_tasks()
        pprint(result.sort_values('accuracy', axis=0))


if __name__ == '__main__':
    unittest.main()
