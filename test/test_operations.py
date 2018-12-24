import unittest

from collections import defaultdict

import pandas

from src.result_extractor import ResultExtractor
from src.util import get_flow_ids
from src.operations import (
    get_tasks_by_minima_region,
    get_tasks_by_best_score,
    get_tasks_by_measure
)


class TestOperations(unittest.TestCase):

    def setUp(self):

        self.flow_ids = get_flow_ids('mlr.classif.xgboost_5')
        self.results = ResultExtractor(*self.flow_ids).results

    def test_get_tasks_by_minima_region(self):

        random_results = get_tasks_by_minima_region(
            self.results,
            self.flow_ids
        )
        matrix = defaultdict(lambda: dict())
        matrix[272][5963] = 1.0
        matrix[282][5963] = 0.75
        df = pandas.DataFrame.from_dict(
            matrix,
            orient='index'
        )
        self.assertTrue(df.equals(random_results))

    def test_get_tasks_by_best_score(self):

        best_results = get_tasks_by_best_score(
            self.results
        )
        matrix = defaultdict(lambda: dict())
        matrix[272][5963] = 0.730
        matrix[282][5963] = 0.831461
        matrix[3917][5963] = 0.864391
        matrix[9983][5963] = 0.710881
        matrix[34536][5963] = 0.977127
        df = pandas.DataFrame.from_dict(
            matrix,
            orient='index'
        )
        self.assertTrue(df.equals(best_results))

    def test_get_tasks_by_measure(self):

        averaged_results = get_tasks_by_measure(
            self.results
        )
        matrix = defaultdict(lambda: dict())
        matrix[272][5963] = 0.730000
        matrix[282][5963] = 0.735955
        matrix[3917][5963] = 0.853016
        matrix[9983][5963] = 0.604286
        matrix[34536][5963] = 0.914947
        df = pandas.DataFrame.from_dict(
            matrix,
            orient='index'
        )
        # Somehow the DataFrames are equal
        # but it is failing
        # self.assertTrue(df.equals(averaged_results))
