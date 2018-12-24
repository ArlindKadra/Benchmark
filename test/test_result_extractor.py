import unittest

from src.result_extractor import ResultExtractor
from src.util import get_flow_ids


# In case a unit test fails,
# check that the number of tasks
# for the filters has not changed.
class TestResultExtractor(unittest.TestCase):

    def test_flow_identifiers(self):

        flow_identifier = 'weka.RandomForest_5'
        flow_ids = get_flow_ids(flow_identifier)
        df = ResultExtractor(*flow_ids).results
        self.assertEqual(len(df.index), 1199)

    def test_task_filter_task_type(self):

        # filtering by type of task
        # 3 corresponds to learning curve
        df = ResultExtractor(task_type=3).results
        self.assertEqual(len(df.index), 252)

    def test_task_filter_uploader(self):

        # filtering by uploader
        df = ResultExtractor(uploader=[86]).results
        self.assertEqual(len(df.index), 104)

    def test_task_filter_tag(self):

        # filtering by tag
        df = ResultExtractor(tag='weka').results
        self.assertEqual(len(df.index), 1632)

    def test_task_filter_combined(self):

        flow_identifier = 'weka.RandomForest_5'
        task_type = 1
        min_task_flow = 5
        flow_ids = get_flow_ids(flow_identifier)
        df = ResultExtractor(
            *flow_ids,
            task_type=task_type,
            min_task_flow=min_task_flow
        ).results
        self.assertEqual(len(df.index), 968)
