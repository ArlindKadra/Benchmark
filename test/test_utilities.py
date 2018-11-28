import unittest

from src.util import get_flow_ids


class TestUtilities(unittest.TestCase):

    def test_get_flow_ids(self):

        # there are 10 versions of the flow
        self.assertEqual(
            len(get_flow_ids('mlr.classif.svm')),
            10
        )
        # should return only 1 item
        self.assertEqual(
            len(get_flow_ids('mlr.classif.svm_2')),
            1
        )
        # there are 11 versions of xgboost
        self.assertEqual(
            len(get_flow_ids('mlr.classif.xgboost')),
            11
        )
        # there was a bug previously when adding
        # versioned flows. Trying both combinations
        # of the flows
        self.assertEqual(
            len(
                get_flow_ids(
                    'mlr.classif.xgboost',
                    'mlr.classif.svm_4'
                )
            ),
            12
        )
        self.assertEqual(
            len(
                get_flow_ids(
                    'mlr.classif.svm_3',
                    'mlr.classif.xgboost'
                )
            ),
            12
        )


if __name__ == '__main__':
    unittest.main()
