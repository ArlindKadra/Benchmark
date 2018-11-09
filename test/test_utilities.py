import unittest
from util import get_flows_ids


class TestUtilities(unittest.TestCase):

    def test_get_flows_ids(self):

        # there are 10 versions of the flow
        self.assertEqual(
            len(get_flows_ids('mlr.classif.svm')),
            10
        )
        # should return only 1 item
        self.assertEqual(
            len(get_flows_ids('mlr.classif.svm_2')),
            1
        )


if __name__ == '__main__':
    unittest.main()
