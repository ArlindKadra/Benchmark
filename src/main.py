from pprint import pprint

from src.benchmark import ResultExtracter

suite = ResultExtracter('mlr.classif.ranger_8', 'weka.classifiers.trees.J48_1')

result = suite.get_interesting_tasks()
pprint(result.sort_values('accuracy', axis=0))
