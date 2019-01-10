import argparse
import os

from src.result_extractor import ResultExtractor
from src.operations import get_tasks_by_minima_region
from src.util import (
    get_flow_ids,
    aggregate_results_for_flow,
    get_tasks_missing_values,
    tasks_contained_in_openml_cc18
)

parser = argparse.ArgumentParser(description="Benchmark config")

parser.add_argument(
    '--path',
    help='Path where the output will be saved.',
    default="C:\\Users\\Lindarx\\Desktop\\output",
    type=str
)
parser.add_argument(
    '--algorithm',
    help='Flow to find interesting tasks for.',
    default="Gradient Boosting",
    type=str
)


arg_parser = parser.parse_args()
algorithm = arg_parser.algorithm

gradient_boosting_flows = \
    [
        'mlr.classif.xgboost_4',
        'mlr.classif.xgboost_5'
    ]

svm_flows = \
    [
        'mlr.classif.svm_7'
    ]

random_forest_flows = \
    [
        'mlr.classif.ranger_7',
        'mlr.classif.ranger_14',
        'mlr.classif.ranger_16'
    ]

if algorithm == "Gradient Boosting":
    flow_ids = get_flow_ids(*gradient_boosting_flows)
elif algorithm == "SVM":
    flow_ids = get_flow_ids(*svm_flows)
elif algorithm == "Random Forest":
    flow_ids = get_flow_ids(*random_forest_flows)

# get the results
results = ResultExtractor(*flow_ids, task_type=1).results
random_results = get_tasks_by_minima_region(results, flow_ids, detailed='Yes')
# aggregate the results over the
# different versions of the same flow
aggregated_results = aggregate_results_for_flow(random_results, algorithm)
task_ids = aggregated_results.index.values

# get the tasks which are contained in OpenMLCC18
# and the have missing values
tasks_missing_values = get_tasks_missing_values(task_ids)
task_in_cc18 = tasks_contained_in_openml_cc18(task_ids)
undesired_tasks = tasks_missing_values.union(task_in_cc18)


aggregated_results.drop(undesired_tasks, inplace=True)
sorted_df = aggregated_results.sort_values(by=[algorithm])

# save results
with open(os.path.join(arg_parser.path, algorithm + ".csv"), "w") as file:
    sorted_df.to_csv(file)
