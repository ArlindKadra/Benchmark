from collections import defaultdict, OrderedDict
from pprint import pprint
import re

import pandas
import numpy as np

import openml


min_task_flow = 1
df = None


def get_flows(*flow_identifiers):
    """Get flow ids for the given flow identifiers.

    Match the flow identifiers to their unique flow_id.

    Parameters
    ----------

    flow_identifiers: list
        flow_identifiers represent a unique flow.
        They should be a combination of name_version
        e.g. 'mlr.classif.ranger_8'.
        It can also be only a flow name, but in that
        case, flows with different version will be
        matched.

    Returns
    -------
    flow_ids: set
        Empty set in case there are no identifiers,
        otherwise set with flow_ids.
    """

    flow_ids = set()
    flows = openml.flows.list_flows()
    flow_version = None

    # user gave input for flow identifiers.
    if len(flow_identifiers) != 0:
        for flow_identifier in flow_identifiers:
            version_match = re.search(r"\d+$", flow_identifier)
            if version_match:
                flow_version = version_match.group(0)
            flow_name = re.sub(r"_\d+$", "", flow_identifier)

            for key, flow in flows.items():
                if flow['name'] == flow_name:
                    if flow_version is not None:
                        if flow['version'] == flow_version:
                            flow_ids.add(key)
                    else:
                        flow_ids.add(key)
    return flow_ids


def build_data_frame(flow_ids):
    """Builds a DataFrame organizing runs based
    on flows and tasks.

    A list of runs will be saved for each task and
    flow.

    Parameters
    ----------

    flow_ids: set
        Ids of the flows to be considered. If None, all
        flows will be considered.


    Returns
    -------
    df: pandas.DataFrame
        A DataFrame containing all the runs for
        each task and flow.
    """
    # The structure below is:
    # outer_dict = {flow_id: inner_dict, ....}
    # inner_dict = {task_id: [run1, run2, run3, ..], ..}
    matrix = defaultdict(lambda: defaultdict(set))

    # go through each run for the given flow_ids
    # or through all. Organize them for flows and
    # tasks.
    for run in openml.runs.list_runs(
        flow=flow_ids if len(flow_ids) > 0 else None
    ).values():
        matrix[run['flow_id']][run['task_id']].add(run['run_id'])

    global df
    df = pandas.DataFrame.from_dict(data=matrix, orient='columns')


def get_interesting_tasks(evaluation_measure='predictive_accuracy'):
    """Get a list of interesting tasks.

    A list of runs will be saved for each dataset that
    meets the minimum number of runs for the flow list.
    The user can restrict the flow list, or the check
    can be performed against all flows.

    Parameters
    ----------
    evaluation_measure: str
        Evaluation measure used to compare the runs..


    Returns
    -------
    pandas.DataFrame
        A DataFrame containing different tasks
        and their accuracy. The accuracy is
        computed as an average for each algorithm
        and an average for all algorithms.
    """

    # place NaN for tasks that do not achieve
    # the minimum number of runs for a certain flow.
    revised_df = df.applymap(lambda x: x if len(x) >= min_task_flow else np.NaN)

    # drop all rows that contain one or more
    # NaN values.
    revised_df.dropna(how='any', inplace=True)
    task_accuracies = OrderedDict()
    algorithm_accuracies = []
    accuracies = []
    for index, row in df.iterrows():
        accuracies.clear()
        for column in revised_df.columns.values.tolist():
            run_ids = row[column]
            for run_id in run_ids:
                _ = openml.runs.get_run(run_id)
                algorithm_accuracies.append(_.evaluations[evaluation_measure])
            accuracies.append(np.mean(algorithm_accuracies))
            algorithm_accuracies.clear()
        task_accuracies[index] = np.mean(accuracies)

    return pandas.DataFrame.from_dict(task_accuracies, orient='index', columns=['accuracy'])


# tasks_for_flow(False, get_flows('mlr.classif.ranger_2'))
flow_ids = get_flows('mlr.classif.ranger_8', 'weka.classifiers.trees.J48_1')
build_data_frame(flow_ids)

result = get_interesting_tasks()
pprint(result.sort_values('accuracy', axis=0))
