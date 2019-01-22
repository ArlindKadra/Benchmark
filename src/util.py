import re

import pandas
import openml
from openml.exceptions import OpenMLServerError
import numpy as np


def get_flow_ids(*flow_qualifiers):
    """Get flow ids for the given
    flow qualifiers.

    Match the flow qualifiers to their
    corresponding ids.

    Parameters
    ----------

    flow_qualifiers: tuple
        Each flow name can be a combination of
        flowname and version:
        e.g. 'mlr.classif.ranger_8'.
        or it can also be only a flow name. The
        later, does not compose a unique flow
        qualifier.

    Returns
    -------
    flow_ids: set
        Empty set in case there are no qualifiers,
        otherwise set with flow_ids.
    """

    flow_ids = set()
    flows = openml.flows.list_flows()

    # user gave input for flow identifiers.
    if len(flow_qualifiers) != 0:
        for flow_qualifier in flow_qualifiers:
            version_match = re.search(r"\d+$", flow_qualifier)
            if version_match:
                flow_version = version_match.group(0)
            else:
                flow_version = None
            flow_name = re.sub(r"_\d+$", "", flow_qualifier)

            for key, flow in flows.items():
                if flow['name'] == flow_name:
                    if flow_version is not None:
                        if flow['version'] == flow_version:
                            flow_ids.add(key)
                    else:
                        flow_ids.add(key)
    return flow_ids


def aggregate_results_for_flow(
        results,
        column_label
):
    """Return a DataFrame with the results
    aggregated for different flow versions.

    Given a DataFrame with results for different
    flow versions and tasks, aggregate the results
    over the different flow versions, weighted by
    their contribution.

    Parameters
    ----------
    results: pandas.DataFrame
        A pandas DataFrame where the results are organized
        as follows:
            rows - are tasks
            columns - are flows
            values - fraction of runs using RandomSearch that
            can reach the minimum region.
    column_label: str
        Name for the column label. Corresponds to the name
        of the flow.

    Returns
    -------
    pandas.DataFrame
        A DataFrame that contains the
        averaged results over different
        versions of the same flow.
    """
    fraction_runs_in_region = list()
    # where the number of runs
    # with RandomSearch for each
    # flow version will be saved
    number_of_runs = list()

    matrix = dict()

    # check if the df has results
    if len(results.index) > 0:
        for index, row in results.iterrows():
            # clear the lists since we passed
            # to a different task
            fraction_runs_in_region.clear()
            number_of_runs.clear()

            # total number of runs
            # with RandomSearch for all flow
            # versions.
            total_nr_runs = 0

            # Each column is a particular version of a flow
            for column in results.columns.values.tolist():
                # tuple of values
                # First value is the number of runs in the
                # minima region
                # Second value is the number of runs in total
                entry_values = row[column]

                if entry_values is not np.NaN:
                    fraction_runs_in_region.append(entry_values[0])
                    number_of_runs.append(entry_values[1])
                    total_nr_runs += entry_values[1]

            # finished iterating over the
            # different flow versions

            # only one flow version
            if len(fraction_runs_in_region) == 1:
                matrix[index] = fraction_runs_in_region[0]
            elif len(fraction_runs_in_region) > 1:
                value = 0
                for fraction_in_region, nr_runs in zip(fraction_runs_in_region, number_of_runs):
                    # weight the percentage for the particular
                    # flow version, according to the contribution.
                    value += fraction_in_region * (nr_runs / total_nr_runs)

                matrix[index] = value

    return pandas.DataFrame.from_dict(matrix, orient='index', columns={column_label})


def get_tasks_missing_values(task_ids):
    """Given a list of task ids,
    return the tasks that have missing values.

    Return the tasks that have missing values
    from the given task collection.

    Parameters
    ----------
    task_ids: list or set
        Collection of task ids.
    Returns
    -------
    task_with_missing_values: set
        A set of tasks that have missing
        values.
    """
    tasks_with_missing_values = set()

    for task_id in task_ids:
        try:
            task = openml.tasks.get_task(task_id)
            number_missing_values = openml.datasets.get_dataset(
                task.dataset_id
            ).qualities["NumberOfMissingValues"]
            if number_missing_values != 0:
                tasks_with_missing_values.add(task_id)
        except OpenMLServerError:
            # cases in which datasets are deactivated
            pass

    return tasks_with_missing_values


def flows_using_random_search():
    """Return flows that make use
    of RandomSearch.

    Returns
    -------
    flows: set
        A set of flows for which there are
        runs using RandomSearch.
    """
    flows = set()
    # Uploader Philipp Probst
    runs = openml.runs.list_runs(uploader=[903])
    for run_id, run_info in runs.items():
        flows.add(run_info['flow_id'])

    return flows


def tasks_contained_in_openml_cc18(task_ids):
    """Given a set of task ids,
    return the ones that are
    contained in OpenMLCC18.

    Return tasks that are part of the
    OpenMLCC18 Benchmark from the set
    of given task ids.

    Parameters
    ----------
    task_ids: set
        Collection of task ids.
    Returns
    -------
    set
        A set of tasks that are part
        of OpenMLCC18.
    """
    tasks = set(openml.study.get_study(99).tasks)

    return tasks.intersection(task_ids)


def join_results(*data_frames):
    """Given a list of DataFrames,
    do an outer join and sort the
    resulting DataFrame.

    Join multiple DataFrames by rows,
    into a single one that contains the
    full information. Sort the resulting
    DataFrame by the index value of rows.

    Parameters
    ----------
    data_frames: tuple
        Set of DataFrames.
    Returns
    -------
    result: pandas.DataFrame
        Sorted DataFrame that contains all
        the information.
    """
    result = pandas.concat(data_frames, axis=1, sort=False)
    result.sort_index(inplace=True)

    return result
