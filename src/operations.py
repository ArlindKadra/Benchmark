from collections import defaultdict

import numpy as np
import pandas
import openml

from src.result_extractor import ResultExtractor


def get_tasks_by_measure(
        data_frame,
        evaluation_measure='predictive_accuracy'
):
    """Get a DataFrame of tasks and flow combinations
    and their corresponding evaluation measure.

    The DataFrame rows represent tasks and the columns
    represent flows. The DataFrame entry is the
    evaluation measure averaged for all runs over
    each flow.
    The evaluation measure is given by the user.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        A pandas DataFrame where the results are organized
        as follows:
            rows - are tasks
            column - are flows
            values - set if there are runs for the flow and
            task combination, otherwise it is NaN.
    evaluation_measure: str
        Evaluation measure used to assess the runs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing task
        and flow combinations along with their
        evaluation measure value.
        The measure is averaged over all runs
        for each flow.
    """
    matrix = defaultdict(lambda: dict())

    # each entry in the list is the averaged
    # accuracy for all runs of the flow.
    flow_accuracies = []
    # check if the pandas df has results
    if len(data_frame.index) > 0:
        for index, row in data_frame.iterrows():
            for column in data_frame.columns.values.tolist():
                # clear the previous results
                flow_accuracies.clear()

                run_ids = row[column]
                if run_ids is not np.NaN:
                    for run_id in run_ids:
                        _ = openml.runs.get_run(run_id)
                        try:
                            flow_accuracies.append(_.evaluations[evaluation_measure])
                        except KeyError:
                            # this evaluation measure is not included
                            pass
                    matrix[index][column] = np.mean(flow_accuracies).item()

    return pandas.DataFrame.from_dict(matrix, orient='index')


def get_tasks_by_minima_region(
    data_frame,
    flow_ids=None,
    task_restrictions=None,
    order='increasing',
    measure='predictive_accuracy',
    threshold=0.05,
    detailed='No'
):
    """Get a DataFrame with different task and flow
     combinations showing the franction of runs using
     RandomSearch that reach the best found minima
     region by a certain threshold.

    The DataFrame contains tasks as rows and flows as
    columns. The values in the DataFrame indicate the
    fraction of runs with RandomSearch for that flow
    and task combination that are within the threshold
    range from the run which represents the best found
    minima for the given measure.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        A pandas DataFrame where the results are organized
        as follows:
            rows - are tasks
            column - are flows
            values - set if there are runs for the flow and
            task combination, otherwise it is NaN.
    flow_ids: set | None
        Restrictions on which flows to consider.
        Should be the same as the flows considered
        in the given data_frame.
    task_restrictions: dict | None
        Restrictions on which tasks to consider.
        Should be the same as the task restrictions
        in the given data_frame.
    order: str
        What to consider as the best value for the run
        prediction. The lowest value 'decreasing'
        or the highest value which corresponds to
        'increasing'.
    measure: str
        Evaluation measure used to compare the runs.
    threshold: float
        Maximal difference allowed from the best value
        to consider a run in the minima region.
    detailed: str
        If 'Yes' the entry values that represent
        fraction of runs (using RandomSearch)
        that reach the minima region will be a
        tuple and also include the number of all
        runs that make use of RandomSearch.
    Returns
    -------
    pandas.DataFrame
        A DataFrame that shows the fraction
        of runs using RandomSearch that fall
        into the best minima region for
        different task and flow combinations.
    """
    # 903 is the id of Philipp Probst
    # His experiments use RandomSearch.

    # There will always be 1 task
    # restriction. The uploader
    # restriction.
    if task_restrictions is None:
        task_restrictions = dict()
    task_restrictions['uploader'] = [903]

    result_extractor = ResultExtractor(
        *flow_ids if flow_ids is not None else None,
        **task_restrictions
    )
    # DataFrame with the best minimas found
    # for the task and flow combinations.
    best_results_df = get_tasks_by_best_score(data_frame, order=order, measure=measure)

    # DataFrame that contains for each task and flow,
    # the set of runs optimized with RandomSearch if
    # available.
    randomsearch_df = result_extractor.results

    matrix = defaultdict(lambda: dict())

    # if there are results
    if len(best_results_df.index) > 0:
        for tuple_best, tuple_rand in zip(best_results_df.iterrows(), randomsearch_df.iterrows()):
            # first element of both tuples is the index
            # second element is the row

            index = tuple_best[0]
            # get rows in each DataFrame
            best_row = tuple_best[1]
            rand_row = tuple_rand[1]

            for column in best_results_df.columns.values.tolist():
                # best value for flow and task
                best_value = best_row[column]

                # if we do have a best value
                # basically as long as there
                # is 1 run
                if best_value is not np.NaN:
                    # get all the runs for the flow
                    # runs are making use of
                    # RandomSearch
                    random_runs = rand_row[column]
                    if random_runs is not np.NaN:
                        nr_all_runs = len(random_runs)
                        nr_runs_minima_region = 0
                        for random_run in random_runs:
                            _ = openml.runs.get_run(random_run)

                            try:
                                predictive_measure = _.evaluations[measure]
                                if abs(predictive_measure - best_value) <= threshold:
                                    nr_runs_minima_region += 1
                            except KeyError:
                            # this evaluation measure is not included
                                pass
                        if detailed == 'Yes':
                            matrix[index][column] = (nr_runs_minima_region / nr_all_runs, nr_all_runs)
                        else:
                            matrix[index][column] = nr_runs_minima_region / nr_all_runs

    return pandas.DataFrame.from_dict(matrix, orient='index')


def get_tasks_by_best_score(
        data_frame,
        order='increasing',
        measure='predictive_accuracy'
):
    """Return a DataFrame with the best found
    minima value for each entry in the given
    DataFrame.

    Build a DataFrame that contains tasks as rows and
    flows as columns. The entries represent the best
    found minima value. The best value is chosen from
    the non-empty entries (run ids) in the argument
    DataFrame.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        A pandas DataFrame where the results are organized
        as follows:
            rows - are tasks
            column - are flows
            values - set if there are runs for the flow and
                task combination, otherwise it is NaN.
        From the non-empty entries in the DataFrame, the run
        ids will be used to determine the run with the best
        value for the given measure.
    order: str
        What to consider as the best value for a task and flow.
        If 'decreasing' the lower the value the better,
        vice versa for'increasing'.
    measure: str
        Evaluation measure used to compare the runs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame that contains the best
        minima value for each task and
        flow combination if runs exist.
    """
    best_score = -1
    matrix = defaultdict(lambda: dict())

    # find the minimum for the flow from all the runs

    # check if the pandas df has results
    if len(data_frame.index) > 0:
        for index, row in data_frame.iterrows():
            for column in data_frame.columns.values.tolist():
                run_ids = row[column]
                if run_ids is not np.NaN:
                    for run_id in run_ids:
                        _ = openml.runs.get_run(run_id)
                        try:
                            predictive_measure = _.evaluations[measure]
                            if order == 'increasing':
                                if predictive_measure >= best_score:
                                    best_score = predictive_measure
                            else:
                                if predictive_measure <= best_score:
                                    best_score = predictive_measure
                        except KeyError:
                            # this evaluation measure is not included
                            pass
                    matrix[index][column] = best_score
                # reset value
                best_score = -1

    return pandas.DataFrame.from_dict(matrix, orient='index')
