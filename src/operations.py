from collections import OrderedDict

import numpy as np
import pandas
import openml


def get_tasks_by_measure(data_frame, evaluation_measure='predictive_accuracy'):
    """Get a list of tasks ordered by the lowest
    value.

    A list of runs will be returned for each dataset ordered
    by the lowest value of the given predictive measure.

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
        Evaluation measure used to compare the runs.


    Returns
    -------
    pandas.DataFrame
        A DataFrame containing different tasks
        and their accuracy. The accuracy is
        averaged over all runs.
    """

    # each entry is a task -> accuracy
    # accuracy is the averaged value for
    # each algorithm.
    task_accuracies = OrderedDict()
    # array to store the accuracies
    # for an algorithm.
    algorithm_accuracies = []
    # each entry is the averaged accuracy for
    # all runs of an algorithm.
    accuracies = []
    # check if the pandas df has results
    if len(data_frame.index) > 0:
        for index, row in data_frame.iterrows():
            accuracies.clear()
            for column in data_frame.columns.values.tolist():
                run_ids = row[column]
                for run_id in run_ids:
                    _ = openml.runs.get_run(run_id)
                    try:
                        algorithm_accuracies.append(_.evaluations[evaluation_measure])
                    except KeyError:
                        # this evaluation measure is not included
                        pass
                accuracies.append(np.mean(algorithm_accuracies))
                algorithm_accuracies.clear()
            task_accuracies[index] = np.mean(accuracies)

    return pandas.DataFrame.from_dict(task_accuracies, orient='index', columns=['accuracy'])
