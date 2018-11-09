from collections import defaultdict, OrderedDict

import pandas
import numpy as np
import openml


class ResultExtracter(object):

    def __init__(self, *flow_ids, **task_restrictions):

        # pandas.DataFrame with all runs for
        # the different flows and tasks
        self.df = None
        self._build_data_frame(
            *flow_ids
        )
        # Put all the task restrictions as attributes
        for key, value in task_restrictions.items():
            setattr(self, key, value)

    def _build_data_frame(self, flow_ids):
        """Builds a DataFrame organizing runs based
        on flows and tasks.

        The user can restrict the flows to be
        considered by giving a list of flow ids,
        or the check can be performed against all
        flows. A list of runs will be saved for
        each task and flow.

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

        # build a dict with the restrictions
        restrictions = dict()
        restrictions['uploader'] = \
            getattr(self, 'uploader', None)
        restrictions['task_type'] = \
            getattr(self, 'task_type', None)
        restrictions['tag'] = \
            getattr(self, 'tag', None)
        restrictions['flow'] = flow_ids
        # go through each run for the given restrictions
        # or through all. Organize them for flows and
        # tasks.
        for run in openml.runs.list_runs(
            **restrictions
        ).values():
            matrix[run['flow_id']][run['task_id']].add(run['run_id'])

        self.df = pandas.DataFrame.from_dict(data=matrix, orient='columns')

    def _validate_entry(self, x):
        """Validate an entry of the pandas
        DataFrame.

        Check if a value in the pandas df is non
        empty and if it passes the requirement.

        Parameters
        ----------
        x: set ｜ NaN
            A value in the pandas DataFrame.

        Returns
        -------
        bool
            Result of the check.
        """
        if isinstance(x, set):
            if len(x) > self.min_task_flow:
                return True
            else:
                return False
        else:
            return False

    def get_interesting_tasks(self, evaluation_measure='predictive_accuracy'):
        """Get a list of interesting tasks.

        A list of runs will be saved for each dataset that
        meets the minimum number of runs for the flow list.

        Parameters
        ----------
        evaluation_measure: str
            Evaluation measure used to compare the runs.


        Returns
        -------
        pandas.DataFrame
            A DataFrame containing different tasks
            and their accuracy. The accuracy is
            averaged over all runs.
        """
        # place NaN for tasks that do not achieve
        # the minimum number of runs for a certain flow.
        revised_df = self.df.applymap(
            lambda x: x if self._validate_entry(x) else np.NaN
        )

        # drop all rows that contain one or more
        # NaN values.
        revised_df.dropna(how='any', inplace=True)

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
        if len(revised_df.index) > 0:
            for index, row in revised_df.iterrows():
                accuracies.clear()
                for column in revised_df.columns.values.tolist():
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
