from collections import defaultdict

import pandas
import numpy as np
import openml


class ResultExtractor(object):

    def __init__(self, *flow_ids, **task_restrictions):

        # pandas.DataFrame with all runs formult
        # the different flows and tasks
        self._df = None

        # Put all the task restrictions as attributes
        for key, value in task_restrictions.items():
            setattr(self, key, value)

        self._build_data_frame(
            flow_ids
        )

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

        self._df = pandas.DataFrame.from_dict(data=matrix, orient='columns')

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

    @property
    def results(self):

        lower_limit = getattr(self, 'min_task_flow', None)

        if lower_limit is not None:
            # place NaN for tasks that do not achieve
            # the minimum number of runs for a certain flow.
            revised_df = self.df.applymap(
                lambda x: x if self._validate_entry(x) else np.NaN
            )

            # drop all rows that contain one or more
            # NaN values.
            revised_df.dropna(how='any', inplace=True)

            return revised_df

        else:
            return self._df
