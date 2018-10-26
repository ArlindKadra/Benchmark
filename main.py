from collections import defaultdict, Counter
from pprint import pprint
import re

import pandas

import openml


min__task_flow = 100


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


def tasks_for_flow(flow_ids):

    matrix = defaultdict(Counter)
    for run in openml.runs.list_runs(
        flow=flow_ids if len(flow_ids) > 0 else None
    ).values():
        matrix[run['flow_id']][run['task_id']] += 1
    data = pandas.DataFrame.from_dict(data=matrix, orient='index')
    pprint(data)

# tasks_for_flow(False, get_flows('mlr.classif.ranger_2'))
flow_ids = get_flows('mlr.classif.ranger_8', 'weka.classifiers.trees.J48_1')
tasks_for_flow(flow_ids)