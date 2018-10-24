from collections import defaultdict, Counter
from pprint import pprint
import re

import openml
min__task_flow = 100

def get_flows(*flow_identifiers):

    # expected flow identifiers
    # flowname_version
    if len(flow_identifiers) != 0:
        for flow_identifier in flow_identifiers:
            print(flow_identifier)
            version_match = re.search(r"\d+$", flow_identifier)
            if version_match:
                flow_version = version_match.group(0)
    else:
        return set(openml.flows.list_flows().keys())


def tasks_for_flow(restrictions, flows):

    matrix = defaultdict(Counter)
    for run in openml.runs.list_runs().values():
        if restrictions:
            if run['flow_id'] not in flows:
                continue
        matrix[run['flow_id']][run['task_id']] += 1

    pprint(matrix)

#tasks_for_flow(False, get_flows('mlr.classif.ranger_2'))
get_flows('mlr.classif.ranger_33')