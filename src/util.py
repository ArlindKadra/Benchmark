import re

import openml


def get_flows_ids(*flow_identifiers):
    """Get flow ids for the given flow identifiers.

    Match the flow identifiers to their unique flow_id.

    Parameters
    ----------

    flow_identifiers: list ｜ str
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

    # user gave input for flow identifiers.
    if len(flow_identifiers) != 0:
        for flow_identifier in flow_identifiers:
            version_match = re.search(r"\d+$", flow_identifier)
            if version_match:
                flow_version = version_match.group(0)
            else:
                flow_version = None
            flow_name = re.sub(r"_\d+$", "", flow_identifier)

            for key, flow in flows.items():
                if flow['name'] == flow_name:
                    if flow_version is not None:
                        if flow['version'] == flow_version:
                            flow_ids.add(key)
                    else:
                        flow_ids.add(key)
    return flow_ids
