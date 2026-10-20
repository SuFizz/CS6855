"""Compute a node's incoming, outgoing, and total connection counts."""
from graphkit.contract import check_node, error, get_graph, node_ref, ok


def run(store, graph_id: str, node: str = None, **params) -> dict:
    g, err = get_graph(store, graph_id)
    if err:
        return err
    if params:
        return error(f"Unknown parameter(s): {', '.join(params)}. node_degree takes graph_id and node.")
    err = check_node(g, node)
    if err:
        return err

    out_degree = len(g.out_neighbors(node))
    in_degree = len(g.in_neighbors(node))
    result = node_ref(g, node, out_degree=out_degree, in_degree=in_degree,
                      degree=out_degree + in_degree if g.directed else out_degree)
    if g.directed:
        summary = (f"{result['label']} has {out_degree} outgoing and {in_degree} incoming "
                   f"connections, for a total degree of {result['degree']}.")
    else:
        summary = f"{result['label']} has degree {result['degree']} in the undirected graph."
    return ok(result, summary)
