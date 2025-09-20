import logging
from typing import Any, Callable, Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hierarchical_graph import build_hierarchy

# Tolerance for zero-length vectors
EPS: float = 1e-8

Layout = Dict[Any, Tuple[float, float]]
ExternalDirMap = Dict[Any, complex]
ClusterScaleFunc = Callable[[int], float]
DirectionAggregator = Callable[[list[complex]], complex]


def default_direction_aggregator(vectors: list[complex]) -> complex:
    """
    Aggregates a list of complex direction vectors into a single unit vector by averaging.
    Returns zero if the input is empty or degenerate.
    """
    if not vectors:
        return 0+0j
    s = sum(vectors)
    norm = abs(s)
    return s / norm if norm > EPS else 0+0j


def _compute_optimal_rotation(
    pos_rel: ExternalDirMap,
    external_dirs: ExternalDirMap
) -> float:
    """
    Compute the rotation angle (in radians) that best aligns each internal
    direction vector to its target external direction.

    Args:
        pos_rel: centered positions as complex numbers
        external_dirs: desired unit vectors as complex numbers

    Returns:
        Optimal rotation angle in radians.
    """
    s: complex = 0+0j
    for node, v_ext in external_dirs.items():
        v_int = pos_rel.get(node, 0+0j)
        if abs(v_int) < EPS or abs(v_ext) < EPS:
            continue
        v_int_u = v_int / abs(v_int)
        v_ext_u = v_ext / abs(v_ext)
        s += v_ext_u * np.conj(v_int_u)
    return float(np.angle(s)) if abs(s) > EPS else 0.0


def hierarchical_divide_and_conquer_with_rotation(
    G: nx.Graph,
    rl_layout_func: Callable[[nx.Graph], Layout],
    max_levels: int = 5,
    cluster_scale: Union[float, ClusterScaleFunc] = 0.5,
    skip_rotation_levels: Optional[set[int]] = None,
    direction_aggregator: DirectionAggregator = default_direction_aggregator,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    **rl_kwargs: Any
) -> Layout:
    """
    Compute a hierarchical layout by recursively partitioning and optimally rotating clusters,
    with level-dependent scaling and optional layout-strength variation.

    Args:
        G: Input graph
        rl_layout_func: Layout function applied at each cluster
        max_levels: Maximum hierarchy depth
        cluster_scale: Scaling factor or a function mapping level→scale
        skip_rotation_levels: Levels at which to skip rotation
        direction_aggregator: Function to aggregate external directions
        k_min: Minimum spring constant (for spring_layout)
        k_max: Maximum spring constant (for spring_layout)
        **rl_kwargs: Additional args for rl_layout_func

    Returns:
        A position map for the nodes in G.
    """
    if max_levels < 1:
        raise ValueError("max_levels must be at least 1")
    skip_rotation_levels = skip_rotation_levels or set()

    logging.info("Building hierarchy up to %d levels", max_levels)
    levels = build_hierarchy(G, max_levels=max_levels)
    n_levels = len(levels)

    # Top-level layout
    top_graph = levels[-1].graph
    top_kwargs = rl_kwargs.copy()
    if k_min is not None and k_max is not None and rl_layout_func == nx.spring_layout:
        # largest k at top-level
        k = k_min + (n_levels - 1) / max(1, n_levels - 1) * (k_max - k_min)
        top_kwargs['k'] = k
    pos_by_level: dict[int, Layout] = {n_levels - 1: rl_layout_func(top_graph, **top_kwargs)}

    # Precompute node->cluster mapping
    level_node2cluster = []
    for level in levels:
        m: Dict[Any, Any] = {}
        for cid, members in level.clusters.items():
            for u in members:
                m[u] = cid
        level_node2cluster.append(m)

    # Bottom-up refinement
    for lvl in range(n_levels - 2, -1, -1):
        level = levels[lvl]
        parent_pos = pos_by_level[lvl + 1]
        current_pos: Layout = {}

        # Determine per-level cluster scale
        scale = cluster_scale(lvl) if callable(cluster_scale) else cluster_scale

        for cid, members in level.clusters.items():
            cx, cy = parent_pos[cid]
            subG = level.graph.subgraph(members)

            # Level-specific layout kwargs
            lvl_kwargs = rl_kwargs.copy()
            if k_min is not None and k_max is not None and rl_layout_func == nx.spring_layout:
                # interpolate k: smaller for lower levels
                k = k_min + (lvl / max(1, n_levels - 1)) * (k_max - k_min)
                lvl_kwargs['k'] = k

            raw_pos = rl_layout_func(subG, **lvl_kwargs)

            # Center sub-layout
            xs, ys = zip(*raw_pos.values())
            mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))
            pos_rel = {u: complex(x - mean_x, y - mean_y) for u, (x, y) in raw_pos.items()}

            # Compute external directions
            external_dirs: ExternalDirMap = {}
            for u in members:
                dirs: list[complex] = []
                for v in G.neighbors(u):
                    if v in members:
                        continue
                    p_cid = level_node2cluster[lvl + 1].get(v)
                    if p_cid is not None:
                        px, py = parent_pos[p_cid]
                        dirs.append(complex(px - cx, py - cy))
                if dirs:
                    external_dirs[u] = direction_aggregator(dirs)

            # Compute rotation
            rot = 1+0j if lvl in skip_rotation_levels else complex(
                np.cos(_compute_optimal_rotation(pos_rel, external_dirs)),
                np.sin(_compute_optimal_rotation(pos_rel, external_dirs))
            )

            # Place nodes
            for u, v in pos_rel.items():
                z = v * rot
                current_pos[u] = (cx + scale * z.real, cy + scale * z.imag)

        pos_by_level[lvl] = current_pos
        logging.info("Completed layout for level %d (scale=%.2f)", lvl, scale)

    return pos_by_level[0]


def plot_graph(
    G: nx.Graph,
    pos: Layout,
    title: str,
    node2cluster: Optional[Dict[Any, Any]] = None
) -> None:
    """
    Draw the graph with optional cluster color-coding.

    Args:
        G: Graph to draw
        pos: Position mapping
        title: Plot title
        node2cluster: Optional cluster assignment per node
    """
    plt.figure(figsize=(6, 6))
    if node2cluster:
        unique = sorted(set(node2cluster.values()))
        cmap = plt.get_cmap('tab20')
        cm = {cid: cmap(i / max(1, len(unique) - 1)) for i, cid in enumerate(unique)}
        colors = [cm[node2cluster[n]] for n in G]
    else:
        colors = 'lightgray'

    nx.draw(
        G,
        pos,
        node_size=[50 + 10 * G.degree(n) for n in G],
        node_color=colors,
        with_labels=False,
        linewidths=0.2
    )
    plt.title(title)
    plt.axis('off')

# Example usage
if __name__ == "__main__":
    import Training.LayoutEvaluator
    from hierarchical.cluster_handler import hierarchical_divide_and_conquer_with_rotation, plot_graph

    # change workdir to src... 2 directories up
    import os

    # Get the current working directory
    current_dir = os.getcwd()

    # Move two directories up
    new_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # Change the working directory
    os.chdir(new_dir)

    print("Changed working directory to:", os.getcwd())





    logging.basicConfig(level=logging.INFO)
    G = nx.barabasi_albert_graph(20, 1)
    G = nx.erdos_renyi_graph(50, 1)
    G = nx.read_gml("graphs/extended_BA/data/ba_17_n377_m3.gml")
    G = nx.relabel_nodes(G, lambda x: int(x))

    init_pos = nx.spring_layout(G)
    plot_graph(G, init_pos, "Initial Spring Layout")
    evalr = Training.LayoutEvaluator.LayoutEvaluatorCore(G, init_pos)
    print("Initial layout evaluation:\n", evalr.evaluate())

    max_lv = 4
    levels = build_hierarchy(G, max_levels=max_lv)


    for lvl in [0, 1]:
        lvl_graph = levels[lvl].graph
        pos_lvl = nx.spring_layout(lvl_graph)
        node2cluster = {u: cid for cid, members in levels[lvl].clusters.items() for u in members}
        plot_graph(lvl_graph, pos_lvl, f"Level {lvl} Clusters", node2cluster)

    final_pos = hierarchical_divide_and_conquer_with_rotation(
        G,
        rl_layout_func=nx.spring_layout,
        max_levels=max_lv,
        cluster_scale=lambda lvl: 0.1 + 0.2 * lvl,
        k_min=0.05,
        k_max=1,
        skip_rotation_levels={0}
    )

    node2cluster0 = {u: cid for cid, members in levels[0].clusters.items() for u in members}
    plot_graph(G, final_pos, "Final Hierarchical Layout", node2cluster0)
    evalr = Training.LayoutEvaluator.LayoutEvaluatorCore(G, final_pos)
    print("Final layout evaluation:\n", evalr.evaluate())
    plt.show()
