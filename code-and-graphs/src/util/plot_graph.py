import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(graph, positions=None, title="Graph Layout"):
    """
    Plots the graph with nodes colored based on their cluster membership.

    Parameters:
        graph (networkx.Graph): The graph to plot.
        positions (dict): A dictionary mapping nodes to their positions (as 2D arrays or lists).
        title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos=positions, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(title)
    plt.show()

def plot_graph_reward(graph, positions=None, reward=-1, title="Graph Layout"):
    """
    Plot graph like plot_graph, but with reward value shown on plot

    :param graph:
    :param positions:
    :param reward:
    :param title:
    :return:
    """
    if not hasattr(plot_graph_reward, "fig"):
        plot_graph_reward.fig, plot_graph_reward.ax = plt.subplots(figsize=(8, 8))

    plot_graph_reward.ax.clear()  # Clear the current plot
    nx.draw(graph, pos=positions, ax=plot_graph_reward.ax, with_labels=True,
            node_color='lightblue', edge_color='gray', node_size=500)
    if reward is not None:
        color = "green" if reward > 0 else "red"
        plot_graph_reward.ax.set_title(f"Reward: {reward}", color=color)
    plt.pause(0.01)  # Pause briefly to update the plot




def visualize_layout(graph, positions, cluster_map, title="Layout"):
    # Assign colors based on clusters
    num_clusters = len(set(cluster_map.values()))
    colors = plt.cm.tab20(range(num_clusters))  # Use a colormap with enough distinct colors
    node_colors = [colors[cluster_map[node]] for node in graph.nodes()]

    # Plot the graph
    plt.figure(figsize=(6, 6))
    nx.draw(graph, pos=positions, with_labels=True, node_color=node_colors, node_size=500)
    plt.title(title)
    plt.show()
