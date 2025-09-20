import os
import json

def export_graph_to_json(graph, positions, width, height, filename, export_dir='exports'):
    try:
        # Ensure the export directory exists
        os.makedirs(export_dir, exist_ok=True)
        print(f"Export directory '{export_dir}' is ready.")

        # Define the path for the output file in the export directory
        output_path = os.path.join(export_dir, filename)
        print(f"Output path is set to '{output_path}'.")

        nodes = []
        for node, pos in positions.items():
            nodes.append({
                "id": node,
                "x": int(pos[0]),
                "y": int(pos[1])
            })

        edges = set()
        for u, v in graph.edges:
            edge = tuple(sorted((u, v)))
            edges.add(edge)

        edges_list = [{"source": u, "target": v} for u, v in edges]

        graph_data = {
            "nodes": nodes,
            "edges": edges_list,
            "width": width,
            "height": height
        }

        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"Graph exported to {output_path}.")
    except Exception as e:
        print(f"An error occurred while exporting the graph: {e}")