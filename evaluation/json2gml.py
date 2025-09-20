#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import networkx as nx

def convert_file(json_path: Path, marker: str = "_converted") -> None:
    """
    Read one JSON graph file, build a NetworkX Graph, and write out GML.
    """
    data = json.loads(json_path.read_text())
    G = nx.Graph()

    # add nodes (keeping id, plus any extra attrs like x,y)
    for n in data.get("nodes", []):
        node_id = n["id"]
        attrs = {k: v for k, v in n.items() if k != "id"}
        G.add_node(node_id, **attrs)

    # add edges
    for e in data.get("edges", []):
        G.add_edge(e["source"], e["target"])

    # carry over graph-level attrs
    for attr in ("width", "height"):
        if attr in data:
            G.graph[attr] = data[attr]

    # build output path: same folder, same stem + marker + .gml
    out_path = json_path.with_name(f"{json_path.stem}{marker}.gml")
    nx.write_gml(G, out_path)
    print(f"→ Wrote {out_path}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h","--help"):
        print(f"Usage: {sys.argv[0]} <folder> [--marker MARKER]")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder!r} is not a directory.")
        sys.exit(2)

    # optional `--marker` argument
    marker = "_converted"
    if len(sys.argv) >= 4 and sys.argv[2] in ("-m","--marker"):
        marker = sys.argv[3]

    # process every .json file in the folder
    for json_file in sorted(folder.glob("*.json")):
        try:
            convert_file(json_file, marker)
        except Exception as e:
            print(f"⚠️  Failed on {json_file.name}: {e}")

if __name__ == "__main__":
    main()
