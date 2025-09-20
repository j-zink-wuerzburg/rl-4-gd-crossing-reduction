#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import networkx as nx

def gml_to_json(gml_path: Path) -> dict:
    """
    Read a GML file into a NetworkX Graph, extract nodes and their x,y from graphics,
    round and shift coordinates to positive ints, extract edges, compute width/height,
    and return a JSON-serializable dict matching the original schema.
    """
    # 1) Read the GML. Node keys default to the GML 'label', original 'id' in data['id'].
    G = nx.read_gml(str(gml_path))
    G = nx.convert_node_labels_to_integers(G)

    # 2) Extract raw node coordinates
    raw = []  # list of tuples (nid, x_float, y_float)
    for nid, data in G.nodes(data=True):
        if "x" in data and "y" in data:
            x, y = data["x"], data["y"]
        else:
            gfx = data.get("graphics", {})
            x, y = gfx.get("x", 0.0), gfx.get("y", 0.0)
        raw.append((int(nid), 10000*x, 10000*y))

    # 3) Determine shifts to make all coords non-negative
    xs = [x for (_nid, x, _y) in raw]
    ys = [y for (_nid, _x, y) in raw]
    min_x = min(xs) if xs else 0.0
    min_y = min(ys) if ys else 0.0
    shift_x = -min_x if min_x < 0 else 0.0
    shift_y = -min_y if min_y < 0 else 0.0

    # 4) Build final node list with rounded ints and track max
    nodes = []
    max_x = 0
    max_y = 0
    for nid, x, y in raw:
        xr = int(round(x + shift_x))
        yr = int(round(y + shift_y))
        nodes.append({
            "id": nid,
            "x": xr,
            "y": yr
        })
        max_x = max(max_x, xr)
        max_y = max(max_y, yr)

    # 5) Extract edges
    edges = []
    for u, v in G.edges():
        du = G.nodes[u]
        dv = G.nodes[v]
        su = int(du.get('id', u)) if 'id' in du else int(u)
        tv = int(dv.get('id', v)) if 'id' in dv else int(v)
        edges.append({"source": su, "target": tv})

    # 6) Assemble output dict
    out = {
        "nodes": nodes,
        "edges": edges,
        "width": int(max_x*1.2),
        "height": int(max_y*1.2)
    }
    return out

def gexf_to_json(gexf_path: Path, scale: float = 1_000.0) -> dict:
    """
    Read a GEXF graph, compute a Kamada-Kawai layout, shift/round coords,
    and return the dict:
        { "nodes":[{id,x,y}, …], "edges":[{source,target}, …],
          "width": W, "height": H }
    """
    # 1) read the graph
    G = nx.read_gexf(gexf_path)

    # 2) Kamada-Kawai layout  (returns floats in roughly [-1,1])
    kk_pos = nx.kamada_kawai_layout(G)

    # 3) collect & scale
    raw = []                         # (nid, x_scaled, y_scaled)
    for key in G.nodes():
        # Preserve integer ids where possible, fall back to string key
        nid = int(key) if str(key).isdigit() else key
        x, y = kk_pos[key]
        raw.append((nid, x * scale, y * scale))

    # 4) shift to make everything non-negative and round to ints
    min_x = min(x for _, x, _ in raw)
    min_y = min(y for _, _, y in raw)
    shift_x = -min_x if min_x < 0 else 0.0
    shift_y = -min_y if min_y < 0 else 0.0

    nodes, max_x, max_y = [], 0, 0
    for nid, x, y in raw:
        xr = int(round(x + shift_x))
        yr = int(round(y + shift_y))
        nodes.append({"id": nid, "x": xr, "y": yr})
        max_x, max_y = max(max_x, xr), max(max_y, yr)

    # 5) edges (undirected by default – change if you need direction)
    edges = []
    for u, v in G.edges():
        su = int(u) if str(u).isdigit() else u
        tv = int(v) if str(v).isdigit() else v
        edges.append({"source": su, "target": tv})

    # 6) final dict
    return {
        "nodes": nodes,
        "edges": edges,
        "width":  int(max_x * 1.2),
        "height": int(max_y * 1.2)
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(f"Usage: {sys.argv[0]} <gml-folder> [--marker MARKER]")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder!r} is not a directory.")
        sys.exit(2)

    # default marker for output .json filenames
    marker = "_fromgml"
    if len(sys.argv) >= 4 and sys.argv[2] in ("-m", "--marker"):
        marker = sys.argv[3]

    for gml_file in sorted(folder.glob("*.gml")):
        try:
            data = gml_to_json(gml_file)
            out_path = gml_file.with_name(f"{gml_file.stem}{marker}.json")
            with open(out_path, "w") as fp:
                json.dump(data, fp, indent=4)
            print(f"✔ Converted {gml_file.name} → {out_path.name}")
        except Exception as e:
            print(f"✖ Failed on {gml_file.name}: {e}")

if __name__ == "__main__":
    main()
