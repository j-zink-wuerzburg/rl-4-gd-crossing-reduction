#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

inline bool bbox_overlap(double ax, double ay, double bx, double by,
                         double cx, double cy, double dx, double dy) {
    double aminx = std::min(ax, bx), amaxx = std::max(ax, bx);
    double aminy = std::min(ay, by), amaxy = std::max(ay, by);
    double cminx = std::min(cx, dx), cmaxx = std::max(cx, dx);
    double cminy = std::min(cy, dy), cmaxy = std::max(cy, dy);
    return !(amaxx < cminx || cmaxx < aminx || amaxy < cminy || cmaxy < aminy);
}

inline int orient(double ax, double ay, double bx, double by, double cx, double cy) {
    // robust-ish, consistent with Python side
    constexpr double EPS = 1e-12;
    double val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (val >  EPS) return  1;
    if (val < -EPS) return -1;
    return 0;
}

inline bool segments_intersect_strict(double ax, double ay, double bx, double by,
                                      double cx, double cy, double dx, double dy) {
    if (!bbox_overlap(ax, ay, bx, by, cx, cy, dx, dy)) return false;
    int o1 = orient(ax, ay, bx, by, cx, cy);
    int o2 = orient(ax, ay, bx, by, dx, dy);
    if (o1 == 0 || o2 == 0) return false; // touch/collinear -> not proper
    int o3 = orient(cx, cy, dx, dy, ax, ay);
    int o4 = orient(cx, cy, dx, dy, bx, by);
    if (o3 == 0 || o4 == 0) return false;
    return (o1 != o2) && (o3 != o4);
}

// Return (crossings: dict[(i,j)] -> list[(u,v)], max_edge:(i,j), global_pairs:int)
std::tuple<py::dict, std::pair<int,int>, int>
compute_crossings(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const std::vector<std::pair<int,int>>& edges
) {
    auto pos = positions.unchecked<2>();
    using Edge = std::pair<int,int>;

    const int M = static_cast<int>(edges.size());
    if (M == 0) {
        return {py::dict(), Edge{-1,-1}, 0};
    }

    // Pre-extract coordinates per edge to cut repeated loads
    struct Seg { double ax, ay, bx, by; int u, v; };
    std::vector<Seg> segs; segs.reserve(M);
    for (const auto& e : edges) {
        int a = e.first, b = e.second;
        segs.push_back({pos(a,0), pos(a,1), pos(b,0), pos(b,1), a, b});
    }

    std::vector<std::vector<int>> adj(M);
    std::vector<int> deg(M, 0);
    int global_pairs = 0;
    int argmax = -1, maxdeg = 0;

    for (int i = 0; i < M; ++i) {
        const auto& s1 = segs[i];
        for (int j = i + 1; j < M; ++j) {
            const auto& s2 = segs[j];
            // skip adjacency (share endpoint)
            if (s1.u == s2.u || s1.u == s2.v || s1.v == s2.u || s1.v == s2.v) continue;

            if (segments_intersect_strict(s1.ax, s1.ay, s1.bx, s1.by,
                                          s2.ax, s2.ay, s2.bx, s2.by)) {
                adj[i].push_back(j);
                adj[j].push_back(i);
                ++global_pairs;

                if (++deg[i] > maxdeg) { maxdeg = deg[i]; argmax = i; }
                if (++deg[j] > maxdeg) { maxdeg = deg[j]; argmax = j; }
            }
        }
    }

    // Build Python dict keyed by (i,j) pairs with values list[(u,v)]
    py::dict out;
    for (int i = 0; i < M; ++i) {
        if (adj[i].empty()) continue;
        const Edge& ei = edges[i];
        py::list lst;
        lst.attr("reserve")(adj[i].size()); // small micro-optim; pybind ok to ignore
        for (int j : adj[i]) {
            const Edge& ej = edges[j];
            lst.append(py::make_tuple(ej.first, ej.second));
        }
        out[py::make_tuple(ei.first, ei.second)] = lst;
    }

    std::pair<int,int> max_edge = (argmax >= 0) ? edges[argmax] : std::pair<int,int>{-1,-1};
    return {out, max_edge, global_pairs};
}


// New function: find crossings between incident_edges and candidate_edges
py::dict find_crossings_for_edges(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const std::vector<std::pair<int,int>>& incident_edges,
    const std::vector<std::pair<int,int>>& candidate_edges
) {
    auto pos = positions.unchecked<2>();
    py::dict out;

    struct Seg { double ax, ay, bx, by; int u, v; };
    std::vector<Seg> inc; inc.reserve(incident_edges.size());
    std::vector<Seg> cand; cand.reserve(candidate_edges.size());

    for (const auto& e : incident_edges) {
        int a = e.first, b = e.second;
        inc.push_back({pos(a,0), pos(a,1), pos(b,0), pos(b,1), a, b});
    }
    for (const auto& e : candidate_edges) {
        int c = e.first, d = e.second;
        cand.push_back({pos(c,0), pos(c,1), pos(d,0), pos(d,1), c, d});
    }

    for (size_t i = 0; i < inc.size(); ++i) {
        const auto& s1 = inc[i];
        py::list crossed;

        for (size_t j = 0; j < cand.size(); ++j) {
            const auto& s2 = cand[j];
            // per-pair adjacency skip (correct, not global)
            if (s1.u == s2.u || s1.u == s2.v || s1.v == s2.u || s1.v == s2.v) continue;
            if (!bbox_overlap(s1.ax, s1.ay, s1.bx, s1.by, s2.ax, s2.ay, s2.bx, s2.by)) continue;

            if (segments_intersect_strict(s1.ax, s1.ay, s1.bx, s1.by,
                                          s2.ax, s2.ay, s2.bx, s2.by)) {
                crossed.append(py::make_tuple(s2.u, s2.v));
            }
        }
        out[py::make_tuple(s1.u, s1.v)] = crossed;
    }
    return out;
}



PYBIND11_MODULE(graph_utils, m) {
    m.def("compute_crossings", &compute_crossings, "Compute edge crossings (full)",
          py::arg("positions"), py::arg("edges"));
    m.def("find_crossings_for_edges", &find_crossings_for_edges,
          "Crossings for incident edges vs candidate edges",
          py::arg("positions"), py::arg("incident_edges"), py::arg("candidate_edges"));
}
