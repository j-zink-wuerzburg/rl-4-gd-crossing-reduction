#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <cstdint>
#include <limits>

namespace py = pybind11;
constexpr double PI_CONST = 3.141592653589793238462643383279502884;

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

inline bool onSegment(double px, double py, double qx, double qy, double rx, double ry) {
    constexpr double EPS = 1e-12;
    return qx <= std::max(px, rx) + EPS && qx >= std::min(px, rx) - EPS &&
           qy <= std::max(py, ry) + EPS && qy >= std::min(py, ry) - EPS;
}

inline bool segments_intersect_strict(double ax, double ay, double bx, double by,
                                      double cx, double cy, double dx, double dy) {
    if (!bbox_overlap(ax, ay, bx, by, cx, cy, dx, dy)) return false;
    int o1 = orient(ax, ay, bx, by, cx, cy);
    int o2 = orient(ax, ay, bx, by, dx, dy);
    int o3 = orient(cx, cy, dx, dy, ax, ay);
    int o4 = orient(cx, cy, dx, dy, bx, by);
    
    // General case
    if (o1 != o2 && o3 != o4) return true;
    
    // Special cases: Collinear and overlapping
    // C is on AB
    if (o1 == 0 && onSegment(ax, ay, cx, cy, bx, by)) return true;
    // D is on AB
    if (o2 == 0 && onSegment(ax, ay, dx, dy, bx, by)) return true;
    // A is on CD
    if (o3 == 0 && onSegment(cx, cy, ax, ay, dx, dy)) return true;
    // B is on CD
    if (o4 == 0 && onSegment(cx, cy, bx, by, dx, dy)) return true;
    
    return false;
}

inline double cross2d(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

inline bool ray_segment_intersection_param(double ox, double oy,
                                           double dx, double dy,
                                           double ax, double ay,
                                           double bx, double by,
                                           double eps,
                                           double& t_out) {
    const double sx = bx - ax;
    const double sy = by - ay;
    const double aox = ax - ox;
    const double aoy = ay - oy;
    const double denom = cross2d(dx, dy, sx, sy);

    if (std::abs(denom) < eps) {
        if (std::abs(cross2d(aox, aoy, dx, dy)) >= eps) {
            return false;
        }

        const double t0 = aox * dx + aoy * dy;
        const double t1 = (bx - ox) * dx + (by - oy) * dy;
        const double lo = std::min(t0, t1);
        const double hi = std::max(t0, t1);
        if (hi < eps) {
            return false;
        }
        if (lo >= eps) {
            t_out = lo;
            return true;
        }
        if (hi >= eps) {
            t_out = hi;
            return true;
        }
        return false;
    }

    const double t = cross2d(aox, aoy, sx, sy) / denom;
    const double u = cross2d(aox, aoy, dx, dy) / denom;
    if (t >= eps && u >= 0.0 && u <= 1.0) {
        t_out = t;
        return true;
    }
    return false;
}

inline std::uint64_t edge_key(int a, int b) {
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
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
//        lst.attr("reserve")(adj[i].size()); // small micro-optim; pybind ok to ignore
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

py::array_t<float> edge_ray_octant_distances(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& edges,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& incident_edges,
    int origin_idx,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& directions,
    double ray_length,
    double eps = 1e-9
) {
    auto pos = positions.unchecked<2>();
    auto edges_arr = edges.unchecked<2>();
    auto inc_arr = incident_edges.unchecked<2>();
    auto dirs_arr = directions.unchecked<2>();

    std::unordered_set<std::uint64_t> incident;
    incident.reserve(static_cast<size_t>(inc_arr.shape(0) * 2 + 1));
    for (py::ssize_t i = 0; i < inc_arr.shape(0); ++i) {
        incident.insert(edge_key(inc_arr(i, 0), inc_arr(i, 1)));
    }

    struct Seg {
        double ax, ay, bx, by;
    };
    std::vector<Seg> segs;
    segs.reserve(static_cast<size_t>(edges_arr.shape(0)));
    for (py::ssize_t i = 0; i < edges_arr.shape(0); ++i) {
        const int u = edges_arr(i, 0);
        const int v = edges_arr(i, 1);
        if (incident.find(edge_key(u, v)) != incident.end()) {
            continue;
        }
        segs.push_back({pos(u, 0), pos(u, 1), pos(v, 0), pos(v, 1)});
    }

    const double ox = pos(origin_idx, 0);
    const double oy = pos(origin_idx, 1);
    auto out = py::array_t<float>(dirs_arr.shape(0));
    auto out_view = out.mutable_unchecked<1>();

    for (py::ssize_t k = 0; k < dirs_arr.shape(0); ++k) {
        const double dx = dirs_arr(k, 0);
        const double dy = dirs_arr(k, 1);
        const double ex = ox + dx * ray_length;
        const double ey = oy + dy * ray_length;

        double best_t = ray_length;
        bool found = false;

        for (const auto& seg : segs) {
            if (!bbox_overlap(ox, oy, ex, ey, seg.ax, seg.ay, seg.bx, seg.by)) {
                continue;
            }

            double t = 0.0;
            if (ray_segment_intersection_param(ox, oy, dx, dy, seg.ax, seg.ay, seg.bx, seg.by, eps, t)) {
                if (!found || t < best_t) {
                    best_t = t;
                    found = true;
                }
            }
        }

        out_view(k) = static_cast<float>(found ? best_t : ray_length);
    }

    return out;
}

py::tuple pixel_edge_min_distances(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& edges,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& edge_ids,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& incident_ids,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& x_flat,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& y_flat
) {
    auto pos = positions.unchecked<2>();
    auto edges_arr = edges.unchecked<2>();
    auto edge_ids_arr = edge_ids.unchecked<1>();
    auto incident_ids_arr = incident_ids.unchecked<1>();
    auto x_arr = x_flat.unchecked<1>();
    auto y_arr = y_flat.unchecked<1>();

    std::unordered_set<int> incident;
    incident.reserve(static_cast<size_t>(incident_ids_arr.shape(0) * 2 + 1));
    for (py::ssize_t i = 0; i < incident_ids_arr.shape(0); ++i) {
        incident.insert(incident_ids_arr(i));
    }

    auto obs_out = py::array_t<double>(x_arr.shape(0));
    auto inc_out = py::array_t<double>(x_arr.shape(0));
    auto obs_view = obs_out.mutable_unchecked<1>();
    auto inc_view = inc_out.mutable_unchecked<1>();

    const double inf = std::numeric_limits<double>::infinity();
    for (py::ssize_t p = 0; p < x_arr.shape(0); ++p) {
        obs_view(p) = inf;
        inc_view(p) = inf;
    }

    for (py::ssize_t i = 0; i < edge_ids_arr.shape(0); ++i) {
        const int eid = edge_ids_arr(i);
        const int u = edges_arr(eid, 0);
        const int v = edges_arr(eid, 1);
        const double ax = pos(u, 0);
        const double ay = pos(u, 1);
        const double bx = pos(v, 0);
        const double by = pos(v, 1);
        const double sx = bx - ax;
        const double sy = by - ay;
        const double l2 = sx * sx + sy * sy;
        const bool is_incident = incident.find(eid) != incident.end();

        if (l2 == 0.0) {
            for (py::ssize_t p = 0; p < x_arr.shape(0); ++p) {
                const double dx = x_arr(p) - ax;
                const double dy = y_arr(p) - ay;
                const double dist_sq = dx * dx + dy * dy;
                if (is_incident) {
                    if (dist_sq < inc_view(p)) {
                        inc_view(p) = dist_sq;
                    }
                } else {
                    if (dist_sq < obs_view(p)) {
                        obs_view(p) = dist_sq;
                    }
                }
            }
            continue;
        }

        for (py::ssize_t p = 0; p < x_arr.shape(0); ++p) {
            const double dx = x_arr(p) - ax;
            const double dy = y_arr(p) - ay;
            double t = (dx * sx + dy * sy) / l2;
            t = std::max(0.0, std::min(1.0, t));
            const double proj_dx = dx - t * sx;
            const double proj_dy = dy - t * sy;
            const double dist_sq = proj_dx * proj_dx + proj_dy * proj_dy;
            if (is_incident) {
                if (dist_sq < inc_view(p)) {
                    inc_view(p) = dist_sq;
                }
            } else {
                if (dist_sq < obs_view(p)) {
                    obs_view(p) = dist_sq;
                }
            }
        }
    }

    return py::make_tuple(obs_out, inc_out);
}

py::array_t<double> edge_pair_crossing_points(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& edge_pairs
) {
    auto pos = positions.unchecked<2>();
    auto pairs = edge_pairs.unchecked<2>();

    auto out = py::array_t<double>({pairs.shape(0), static_cast<py::ssize_t>(2)});
    auto out_view = out.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < pairs.shape(0); ++i) {
        const int u1 = pairs(i, 0);
        const int v1 = pairs(i, 1);
        const int u2 = pairs(i, 2);
        const int v2 = pairs(i, 3);

        const double x1 = pos(u1, 0);
        const double y1 = pos(u1, 1);
        const double x2 = pos(v1, 0);
        const double y2 = pos(v1, 1);
        const double x3 = pos(u2, 0);
        const double y3 = pos(u2, 1);
        const double x4 = pos(v2, 0);
        const double y4 = pos(v2, 1);

        const double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (std::abs(denom) < 1e-12) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            out_view(i, 0) = nan;
            out_view(i, 1) = nan;
            continue;
        }

        const double a = (x1 * y2 - y1 * x2);
        const double b = (x3 * y4 - y3 * x4);
        out_view(i, 0) = (a * (x3 - x4) - (x1 - x2) * b) / denom;
        out_view(i, 1) = (a * (y3 - y4) - (y1 - y2) * b) / denom;
    }

    return out;
}

py::array_t<double> pixel_crossing_min_distances(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& crossing_points,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& x_flat,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& y_flat
) {
    auto cps = crossing_points.unchecked<2>();
    auto x_arr = x_flat.unchecked<1>();
    auto y_arr = y_flat.unchecked<1>();

    auto out = py::array_t<double>(x_arr.shape(0));
    auto out_view = out.mutable_unchecked<1>();

    const double inf = std::numeric_limits<double>::infinity();
    for (py::ssize_t p = 0; p < x_arr.shape(0); ++p) {
        out_view(p) = inf;
    }

    for (py::ssize_t i = 0; i < cps.shape(0); ++i) {
        const double cpx = cps(i, 0);
        const double cpy = cps(i, 1);
        if (!std::isfinite(cpx) || !std::isfinite(cpy)) {
            continue;
        }
        for (py::ssize_t p = 0; p < x_arr.shape(0); ++p) {
            const double dx = x_arr(p) - cpx;
            const double dy = y_arr(p) - cpy;
            const double dist_sq = dx * dx + dy * dy;
            if (dist_sq < out_view(p)) {
                out_view(p) = dist_sq;
            }
        }
    }

    return out;
}

py::tuple batch_octant_observations(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& all_edges,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& node_indices,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& neighbor_offsets,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& neighbor_indices,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& incident_offsets,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& incident_edges,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& incident_other_indices,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& incident_cross_counts,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& directions,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& octant_cos,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& octant_sin,
    double ray_length,
    double eps = 1e-9
) {
    auto pos = positions.unchecked<2>();
    auto edges_arr = all_edges.unchecked<2>();
    auto node_idx_arr = node_indices.unchecked<1>();
    auto neigh_off = neighbor_offsets.unchecked<1>();
    auto neigh_idx_arr = neighbor_indices.unchecked<1>();
    auto inc_off = incident_offsets.unchecked<1>();
    auto inc_edges_arr = incident_edges.unchecked<2>();
    auto inc_other_arr = incident_other_indices.unchecked<1>();
    auto inc_cross_arr = incident_cross_counts.unchecked<1>();
    auto dirs_arr = directions.unchecked<2>();
    auto cos_arr = octant_cos.unchecked<1>();
    auto sin_arr = octant_sin.unchecked<1>();

    const py::ssize_t num_nodes = node_idx_arr.shape(0);
    const py::ssize_t total_nodes = pos.shape(0);
    const py::ssize_t num_dirs = dirs_arr.shape(0);

    auto local_view_out = py::array_t<float>({num_nodes, static_cast<py::ssize_t>(40)});
    auto cross_out = py::array_t<float>({num_nodes, static_cast<py::ssize_t>(8)});
    auto cross_local_out = py::array_t<float>({num_nodes, static_cast<py::ssize_t>(8)});
    auto rotation_out = py::array_t<int>(num_nodes);
    auto local_view = local_view_out.mutable_unchecked<2>();
    auto cross_view = cross_out.mutable_unchecked<2>();
    auto cross_local_view = cross_local_out.mutable_unchecked<2>();
    auto rotation_view = rotation_out.mutable_unchecked<1>();

    std::vector<unsigned char> neigh_mask(static_cast<size_t>(total_nodes));

    for (py::ssize_t node_slot = 0; node_slot < num_nodes; ++node_slot) {
        std::fill(neigh_mask.begin(), neigh_mask.end(), static_cast<unsigned char>(0));
        for (int idx = neigh_off(node_slot); idx < neigh_off(node_slot + 1); ++idx) {
            neigh_mask[static_cast<size_t>(neigh_idx_arr(idx))] = static_cast<unsigned char>(1);
        }

        const int node_idx = node_idx_arr(node_slot);
        const double nx = pos(node_idx, 0);
        const double ny = pos(node_idx, 1);

        float counts[8] = {0};
        float neigh_d[8];
        float nonneigh_d[8];
        for (int b = 0; b < 8; ++b) {
            neigh_d[b] = std::numeric_limits<float>::infinity();
            nonneigh_d[b] = std::numeric_limits<float>::infinity();
        }

        for (py::ssize_t j = 0; j < total_nodes; ++j) {
            if (j == node_idx) {
                continue;
            }
            const double dx = pos(j, 0) - nx;
            const double dy = pos(j, 1) - ny;
            double angle = std::atan2(dy, dx) * 180.0 / PI_CONST;
            angle = std::fmod(angle + 360.0, 360.0);
            const int bin = static_cast<int>(std::floor(angle / 45.0)) % 8;
            counts[bin] += 1.0f;

            const float dist = static_cast<float>(std::hypot(dx, dy));
            if (neigh_mask[static_cast<size_t>(j)] != 0U) {
                if (dist < neigh_d[bin]) {
                    neigh_d[bin] = dist;
                }
            } else {
                if (dist < nonneigh_d[bin]) {
                    nonneigh_d[bin] = dist;
                }
            }
        }

        float rel_counts[8];
        float rel_abs_counts[8];
        float cross_oct[8] = {0};
        float cross_oct_local[8] = {0};
        float edge_ray[8];

        float count_sum = 0.0f;
        float count_max = 0.0f;
        for (int b = 0; b < 8; ++b) {
            count_sum += counts[b];
            if (counts[b] > count_max) {
                count_max = counts[b];
            }
            if (!std::isfinite(neigh_d[b])) {
                neigh_d[b] = 0.0f;
            }
            if (!std::isfinite(nonneigh_d[b])) {
                nonneigh_d[b] = 0.0f;
            }
        }
        for (int b = 0; b < 8; ++b) {
            rel_counts[b] = (count_sum > 0.0f) ? (counts[b] / count_sum) : 0.0f;
            rel_abs_counts[b] = (count_max > 0.0f) ? (counts[b] / count_max) : 0.0f;
        }

        std::unordered_set<std::uint64_t> incident;
        incident.reserve(static_cast<size_t>((inc_off(node_slot + 1) - inc_off(node_slot)) * 2 + 1));
        for (int idx = inc_off(node_slot); idx < inc_off(node_slot + 1); ++idx) {
            const int u = inc_edges_arr(idx, 0);
            const int v = inc_edges_arr(idx, 1);
            incident.insert(edge_key(u, v));

            const int other_idx = inc_other_arr(idx);
            const double dx = pos(other_idx, 0) - nx;
            const double dy = pos(other_idx, 1) - ny;
            if (dx == 0.0 && dy == 0.0) {
                continue;
            }
            double angle = std::atan2(dy, dx) * 180.0 / PI_CONST;
            angle = std::fmod(angle + 360.0, 360.0);
            const int bin = static_cast<int>(std::floor(angle / 45.0)) % 8;
            const float c = inc_cross_arr(idx);
            cross_oct[bin] += c;
            if (c > cross_oct_local[bin]) {
                cross_oct_local[bin] = c;
            }
        }

        float cross_max = 0.0f;
        float cross_local_max = 0.0f;
        for (int b = 0; b < 8; ++b) {
            if (cross_oct[b] > cross_max) {
                cross_max = cross_oct[b];
            }
            if (cross_oct_local[b] > cross_local_max) {
                cross_local_max = cross_oct_local[b];
            }
        }
        if (cross_max > 0.0f) {
            for (int b = 0; b < 8; ++b) {
                cross_oct[b] /= cross_max;
            }
        }
        if (cross_local_max > 0.0f) {
            for (int b = 0; b < 8; ++b) {
                cross_oct_local[b] /= cross_local_max;
            }
        }

        for (py::ssize_t k = 0; k < num_dirs; ++k) {
            const double dx = dirs_arr(k, 0);
            const double dy = dirs_arr(k, 1);
            const double ex = nx + dx * ray_length;
            const double ey = ny + dy * ray_length;
            double best_t = ray_length;
            bool found = false;
            for (py::ssize_t eidx = 0; eidx < edges_arr.shape(0); ++eidx) {
                const int u = edges_arr(eidx, 0);
                const int v = edges_arr(eidx, 1);
                if (incident.find(edge_key(u, v)) != incident.end()) {
                    continue;
                }
                const double ax = pos(u, 0);
                const double ay = pos(u, 1);
                const double bx = pos(v, 0);
                const double by = pos(v, 1);
                if (!bbox_overlap(nx, ny, ex, ey, ax, ay, bx, by)) {
                    continue;
                }
                double t = 0.0;
                if (ray_segment_intersection_param(nx, ny, dx, dy, ax, ay, bx, by, eps, t)) {
                    if (!found || t < best_t) {
                        best_t = t;
                        found = true;
                    }
                }
            }
            edge_ray[k] = static_cast<float>(found ? best_t : ray_length);
        }

        double vx = 0.0;
        double vy = 0.0;
        for (int b = 0; b < 8; ++b) {
            vx += static_cast<double>(cross_oct[b]) * cos_arr(b);
            vy += static_cast<double>(cross_oct[b]) * sin_arr(b);
        }
        const double angle = std::atan2(vy, vx);
        const int rot = ((static_cast<int>(std::llround(angle / (2.0 * PI_CONST / 8.0)))) % 8 + 8) % 8;
        rotation_view(node_slot) = rot;

        for (int b = 0; b < 8; ++b) {
            const int src = (b + rot) % 8;
            cross_view(node_slot, b) = cross_oct[src];
            cross_local_view(node_slot, b) = cross_oct_local[src];
            local_view(node_slot, b) = rel_counts[src];
            local_view(node_slot, 8 + b) = neigh_d[src];
            local_view(node_slot, 16 + b) = nonneigh_d[src];
            local_view(node_slot, 24 + b) = rel_abs_counts[src];
            local_view(node_slot, 32 + b) = edge_ray[src];
        }
    }

    return py::make_tuple(local_view_out, cross_out, cross_local_out, rotation_out);
}

py::array_t<float> batch_pixel_maps(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& edges,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& node_indices,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& rotation_indices,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& edge_offsets,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& edge_ids_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& incident_offsets,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& incident_ids_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& crossing_offsets,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& crossing_points_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pixel_dx_flat_arr,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pixel_dy_flat_arr,
    int patch_size,
    double pixel_decay_alpha
) {
    auto pos = positions.unchecked<2>();
    auto edges_arr = edges.unchecked<2>();
    auto node_idx_arr = node_indices.unchecked<1>();
    auto rot_arr = rotation_indices.unchecked<1>();
    auto edge_off = edge_offsets.unchecked<1>();
    auto edge_ids = edge_ids_arr.unchecked<1>();
    auto inc_off = incident_offsets.unchecked<1>();
    auto inc_ids = incident_ids_arr.unchecked<1>();
    auto cross_off = crossing_offsets.unchecked<1>();
    auto cross_pts = crossing_points_arr.unchecked<2>();
    auto pixel_dx = pixel_dx_flat_arr.unchecked<2>();
    auto pixel_dy = pixel_dy_flat_arr.unchecked<2>();

    const py::ssize_t num_slots = node_idx_arr.shape(0);
    const py::ssize_t num_pixels = static_cast<py::ssize_t>(patch_size) *
                                   static_cast<py::ssize_t>(patch_size);

    auto out = py::array_t<float>({num_slots, static_cast<py::ssize_t>(3),
                                    static_cast<py::ssize_t>(patch_size),
                                    static_cast<py::ssize_t>(patch_size)});
    auto out_view = out.mutable_unchecked<4>();

    const double inf = std::numeric_limits<double>::infinity();
    const double alpha = pixel_decay_alpha;

    std::vector<double> obs_buf(num_pixels);
    std::vector<double> inc_buf(num_pixels);
    std::vector<double> cross_buf(num_pixels);
    std::vector<double> x_pix(num_pixels);
    std::vector<double> y_pix(num_pixels);

    for (py::ssize_t slot = 0; slot < num_slots; ++slot) {
        const int node_idx = node_idx_arr(slot);
        const int rot = rot_arr(slot) % 8;
        const double nx = pos(node_idx, 0);
        const double ny = pos(node_idx, 1);

        for (py::ssize_t p = 0; p < num_pixels; ++p) {
            x_pix[p] = nx + pixel_dx(rot, p);
            y_pix[p] = ny + pixel_dy(rot, p);
            obs_buf[p] = inf;
            inc_buf[p] = inf;
            cross_buf[p] = inf;
        }

        const int inc_start = inc_off(slot);
        const int inc_end = inc_off(slot + 1);
        std::unordered_set<int> incident_set;
        incident_set.reserve(static_cast<size_t>((inc_end - inc_start) * 2));
        for (int idx = inc_start; idx < inc_end; ++idx) {
            incident_set.insert(inc_ids(idx));
        }

        const int edge_start = edge_off(slot);
        const int edge_end = edge_off(slot + 1);
        for (int eidx = edge_start; eidx < edge_end; ++eidx) {
            const int eid = edge_ids(eidx);
            const int u = edges_arr(eid, 0);
            const int v = edges_arr(eid, 1);
            const double ax = pos(u, 0);
            const double ay = pos(u, 1);
            const double bx = pos(v, 0);
            const double by = pos(v, 1);
            const double sx = bx - ax;
            const double sy = by - ay;
            const double l2 = sx * sx + sy * sy;
            const bool is_incident = incident_set.find(eid) != incident_set.end();

            auto& target_buf = is_incident ? inc_buf : obs_buf;

            if (l2 == 0.0) {
                for (py::ssize_t p = 0; p < num_pixels; ++p) {
                    const double dx = x_pix[p] - ax;
                    const double dy = y_pix[p] - ay;
                    const double dist_sq = dx * dx + dy * dy;
                    if (dist_sq < target_buf[p]) {
                        target_buf[p] = dist_sq;
                    }
                }
            } else {
                for (py::ssize_t p = 0; p < num_pixels; ++p) {
                    const double dx = x_pix[p] - ax;
                    const double dy = y_pix[p] - ay;
                    const double t = std::max(0.0, std::min(1.0, (dx * sx + dy * sy) / l2));
                    const double proj_dx = dx - t * sx;
                    const double proj_dy = dy - t * sy;
                    const double dist_sq = proj_dx * proj_dx + proj_dy * proj_dy;
                    if (dist_sq < target_buf[p]) {
                        target_buf[p] = dist_sq;
                    }
                }
            }
        }

        const int cross_start = cross_off(slot);
        const int cross_end = cross_off(slot + 1);
        for (int cidx = cross_start; cidx < cross_end; ++cidx) {
            const double cpx = cross_pts(cidx, 0);
            const double cpy = cross_pts(cidx, 1);
            if (!std::isfinite(cpx) || !std::isfinite(cpy)) {
                continue;
            }
            for (py::ssize_t p = 0; p < num_pixels; ++p) {
                const double dx = x_pix[p] - cpx;
                const double dy = y_pix[p] - cpy;
                const double dist_sq = dx * dx + dy * dy;
                if (dist_sq < cross_buf[p]) {
                    cross_buf[p] = dist_sq;
                }
            }
        }

        for (py::ssize_t p = 0; p < num_pixels; ++p) {
            const double obs_val = std::exp(-alpha * std::sqrt(obs_buf[p]));
            const double inc_val = std::exp(-alpha * std::sqrt(inc_buf[p]));
            const double cross_val = std::exp(-alpha * std::sqrt(cross_buf[p]));
            const py::ssize_t h = p / patch_size;
            const py::ssize_t w = p % patch_size;
            out_view(slot, 0, h, w) = static_cast<float>(obs_val);
            out_view(slot, 1, h, w) = static_cast<float>(inc_val);
            out_view(slot, 2, h, w) = static_cast<float>(cross_val);
        }
    }

    return out;
}

PYBIND11_MODULE(graph_utils, m) {
    m.def("compute_crossings", &compute_crossings, "Compute edge crossings (full)",
          py::arg("positions"), py::arg("edges"));
    m.def("find_crossings_for_edges", &find_crossings_for_edges,
          "Crossings for incident edges vs candidate edges",
          py::arg("positions"), py::arg("incident_edges"), py::arg("candidate_edges"));
    m.def("edge_ray_octant_distances", &edge_ray_octant_distances,
          "Distance to first non-incident edge hit for each ray direction",
          py::arg("positions"), py::arg("edges"), py::arg("incident_edges"),
          py::arg("origin_idx"), py::arg("directions"), py::arg("ray_length"),
          py::arg("eps") = 1e-9);
    m.def("pixel_edge_min_distances", &pixel_edge_min_distances,
          "Minimum squared distances from patch pixels to non-incident and incident edges",
          py::arg("positions"), py::arg("edges"), py::arg("edge_ids"),
          py::arg("incident_ids"), py::arg("x_flat"), py::arg("y_flat"));
    m.def("edge_pair_crossing_points", &edge_pair_crossing_points,
          "Compute line-intersection points for a batch of crossing edge pairs",
          py::arg("positions"), py::arg("edge_pairs"));
    m.def("pixel_crossing_min_distances", &pixel_crossing_min_distances,
          "Minimum squared distances from patch pixels to crossing points",
          py::arg("crossing_points"), py::arg("x_flat"), py::arg("y_flat"));
    m.def("batch_pixel_maps", &batch_pixel_maps,
          "Batch pixel-map computation for multiple candidate nodes",
          py::arg("positions"), py::arg("edges"), py::arg("node_indices"),
          py::arg("rotation_indices"), py::arg("edge_offsets"),
          py::arg("edge_ids_arr"), py::arg("incident_offsets"),
          py::arg("incident_ids_arr"), py::arg("crossing_offsets"),
          py::arg("crossing_points_arr"), py::arg("pixel_dx_flat_arr"),
          py::arg("pixel_dy_flat_arr"), py::arg("patch_size"),
          py::arg("pixel_decay_alpha"));
    m.def("batch_octant_observations", &batch_octant_observations,
          "Batch octant/vector observations for multiple candidate nodes",
          py::arg("positions"), py::arg("all_edges"), py::arg("node_indices"),
          py::arg("neighbor_offsets"), py::arg("neighbor_indices"),
          py::arg("incident_offsets"), py::arg("incident_edges"),
          py::arg("incident_other_indices"), py::arg("incident_cross_counts"),
          py::arg("directions"), py::arg("octant_cos"), py::arg("octant_sin"),
          py::arg("ray_length"), py::arg("eps") = 1e-9);
}
