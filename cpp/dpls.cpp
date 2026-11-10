// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 S. Ardizzoni, L. Consolini, M. Locatelli, I. Saccani
// dpls: DP-based local search for MAPF on directed graphs (C++ implementation
// of mapf_bench/local_search.py; same neighborhoods, same tie-breaking, so the
// two implementations return identical plans).
//
// Input file:
//   N M            nodes, arcs (self-loops are added automatically)
//   u v            M lines, directed arcs
//   n              number of agents
//   s_1 ... s_n    starts
//   g_1 ... g_n    goals
//   T              joint length of the initial plan
//   T+1 lines with n node ids: the configurations x^0..x^T
//
// Usage:
//   dpls input.txt output.txt [--nb SPEC]... [--tiebreak summin|none]
//        [--time SECONDS] [--expansions K] [--max-searches K]
//        [--objective makespan|soc|soc_proxy] [--vnd] [--keep-radius] [--astar]
// With --astar each neighborhood search is A* with a consistent heuristic
// (distances to the goals, ignoring collisions): same optimal cost within
// the neighborhood, fewer states, possibly a different optimal plan.
// The neighborhoods are searched in turn until none improves the plan; with
// --vnd the search restarts from the first neighborhood after every
// improvement (variable neighborhood descent).
// Adaptive radius: a SPEC of the form TEMPLATE@RMIN-RMAX, where TEMPLATE
// contains "{R}" (e.g. gc:{R}:1:none:1@1-8), starts at radius RMIN; when a
// search ends without improvement and within its budget, it is repeated with
// the radius increased by 1, up to RMAX. After every improvement all radii
// return to their minimum. A SPEC may end with #K to give that neighborhood
// its own state budget K (e.g. window:1#200000) (with --keep-radius, every neighborhood keeps the
// radius at which it last improved).
// With --objective soc every agent carries a counter of consecutive waits at
// its goal (Appendix A.3): waiting at the goal costs 0, leaving it costs the
// counter + 1, any other action costs 1, so a plan costs its Sum-of-Costs.
// With --objective soc_proxy there are no counters: waiting at the goal costs
// 0 and any other action 1 (a lower bound on the Sum-of-Costs, exact unless an
// agent leaves its goal after waiting there); a plan is accepted only if its
// true Sum-of-Costs decreases.
// Neighborhood SPECs:
//   path:R  pointwise:R  agent:R  combined:R_ag:R_sm  window:W[:K[:multi]]
//   retime   (same routes and visit orders, earliest schedule; not DP)
//   swap     (exchange two consecutive visits of a node, or a pair of agents
//             at every node where one follows the other; then retime)
//   gc:R:lambda:GROUPS:RESTRICTED[:WINDOW:MAXGROUPS[:K:I]]
//      K > 0: the I-th candidate free set of K agents moves freely (no
//      distance term); the other agents keep the route-lag constraint;
//      gc:...:K:I:C restricts the free agents to the nodes within distance C
//      of their reference routes (corridors)
//      GROUPS = none (single clock = route-lag) | auto (interaction groups)
//      RESTRICTED = 0 | 1 (route-restricted variant)
// The output file contains the final plan (T'+1 lines); a summary line
// with statistics is printed on stdout.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <iostream>
#include <limits>
#include <array>
#include <map>
#include <tuple>
#include <set>
#include <memory>
#include <queue>
#include <deque>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
typedef vector<int> Config;
static const long long BIG = 1000000;
// --region-h: route-lag and grouped-clock neighborhoods give the A* heuristic the
// distances to the goals inside each agent's region (see GroupedClock::goal_dist)
static bool g_region_h = false;
// --conflict-h: add to the Sum-of-Costs heuristic a lower bound on the extra cost
// forced by agents resting on their goals (see ConflictBound)
static bool g_conflict_h = false;
// statistics of the A* searches: expansions below and at the optimal key of
// the searches that found a plan; for those that ran out of budget, the gap
// between the incumbent and the last key reached
static atomic<long long> g_exp_below{0}, g_exp_at{0}, g_found{0}, g_budget_n{0}, g_budget_gap{0};
// --threads T: the candidates of a cyclic neighborhood (free sets) are searched in
// batches of T in parallel; the best improvement of the batch is accepted
static int g_threads = 1;
static mutex g_build_mutex;           // lazy construction of shared heuristic tables
static const double INF = numeric_limits<double>::infinity();
typedef chrono::steady_clock Clock;

// ---------------------------------------------------------------------------
struct Graph {
    int N;
    vector<vector<int>> out, inn;
    vector<vector<long long>> dV;  // min(max(dist(u,v), dist(v,u)), BIG)
    void finish() {
        for (int u = 0; u < N; u++) { out[u].push_back(u); inn[u].push_back(u); }
        for (int u = 0; u < N; u++) {
            sort(out[u].begin(), out[u].end());
            out[u].erase(unique(out[u].begin(), out[u].end()), out[u].end());
            sort(inn[u].begin(), inn[u].end());
            inn[u].erase(unique(inn[u].begin(), inn[u].end()), inn[u].end());
        }
        vector<vector<long long>> dist(N, vector<long long>(N, BIG));
        for (int s = 0; s < N; s++) {
            vector<int> q{s};
            dist[s][s] = 0;
            for (size_t h = 0; h < q.size(); h++)
                for (int v : out[q[h]])
                    if (dist[s][v] == BIG) { dist[s][v] = dist[s][q[h]] + 1; q.push_back(v); }
        }
        dV.assign(N, vector<long long>(N, BIG));
        for (int u = 0; u < N; u++)
            for (int v = 0; v < N; v++)
                dV[u][v] = min(BIG, max(dist[u][v], dist[v][u]));
    }
};

static vector<Config> trim(vector<Config> p) {
    while (p.size() > 1 && p.back() == p[p.size() - 2]) p.pop_back();
    return p;
}
static int makespan(const vector<Config>& p) { return (int)trim(p).size() - 1; }
static long long sum_of_costs(const vector<Config>& p, const Config& goals) {
    long long s = 0;
    for (size_t k = 0; k < goals.size(); k++) {
        int a = 0;
        for (int t = (int)p.size() - 1; t >= 0; t--) if (p[t][k] != goals[k]) { a = t + 1; break; }
        s += a;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Search state: z (n), y (i entries), aux (neighborhood dependent)
struct State {
    Config z, y;
    vector<long long> aux;
    Config cnt;   // Sum-of-Costs only: consecutive waits at the goal (Appendix A.3)
    bool operator==(const State& o) const {
        return z == o.z && y == o.y && aux == o.aux && cnt == o.cnt;
    }
};
struct StateHash {
    size_t operator()(const State& s) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&](long long v) { h ^= (size_t)v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        for (int v : s.z) mix(v);
        mix(-1);
        for (int v : s.y) mix(v);
        mix(-2);
        for (long long v : s.aux) mix(v);
        mix(-3);
        for (int v : s.cnt) mix(v);
        return h;
    }
};

// ---------------------------------------------------------------------------
struct Neighborhood {
    const Graph* G = nullptr;
    const vector<Config>* ref = nullptr;
    int n = 0, T = 0;
    virtual ~Neighborhood() {}
    // memory of the per-agent tables that prepare() would allocate (bytes)
    virtual size_t table_bytes(const Graph& g, const vector<Config>& r) const { return 0; }
    virtual void prepare(const Graph& g, const vector<Config>& r) {
        G = &g; ref = &r; n = (int)r[0].size(); T = (int)r.size() - 1;
    }
    bool z_unchanged = false;     // set by the search: same z as the previous expand (chain)
    virtual vector<long long> init_aux() { return {}; }
    // called on the expanded state before its successors are generated
    virtual void expand(const State& s) {}
    // returns false if the new state is outside the neighborhood
    virtual bool step(const State& s, int agent, int w, const Config& z2,
                      const Config& y2, bool done, vector<long long>& aux2) = 0;
    virtual bool marked(const State& s) { return true; }
    const Config& ref_at(int t) const { return (*ref)[min(t, T)]; }
    // A* hooks: per-agent distances to the goal valid inside the neighborhood
    // (nullptr = graph distances), and a lower bound on the number of joint
    // steps still needed that depends on the auxiliary state (0 = none).
    virtual const vector<vector<int>>* goal_dist() const { return nullptr; }
    virtual long long rounds_lb(const vector<long long>& aux) const { return 0; }
};

// ---- anchored sums: path, route-lag, route-restricted, grouped clocks -----
static vector<vector<int>> interaction_groups(const vector<Config>& ref, int window, int max_groups) {
    int n = ref[0].size();
    vector<int> parent(n);
    for (int k = 0; k < n; k++) parent[k] = k;
    function<int(int)> find = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    map<int, vector<pair<int, int>>> occ;
    for (int t = 0; t < (int)ref.size(); t++)
        for (int k = 0; k < n; k++) occ[ref[t][k]].push_back({t, k});
    for (auto& kv : occ) {
        auto& lst = kv.second;
        sort(lst.begin(), lst.end());
        for (size_t a = 0; a < lst.size(); a++)
            for (size_t b = a + 1; b < lst.size(); b++) {
                if (lst[b].first - lst[a].first > window) break;
                int ra = find(lst[a].second), rb = find(lst[b].second);
                if (ra != rb) parent[ra] = rb;
            }
    }
    vector<int> order;            // roots in order of first member
    map<int, vector<int>> groups;
    for (int k = 0; k < n; k++) {
        int r = find(k);
        if (!groups.count(r)) order.push_back(r);
        groups[r].push_back(k);
    }
    vector<vector<int>> gs;
    for (int r : order) gs.push_back(groups[r]);
    stable_sort(gs.begin(), gs.end(), [](const vector<int>& a, const vector<int>& b) { return a.size() < b.size(); });
    while ((int)gs.size() > max_groups) {
        vector<int> a = gs.front();
        gs.erase(gs.begin());
        gs[0].insert(gs[0].end(), a.begin(), a.end());
        sort(gs[0].begin(), gs[0].end());
        stable_sort(gs.begin(), gs.end(), [](const vector<int>& x, const vector<int>& y) { return x.size() < y.size(); });
    }
    for (auto& g : gs) sort(g.begin(), g.end());
    return gs;
}

struct AnchoredSum : Neighborhood {
    double R = 0;
    bool no_moves = false;   // set when a free-set candidate index is out of range
    vector<vector<int>> groups;
    vector<int> grp;
    vector<double> PSI;  // [k][v][s], s = 0..T+1
    int S = 0;
    // incremental data of the expanded state
    vector<vector<double>> Zv, Yv;
    vector<double> mins;
    double psi(int k, int v, int s) const { return PSI[((size_t)k * G->N + v) * S + s]; }
    size_t table_bytes(const Graph& g, const vector<Config>& r) const override {
        return (size_t)r[0].size() * g.N * (r.size() + 1) * sizeof(double);
    }
    void set_groups(const string& spec, int window, int max_groups) {
        if (spec == "auto") groups = interaction_groups(*ref, window, max_groups);
        else { groups.assign(1, vector<int>()); for (int k = 0; k < n; k++) groups[0].push_back(k); }
        grp.assign(n, 0);
        for (size_t g = 0; g < groups.size(); g++) for (int k : groups[g]) grp[k] = g;
    }
    // Exact restriction to the relevant times. Every term psi is nonnegative
    // and a state is accepted only if the total is <= R, so a time s at which
    // some term exceeds R can never make a state acceptable. For every (agent,
    // node) we keep the smallest interval [rlo, rhi] containing the times with
    // psi <= R; outside it the term is treated as infinite. Group sums are
    // computed only on the intersection of the intervals of their agents (same
    // additions, in the same order, as on the full arrays), and the minima are
    // taken over those intersections: every decision is unchanged.
    vector<int> rlo, rhi;
    void compute_ranges() {
        rlo.assign((size_t)n * G->N, 1);
        rhi.assign((size_t)n * G->N, 0);
        const double lim = R + 1e-9;
        for (size_t kv = 0; kv < (size_t)n * G->N; kv++) {
            const double* row = &PSI[kv * S];
            int lo = -1, hi = -1;
            for (int t = 0; t < S; t++) if (row[t] <= lim) { if (lo < 0) lo = t; hi = t; }
            if (lo >= 0) { rlo[kv] = lo; rhi[kv] = hi; }
        }
    }
    vector<int> YL, YH;             // valid intervals of the group sums (L > H: empty)
    vector<vector<double>> ZY;      // ZY[g][t] = Z[g][t] + Y[g][t + 1] on [ZYL[g], ZYH[g]]
    vector<int> ZYL, ZYH;
    struct ZEntry { Config z; vector<vector<double>> Zv; vector<int> L, H; };
    vector<ZEntry> zcache;
    const ZEntry* zc = nullptr;     // cache entry of the expanded state's z
    // Sums over the decided agents, cached by the sequence y of decisions: the
    // sums of y = (y_1, ..., y_i) are those of (y_1, ..., y_{i-1}) plus the row
    // of agent i, added in the same order as a full recomputation.
    struct YEntry { Config y; vector<vector<double>> V; vector<int> L, H; bool ok = false; };
    vector<YEntry> ycache;
    const YEntry* yc = nullptr;
    void reset_cache() {
        zcache.assign(1 << 12, ZEntry()); zc = nullptr;
        ycache.assign(1 << 13, YEntry()); yc = nullptr;
    }
    static uint64_t hash_step(uint64_t h, int x) { h ^= (uint64_t)x + 0x9E37U; return h * 1099511628211ULL; }
    static uint64_t hash_seq(const Config& x, size_t len) {
        uint64_t h = 1469598103934665603ULL;
        for (size_t k = 0; k < len; k++) h = hash_step(h, x[k]);
        return h;
    }
    uint64_t y_last_hash = 0; size_t y_last_len = (size_t)-1;   // hash of the previous y (chains)
    void y_sums(const Config& y) {
        size_t i = y.size(), mask = ycache.size() - 1;
        // within a chain y extends the previous one by one decision
        bool ext = z_unchanged && i >= 1 && y_last_len == i - 1;
        uint64_t hp = ext ? y_last_hash : (i >= 1 ? hash_seq(y, i - 1) : 0);
        uint64_t hy = i >= 1 ? hash_step(i >= 1 ? hp : 0, y[i - 1]) : hash_seq(y, 0);
        y_last_hash = hy; y_last_len = i;
        YEntry& ye = ycache[hy & mask];
        if (ye.ok && ye.y == y) { yc = &ye; return; }
        bool done = false;
        if (i > 0) {
            YEntry& pe = ycache[hp & mask];
            if (&pe != &ye && pe.ok && pe.y.size() == i - 1 && equal(pe.y.begin(), pe.y.end(), y.begin())) {
                int k = (int)i - 1, g = grp[k];
                size_t kv = (size_t)k * G->N + y[k];
                ye.V = pe.V; ye.L = pe.L; ye.H = pe.H;           // other groups unchanged
                ye.L[g] = max(ye.L[g], rlo[kv]); ye.H[g] = min(ye.H[g], rhi[kv]);
                if (ye.L[g] <= ye.H[g]) {
                    const double* row = &PSI[kv * S];
                    double* Vg = ye.V[g].data();
                    for (int s = ye.L[g]; s <= ye.H[g]; s++) Vg[s] += row[s];
                }
                done = true;
            }
        }
        if (!done) group_sums(y, (int)i, ye.V, ye.L, ye.H, 0);
        ye.y = y; ye.ok = true;
        yc = &ye;
    }
    template <class Sel>
    void group_sums(const Config& x, int count, vector<vector<double>>& V, vector<int>& L, vector<int>& H, Sel) {
        size_t G_ = groups.size();
        V.resize(G_); L.assign(G_, 0); H.assign(G_, S - 1);
        for (int k = 0; k < count; k++) {
            size_t kv = (size_t)k * G->N + x[k];
            int g = grp[k];
            L[g] = max(L[g], rlo[kv]); H[g] = min(H[g], rhi[kv]);
        }
        for (size_t g = 0; g < G_; g++) {
            if ((int)V[g].size() != S) V[g].assign(S, 0.0);
            if (L[g] <= H[g]) fill(V[g].begin() + L[g], V[g].begin() + H[g] + 1, 0.0);
        }
        for (int k = 0; k < count; k++) {
            int g = grp[k];
            if (L[g] > H[g]) continue;
            const double* row = &PSI[((size_t)k * G->N + x[k]) * S];
            double* Vg = V[g].data();
            for (int s = L[g]; s <= H[g]; s++) Vg[s] += row[s];
        }
    }
    void expand(const State& st) override {
        size_t G_ = groups.size();
        if (zcache.empty()) { reset_cache(); compute_ranges(); }
        if (!(z_unchanged && zc)) {
            uint64_t h = 1469598103934665603ULL;
            for (int v : st.z) { h ^= (uint64_t)v; h *= 1099511628211ULL; }
            ZEntry& ze = zcache[h & (zcache.size() - 1)];
            if (!(ze.z == st.z && !ze.Zv.empty())) {      // compute in place (no copies)
                group_sums(st.z, n, ze.Zv, ze.L, ze.H, 0);
                ze.z = st.z;
            }
            zc = &ze;                                   // valid until the next expand
        }
        y_sums(st.y);
        mins.assign(G_, INF);
        ZY.resize(G_); ZYL.assign(G_, 1); ZYH.assign(G_, 0);
        for (size_t g = 0; g < G_; g++) {
            // min over s of Z[s] + Y[s + 1], s in Z's interval and s + 1 in Y's;
            // the sums are kept for step() (same additions, same order)
            int lo = max(zc->L[g], yc->L[g] - 1), hi = min(min(zc->H[g], yc->H[g] - 1), T);
            double m = INF;
            const double* Zg = zc->Zv[g].data(); const double* Yg = yc->V[g].data();
            if ((int)ZY[g].size() != S) ZY[g].assign(S, 0.0);
            double* zy = ZY[g].data();
            for (int s = lo; s <= hi; s++) { zy[s] = Zg[s] + Yg[s + 1]; m = min(m, zy[s]); }
            ZYL[g] = lo; ZYH[g] = hi;
            mins[g] = m;
        }
    }
    bool step(const State& s, int agent, int w, const Config& z2, const Config& y2,
              bool done, vector<long long>& aux2) override {
        if (no_moves) return false;
        int g = grp[agent];
        size_t kv = (size_t)agent * G->N + w;
        int alo = rlo[kv], ahi = rhi[kv];
        const double* row = &PSI[kv * S];
        double total = 0;
        if (done) {
            for (size_t h = 0; h < groups.size(); h++) {
                int lo = yc->L[h], hi = min(yc->H[h], T);
                if (h == (size_t)g) { lo = max(lo, alo); hi = min(hi, ahi); }
                double m = INF;
                const double* Yh = yc->V[h].data();
                if (h == (size_t)g) for (int t = lo; t <= hi; t++) m = min(m, Yh[t] + row[t]);
                else for (int t = lo; t <= hi; t++) m = min(m, Yh[t]);
                total += m;
                if (total > R + 1e-9) return false;
            }
        } else {
            for (size_t h = 0; h < groups.size(); h++) {
                if ((int)h != g) { total += mins[h]; if (total > R + 1e-9) return false; continue; }
                // t in Z's interval, t + 1 in Y's and in the agent's interval
                // t in Z's interval, t + 1 in Y's and in the agent's interval:
                // the intersection of ZY's interval with [alo - 1, ahi - 1]
                int lo = max(ZYL[h], alo - 1), hi = min(ZYH[h], ahi - 1);
                double m = INF;
                const double* zy = ZY[h].data();
                for (int t = lo; t <= hi; t++) m = min(m, zy[t] + row[t + 1]);
                total += m;
                if (total > R + 1e-9) return false;
            }
        }
        return total <= R + 1e-9;
    }
};

struct PathDistance : AnchoredSum {
    PathDistance(double r) { R = r; }
    void prepare(const Graph& g, const vector<Config>& r) override {
        Neighborhood::prepare(g, r);
        zcache.clear();
        set_groups("none", 0, 1);
        S = T + 2;
        PSI.assign((size_t)n * G->N * S, INF);
        for (int k = 0; k < n; k++)
            for (int v = 0; v < G->N; v++)
                for (int s = 0; s <= T; s++)
                    PSI[((size_t)k * G->N + v) * S + s] = (double)G->dV[v][r[s][k]];
    }
};

struct GroupedClock : AnchoredSum {
    double lam; string gspec; bool restricted; int window, max_groups;
    vector<vector<int>> route_dist;   // restricted variant: distances inside the route
    // Free set S (destroy-set neighborhood): the agents of S have no term in the
    // distance and move freely; S is the cand_idx-th candidate of size K_free
    // computed from the reference (see free_candidates). K_free = 0: no free set.
    int K_free = 0, cand_idx = 0;
    int corridor = -1;        // free agents restricted to nodes within this distance
                              // of their reference route (-1: anywhere)
    bool empty_set = false;
    vector<char> is_free;
    GroupedClock(double r, double l, string gs, bool rs, int w, int mg, int kf = 0, int ci = 0,
                 int cor = -1)
        : lam(l), gspec(gs), restricted(rs), window(w), max_groups(mg), K_free(kf), cand_idx(ci),
          corridor(cor) { R = r; }
    // Candidate free sets: for every agent a, by decreasing delay (arrival time in
    // the reference minus shortest-path length), {a} plus the K-1 agents that
    // occupy the same nodes as a at the closest times; then the K most delayed
    // agents through the node visited by most agents. Duplicates removed.
    vector<vector<int>> free_candidates(const vector<Config>& r) const {
        int K = min(K_free, n);
        vector<vector<int>> dist_to_goal(n);
        vector<int> delay(n);
        for (int k = 0; k < n; k++) {
            int goal = r[T][k], arr = 0;
            for (int t = 0; t <= T; t++) if (r[t][k] != goal) arr = t + 1;
            vector<int> d(G->N, INT_MAX / 4);
            vector<int> q{goal};
            d[goal] = 0;
            for (size_t h = 0; h < q.size(); h++)
                for (int u : G->inn[q[h]]) if (d[u] == INT_MAX / 4) { d[u] = d[q[h]] + 1; q.push_back(u); }
            delay[k] = arr - d[r[0][k]];
        }
        // interaction: min |t_a - t_b| over common nodes (smaller = stronger)
        map<int, vector<pair<int, int>>> visits;
        for (int t = 0; t <= T; t++) for (int k = 0; k < n; k++) visits[r[t][k]].push_back({k, t});
        vector<vector<int>> close(n, vector<int>(n, INT_MAX));
        vector<int> nodes_by_agents;
        int best_node = -1, best_cnt = -1;
        for (auto& kv : visits) {
            auto& L = kv.second;
            set<int> ag;
            for (auto& p : L) ag.insert(p.first);
            if ((int)ag.size() > best_cnt) { best_cnt = ag.size(); best_node = kv.first; }
            for (size_t a = 0; a < L.size(); a++)
                for (size_t b = 0; b < L.size(); b++)
                    if (L[a].first != L[b].first)
                        close[L[a].first][L[b].first] = min(close[L[a].first][L[b].first],
                                                            abs(L[a].second - L[b].second));
        }
        vector<int> order(n);
        for (int k = 0; k < n; k++) order[k] = k;
        stable_sort(order.begin(), order.end(), [&](int a, int b) { return delay[a] > delay[b]; });
        vector<vector<int>> cands;
        set<vector<int>> seen;
        auto add = [&](vector<int> S) {
            sort(S.begin(), S.end());
            if (seen.insert(S).second) cands.push_back(S);
        };
        for (int a : order) {
            vector<int> others;
            for (int b = 0; b < n; b++) if (b != a) others.push_back(b);
            stable_sort(others.begin(), others.end(), [&](int x, int y) {
                if (close[a][x] != close[a][y]) return close[a][x] < close[a][y];
                return delay[x] > delay[y];
            });
            vector<int> S{a};
            for (int j = 0; j < K - 1 && j < (int)others.size(); j++) S.push_back(others[j]);
            add(S);
        }
        if (K < 4) {                       // small free sets: as before
            if (best_node >= 0) {
                vector<int> through;
                for (int a : order)
                    for (auto& p : visits[best_node]) if (p.first == a) { through.push_back(a); break; }
                vector<int> S(through.begin(), through.begin() + min<size_t>(K, through.size()));
                for (int a : order) if ((int)S.size() < K && find(S.begin(), S.end(), a) == S.end()) S.push_back(a);
                add(S);
            }
            return cands;
        }
        // Larger free sets (LNS-style destroy heuristics): the agent-based sets
        // above, intersection-based sets (the most delayed agents through each
        // of the 5 nodes visited by most agents) and random sets (fixed seed),
        // interleaved as MAPF-LNS alternates its heuristics
        vector<vector<int>> agent_sets = cands, inter_sets, rand_sets;
        vector<pair<int, int>> crowd;
        for (auto& kv : visits) {
            set<int> ag;
            for (auto& p : kv.second) ag.insert(p.first);
            crowd.push_back({-(int)ag.size(), kv.first});
        }
        sort(crowd.begin(), crowd.end());
        for (size_t c = 0; c < crowd.size() && c < 5; c++) {
            int node = crowd[c].second;
            vector<int> S;
            for (int a : order) {
                if ((int)S.size() >= K) break;
                for (auto& p : visits[node]) if (p.first == a) { S.push_back(a); break; }
            }
            for (int a : order) if ((int)S.size() < K && find(S.begin(), S.end(), a) == S.end()) S.push_back(a);
            sort(S.begin(), S.end());
            inter_sets.push_back(S);
        }
        mt19937 rng(12345 + T * 31 + n);
        for (int c = 0; c < 5; c++) {
            vector<int> all(n);
            for (int k = 0; k < n; k++) all[k] = k;
            shuffle(all.begin(), all.end(), rng);
            vector<int> S(all.begin(), all.begin() + K);
            sort(S.begin(), S.end());
            rand_sets.push_back(S);
        }
        cands.clear(); seen.clear();
        for (size_t j = 0; j < max({agent_sets.size(), inter_sets.size(), rand_sets.size()}); j++) {
            if (j < agent_sets.size()) add(agent_sets[j]);
            if (j < inter_sets.size()) add(inter_sets[j]);
            if (j < rand_sets.size()) add(rand_sets[j]);
        }
        return cands;
    }
    // Region of agent k: the nodes v with psi_k(v, s) <= R for some s. Every
    // term of the distance is nonnegative and their sum is at most R, so in
    // every plan of the neighborhood each agent stays in its region; the
    // distance to the goal inside the region is a consistent lower bound on its
    // remaining cost, tighter than the distance in the whole graph.
    mutable vector<vector<int>> region_dist;
    mutable bool region_ready = false;
    const vector<vector<int>>* goal_dist() const override {
        if (restricted || !route_dist.empty()) return &route_dist;
        if (!g_region_h) return nullptr;
        if (!region_ready) {
            region_dist.assign(n, vector<int>(G->N, INT_MAX / 4));
            vector<char> in(G->N);
            const double lim = R + 1e-9;
            for (int k = 0; k < n; k++) {
                for (int v = 0; v < G->N; v++) {
                    const double* row = &PSI[((size_t)k * G->N + v) * S];
                    char ok = 0;
                    for (int t = 0; t < S && !ok; t++) if (row[t] <= lim) ok = 1;
                    in[v] = ok;
                }
                vector<int>& d = region_dist[k];
                int goal = (*ref)[T][k];
                vector<int> q{goal};
                d[goal] = 0;
                for (size_t h = 0; h < q.size(); h++)
                    for (int u : G->inn[q[h]])
                        if (in[u] && d[u] == INT_MAX / 4) { d[u] = d[q[h]] + 1; q.push_back(u); }
            }
            region_ready = true;
        }
        return &region_dist;
    }
    void prepare(const Graph& g, const vector<Config>& r) override {
        Neighborhood::prepare(g, r);
        region_ready = false;
        zcache.clear();
        set_groups(gspec, window, max_groups);
        S = T + 2;
        PSI.assign((size_t)n * G->N * S, INF);
        is_free.assign(n, 0);
        empty_set = false;
        no_moves = false;
        if (K_free > 0) {
            auto cands = free_candidates(r);
            if (cand_idx >= (int)cands.size()) { empty_set = true; no_moves = true; }
            else for (int k : cands[cand_idx]) is_free[k] = 1;
        }
        // routes and progress
        int W = (int)floor(R / lam + 1e-9) + 1;   // indices beyond W steps only give values > R
        if (restricted) route_dist.assign(n, vector<int>(G->N, INT_MAX / 4));
        for (int k = 0; k < n; k++) {
            vector<int> pi{r[0][k]}, rk{0};
            for (int t = 1; t <= T; t++) { if (r[t][k] != r[t - 1][k]) pi.push_back(r[t][k]); rk.push_back(pi.size() - 1); }
            if (restricted) {   // BFS to the goal inside the subgraph induced by the route
                vector<char> on(G->N, 0);
                for (int v : pi) on[v] = 1;
                vector<int>& d = route_dist[k];
                int goal = r[T][k];
                vector<int> q{goal};
                d[goal] = 0;
                for (size_t h = 0; h < q.size(); h++)
                    for (int u : G->inn[q[h]])
                        if (on[u] && d[u] == INT_MAX / 4) { d[u] = d[q[h]] + 1; q.push_back(u); }
            }
            int m = pi.size();
            for (int s = 0; s < S; s++) {
                int rs = rk[min(s, T)];
                int j0 = max(0, rs - W), j1 = min(m - 1, rs + W);
                for (int v = 0; v < G->N; v++) {
                    double best = INF;
                    for (int j = j0; j <= j1; j++) {
                        double d = restricted ? (v == pi[j] ? 0.0 : INF) : (double)G->dV[v][pi[j]];
                        best = min(best, d + lam * abs(j - rs));
                    }
                    if (!(best <= R)) best = (best == INF && restricted) ? INF : max(best, R + 1.0);
                    PSI[((size_t)k * G->N + v) * S + s] = best;
                }
            }
            if (is_free[k]) {               // free agent: no term in the distance
                // corridor: nodes within distance `corridor` of the route (d_V)
                vector<char> in(G->N, 1);
                if (corridor >= 0)
                    for (int v = 0; v < G->N; v++) {
                        long long dmin = LLONG_MAX;
                        for (int u : pi) dmin = min(dmin, G->dV[v][u]);
                        in[v] = dmin <= corridor;
                    }
                for (int v = 0; v < G->N; v++)
                    for (int s2 = 0; s2 < S; s2++)
                        PSI[((size_t)k * G->N + v) * S + s2] = in[v] ? 0.0 : INF;
                if (!route_dist.empty() || corridor >= 0) {   // A* distances inside the corridor
                    if (route_dist.empty()) route_dist.assign(n, vector<int>());
                    if (!restricted && route_dist[k].empty()) {
                        // non-restricted others use graph distances: fill them for all agents
                        for (int a = 0; a < n; a++) if (route_dist[a].empty()) {
                            route_dist[a].assign(G->N, INT_MAX / 4);
                            int ga = r[T][a];
                            vector<int> q{ga};
                            route_dist[a][ga] = 0;
                            for (size_t h = 0; h < q.size(); h++)
                                for (int u : G->inn[q[h]])
                                    if (route_dist[a][u] == INT_MAX / 4) { route_dist[a][u] = route_dist[a][q[h]] + 1; q.push_back(u); }
                        }
                    }
                    vector<int>& d = route_dist[k];
                    d.assign(G->N, INT_MAX / 4);
                    int goal = r[T][k];
                    vector<int> q{goal};
                    d[goal] = 0;
                    for (size_t h = 0; h < q.size(); h++)
                        for (int u : G->inn[q[h]])
                            if (in[u] && d[u] == INT_MAX / 4) { d[u] = d[q[h]] + 1; q.push_back(u); }
                }
            }
        }
    }
};

struct Pointwise : Neighborhood {
    long long R;
    Pointwise(long long r) : R(r) {}
    vector<long long> init_aux() override { return {0}; }
    bool step(const State& s, int agent, int w, const Config& z2, const Config& y2,
              bool done, vector<long long>& aux2) override {
        long long tau = s.aux[0] + (done ? 1 : 0);
        if (tau > T || (!y2.empty() && tau + 1 > T)) return false;
        long long val = 0;
        for (int k = 0; k < n; k++) val += G->dV[z2[k]][(*ref)[tau][k]];
        for (size_t k = 0; k < y2.size(); k++) val += G->dV[y2[k]][(*ref)[tau + 1][k]];
        aux2 = {tau};
        return val <= R;
    }
};

struct AgentDistance : Neighborhood {
    // At most R agents change their timed trajectory. aux = (c, d_1, ..., d_R):
    // c = joint step in progress, d_1 < ... < d_m the deviating agents, padded
    // with -1 (no limit on the number of agents; same states as a bit mask).
    int R;
    vector<int> tau;            // last time at which agent k is not at its final node
    vector<int> cntP;           // cntP[c] = |{k : tau_k >= c}|
    vector<int> by_tau;         // agents by decreasing tau
    AgentDistance(int r) : R(r) {}
    void prepare(const Graph& g, const vector<Config>& r) override {
        Neighborhood::prepare(g, r);
        tau.assign(n, 0);
        for (int k = 0; k < n; k++) for (int t = 0; t <= T; t++) if (r[t][k] != r[T][k]) tau[k] = t;
        cntP.assign(T + 2, 0);
        for (int c = 1; c <= T + 1; c++) for (int k = 0; k < n; k++) if (tau[k] >= c) cntP[c]++;
        by_tau.resize(n);
        for (int k = 0; k < n; k++) by_tau[k] = k;
        stable_sort(by_tau.begin(), by_tau.end(), [&](int a, int b) { return tau[a] > tau[b]; });
    }
    vector<long long> init_aux() override {
        vector<long long> a(1 + R, -1);
        a[0] = 1;
        return a;
    }
    static int size_of(const vector<long long>& aux) {
        int m = 0;
        for (size_t q = 1; q < aux.size(); q++) if (aux[q] >= 0) m++;
        return m;
    }
    static bool contains(const vector<long long>& aux, int k) {
        for (size_t q = 1; q < aux.size(); q++) if (aux[q] == k) return true;
        return false;
    }
    long long rounds_lb(const vector<long long>& aux) const override {
        if (size_of(aux) < R) return 0;
        long long c = aux[0];
        for (int k : by_tau)                     // most delayed agent outside D
            if (!contains(aux, k)) return max(0LL, (long long)tau[k] - (c - 1));
        return 0;
    }
    bool step(const State& s, int agent, int w, const Config& z2, const Config& y2,
              bool done, vector<long long>& aux2) override {
        long long c = s.aux[0];
        aux2 = s.aux;
        if (w != ref_at((int)c)[agent] && !contains(aux2, agent)) {
            int m = size_of(aux2);
            if (m + 1 > R) return false;
            aux2[1 + m] = agent;
            sort(aux2.begin() + 1, aux2.begin() + 2 + m);
        }
        if (done) aux2[0] = min<long long>(c + 1, T + 1);
        return true;
    }
    bool marked(const State& s) override {
        long long c = min<long long>(s.aux[0], T + 1);
        int cnt = cntP[c];
        for (size_t q = 1; q < s.aux.size(); q++)
            if (s.aux[q] >= 0 && tau[s.aux[q]] < c) cnt++;
        return cnt <= R;
    }
};
struct Window : Neighborhood {
    int W, K;
    bool multi;
    struct VHash {
        size_t operator()(const Config& z) const {
            size_t h = 1469598103934665603ULL;
            for (int v : z) { h ^= (size_t)v; h *= 1099511628211ULL; }
            return h;
        }
    };
    unordered_map<Config, int, VHash> latest;          // configuration -> latest time
    vector<unordered_map<uint64_t, vector<int>>> pref; // prefix length -> hash -> times
    Window(int w, int k, bool m) : W(w), K(k), multi(m) {}
    static uint64_t phash(const Config& y, size_t len) {
        uint64_t h = 1469598103934665603ULL;
        for (size_t k = 0; k < len; k++) { h ^= (uint64_t)y[k]; h *= 1099511628211ULL; }
        return h;
    }
    void prepare(const Graph& g, const vector<Config>& r) override {
        Neighborhood::prepare(g, r);
        latest.clear();
        for (int t = 0; t <= T; t++) latest[r[t]] = t;
        pref.assign(n + 1, {});
        for (int t = 0; t <= T; t++)
            for (int i = 1; i <= n; i++) pref[i][phash(r[t], i)].push_back(t);
    }
    bool is_prefix(const Config& y) const {
        auto it = pref[y.size()].find(phash(y, y.size()));
        if (it == pref[y.size()].end()) return false;
        for (int t : it->second)
            if (equal(y.begin(), y.end(), (*ref)[t].begin())) return true;
        return false;
    }
    vector<long long> init_aux() override { return {1, 0, 0, 0, 0}; }
    bool step(const State& s, int agent, int w, const Config& z2, const Config& y2,
              bool done, vector<long long>& aux2) override {
        long long c = s.aux[0], f = s.aux[1], used = s.aux[2], t0 = s.aux[3];
        uint64_t mask = (uint64_t)s.aux[4];
        if (f == 0) {
            if (w != ref_at((int)c)[agent]) {            // the window opens
                if (used && !multi) return false;
                f = 1;
                if (K >= 0) {
                    t0 = c - 1;                          // the round started at x^{c-1}
                    const Config& x0 = ref_at((int)t0);
                    for (int k = 0; k < agent; k++) if (s.y[k] != x0[k]) mask |= 1ULL << k;
                }
                c = 0;
            }
        }
        if (f > 0 && K >= 0) {
            if (w != ref_at((int)t0)[agent]) mask |= 1ULL << agent;
            if (__builtin_popcountll(mask) > K) return false;
        }
        if (f > 0 && f == W && !done && !is_prefix(y2)) return false;   // last free round
        if (done) {
            if (f == 0) c = min<long long>(c + 1, T + 1);
            else {
                if (f > W) return false;
                auto it = latest.find(z2);
                if (it != latest.end()) {                // back on the reference
                    c = min(it->second + 1, T + 1); f = 0; used = 1; t0 = 0; mask = 0;
                }
                else if (f == W) return false;           // window exhausted
                else f++;
            }
        }
        aux2 = {c, f, used, t0, (long long)mask};
        return true;
    }
};

struct SumMin {
    vector<vector<long long>> mu;
    void prepare(const Graph& G, const vector<Config>& ref) {
        int n = ref[0].size();
        mu.assign(n, vector<long long>(G.N, BIG));
        // mu[k][v] = min over the nodes w of agent k's reference of dV(v, w):
        // the minimum is taken over those nodes only (same values)
        vector<char> in(G.N, 0);
        vector<int> nodes;
        for (int k = 0; k < n; k++) {
            nodes.clear();
            for (auto& x : ref) if (!in[x[k]]) { in[x[k]] = 1; nodes.push_back(x[k]); }
            for (int v = 0; v < G.N; v++) {
                long long m = mu[k][v];
                for (int w : nodes) m = min(m, (long long)G.dV[v][w]);
                mu[k][v] = m;
            }
            for (int w : nodes) in[w] = 0;
        }
    }
    long long value(const Config& z, const Config& y) const {
        long long s = 0;
        for (size_t k = 0; k < z.size(); k++) s += mu[k][k < y.size() ? y[k] : z[k]];
        return s;
    }
};

struct Combined : AgentDistance {
    long long Rsm; SumMin sm;
    Combined(int rag, long long rsm) : AgentDistance(rag), Rsm(rsm) {}
    void prepare(const Graph& g, const vector<Config>& r) override {
        AgentDistance::prepare(g, r);
        sm.prepare(g, r);
    }
    bool step(const State& s, int agent, int w, const Config& z2, const Config& y2,
              bool done, vector<long long>& aux2) override {
        if (sm.value(z2, y2) > Rsm) return false;
        return AgentDistance::step(s, agent, w, z2, y2, done, aux2);
    }
};

// ---------------------------------------------------------------------------

// Compact state store: every state is a fixed-length record of 16-bit words
//   z[0..n) | len(y) | y[0..n) (padded) | aux (4 words per 64-bit value) | cnt[0..n) (SoC only)
// kept in one contiguous array, with an open-addressing hash table of state
// ids. This avoids the per-state vectors and copies of the straightforward
// implementation (same search, same order, same tie-breaking).
struct StateStore {
    int n = 0, A = 0, K = 0;
    bool with_cnt = false;
    vector<uint16_t> pool;
    vector<uint64_t> hashes;
    vector<uint64_t> table;   // (hash tag << 32) | (id + 1); 0 = empty
    size_t mask = 0, count = 0;
    void init(int n_, int A_, bool cnt, size_t reserve) {
        n = n_; A = A_; with_cnt = cnt;
        K = 2 * n + 1 + 4 * A + (cnt ? n : 0);
        pool.clear(); hashes.clear(); count = 0;
        pool.reserve(reserve * K); hashes.reserve(reserve);
        size_t cap = 1024;
        while (cap < 2 * reserve) cap <<= 1;
        table.assign(cap, 0); mask = cap - 1;
    }
    static uint64_t hash_words(const uint16_t* w, int K) {
        uint64_t h = 0x9E3779B97F4A7C15ULL;
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            uint64_t v; memcpy(&v, w + k, 8);
            h ^= v; h *= 0xff51afd7ed558ccdULL; h ^= h >> 32;
        }
        for (; k < K; k++) { h ^= w[k]; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 29; }
        h ^= h >> 33;
        return h;
    }
    void encode(const Config& z, const Config& y, const vector<long long>& aux,
                const Config& cnt, uint16_t* w) const {
        for (int k = 0; k < n; k++) w[k] = (uint16_t)z[k];
        w[n] = (uint16_t)y.size();
        for (int k = 0; k < n; k++) w[n + 1 + k] = k < (int)y.size() ? (uint16_t)y[k] : 0xFFFF;
        uint16_t* a = w + 2 * n + 1;
        for (int q = 0; q < A; q++) {
            uint64_t v = (uint64_t)aux[q];
            for (int b = 0; b < 4; b++) a[4 * q + b] = (uint16_t)(v >> (16 * b));
        }
        if (with_cnt) for (int k = 0; k < n; k++) w[2 * n + 1 + 4 * A + k] = (uint16_t)cnt[k];
    }
    void decode(int id, State& s) const {
        const uint16_t* w = &pool[(size_t)id * K];
        s.z.resize(n);
        for (int k = 0; k < n; k++) s.z[k] = w[k];
        s.y.resize(w[n]);
        for (int k = 0; k < (int)w[n]; k++) s.y[k] = w[n + 1 + k];
        s.aux.resize(A);
        const uint16_t* a = w + 2 * n + 1;
        for (int q = 0; q < A; q++) {
            uint64_t v = 0;
            for (int b = 0; b < 4; b++) v |= (uint64_t)a[4 * q + b] << (16 * b);
            s.aux[q] = (long long)v;
        }
        if (with_cnt) { s.cnt.resize(n); for (int k = 0; k < n; k++) s.cnt[k] = w[2 * n + 1 + 4 * A + k]; }
        else s.cnt.clear();
    }
    static uint64_t entry(uint64_t h, size_t id) { return (h & 0xFFFFFFFF00000000ULL) | (uint64_t)(id + 1); }
    void grow() {
        size_t cap = table.size() * 2;
        table.assign(cap, 0); mask = cap - 1;
        for (size_t id = 0; id < count; id++) {
            size_t p = hashes[id] & mask;
            while (table[p]) p = (p + 1) & mask;
            table[p] = entry(hashes[id], id);
        }
    }
    // returns (id, inserted)
    void prefetch(uint64_t h) const { __builtin_prefetch(&table[h & mask]); }
    pair<int, bool> find_or_insert(const uint16_t* w) { return find_or_insert_h(w, hash_words(w, K)); }
    // the same with the hash already computed (successors are hashed and their
    // table slots prefetched in a batch before being looked up)
    pair<int, bool> find_or_insert_h(const uint16_t* w, uint64_t h) {
        size_t p = h & mask;
        const uint64_t tag = h & 0xFFFFFFFF00000000ULL;
        while (uint64_t e = table[p]) {
            if ((e & 0xFFFFFFFF00000000ULL) == tag) {
                int id = (int)(e & 0xFFFFFFFFULL) - 1;
                if (memcmp(&pool[(size_t)id * K], w, K * sizeof(uint16_t)) == 0) return {id, false};
            }
            p = (p + 1) & mask;
        }
        int id = (int)count++;
        pool.insert(pool.end(), w, w + K);
        hashes.push_back(h);
        table[p] = entry(h, id);
        if (2 * count > table.size()) grow();
        return {id, true};
    }
};

struct SearchResult {
    bool found = false, budget = false;
    bool pruned = false;          // the open list was truncated (beam search): not exact
    vector<Config> plan;
    long long expansions = 0, states = 0;
};
// ---------------------------------------------------------------------------
// Order-based neighborhoods (not dynamic programming).
// Every agent keeps its route (its nodes, waits removed) and every node keeps
// the order of its visits; the earliest schedule compatible with
//   e_k(j) >= e_k(j-1) + 1                      (one step per move)
//   e_b(j_b) >= e_a(j_a + 1)                    (b enters v when a has left it,
//                                                for consecutive visits of v)
// is computed by longest paths (Bellman-Ford with zero-weight cycles, i.e.
// rotations); every arrival time is minimal, so makespan and Sum-of-Costs are
// optimal among the plans with the same routes and orders. Order swaps
// exchange two consecutive visits of a node, or of a pair of agents at every
// node where one follows the other, and reschedule.
struct OrderModel {
    int n = 0;
    vector<vector<int>> route;                    // route[k][j]
    vector<int> offset;                           // event index of (k, 0)
    int E = 0;
    vector<vector<pair<int, int>>> visits;        // visits[v] = [(k, j)] in order
    void build(const vector<Config>& plan, int N) {
        n = plan[0].size();
        int T = plan.size() - 1;
        route.assign(n, {});
        vector<vector<int>> entry(n);
        for (int k = 0; k < n; k++) {
            route[k].push_back(plan[0][k]);
            entry[k].push_back(0);
            for (int t = 1; t <= T; t++)
                if (plan[t][k] != plan[t - 1][k]) { route[k].push_back(plan[t][k]); entry[k].push_back(t); }
        }
        offset.assign(n + 1, 0);
        for (int k = 0; k < n; k++) offset[k + 1] = offset[k] + route[k].size();
        E = offset[n];
        visits.assign(N, {});
        vector<tuple<int, int, int, int>> all;    // (entry time, node, agent, index)
        for (int k = 0; k < n; k++)
            for (size_t j = 0; j < route[k].size(); j++) all.push_back({entry[k][j], route[k][j], k, (int)j});
        sort(all.begin(), all.end());
        for (auto& [t, v, k, j] : all) visits[v].push_back({k, j});
    }
    // earliest schedule for the given visit orders; empty if infeasible
    vector<Config> schedule(const vector<vector<pair<int, int>>>& vis) const {
        vector<array<int, 3>> cons;               // e[to] >= e[from] + weight
        for (int k = 0; k < n; k++)
            for (size_t j = 1; j < route[k].size(); j++)
                cons.push_back({offset[k] + (int)j - 1, offset[k] + (int)j, 1});
        for (size_t v = 0; v < vis.size(); v++)
            for (size_t p = 0; p + 1 < vis[v].size(); p++) {
                auto [a, ja] = vis[v][p];
                auto [b, jb] = vis[v][p + 1];
                if (ja + 1 >= (int)route[a].size()) return {};   // a never leaves v
                if (jb == 0) return {};                            // b starts at v: must be first
                cons.push_back({offset[a] + ja + 1, offset[b] + jb, 0});
            }
        vector<long long> e(E, 0);
        for (int k = 0; k < n; k++) for (size_t j = 0; j < route[k].size(); j++) e[offset[k] + j] = j;
        for (int it = 0;; it++) {
            bool changed = false;
            for (auto& c : cons)
                if (e[c[1]] < e[c[0]] + c[2]) { e[c[1]] = e[c[0]] + c[2]; changed = true; }
            for (int k = 0; k < n; k++) if (e[offset[k]] != 0) return {};   // starts at time 0
            if (!changed) break;
            if (it > E + 1) return {};                                       // positive cycle
        }
        long long Tn = 0;
        for (int k = 0; k < n; k++) Tn = max(Tn, e[offset[k + 1] - 1]);
        vector<Config> plan(Tn + 1, Config(n));
        for (int k = 0; k < n; k++) {
            size_t j = 0;
            for (long long t = 0; t <= Tn; t++) {
                while (j + 1 < route[k].size() && e[offset[k] + j + 1] <= t) j++;
                plan[t][k] = route[k][j];
            }
        }
        return plan;
    }
};

static bool valid_plan(const Graph& G, const vector<Config>& p, const Config& starts, const Config& goals) {
    if (p.empty() || p[0] != starts || p.back() != goals) return false;
    int n = starts.size();
    for (auto& x : p) {
        Config y = x;
        sort(y.begin(), y.end());
        if (adjacent_find(y.begin(), y.end()) != y.end()) return false;
    }
    for (size_t t = 1; t < p.size(); t++)
        for (int a = 0; a < n; a++) {
            if (!binary_search(G.out[p[t - 1][a]].begin(), G.out[p[t - 1][a]].end(), p[t][a])) return false;
            for (int b = a + 1; b < n; b++)
                if (p[t][a] == p[t - 1][b] && p[t][b] == p[t - 1][a] && p[t][a] != p[t - 1][a]) return false;
        }
    return true;
}

// Fast evaluation of visit orders: the constraints are encoded in arrays (the
// next event of the same agent is implicit; onext[x] is the event of the next
// visitor of the node of event x), candidate orders patch a few entries of
// onext and are undone after evaluation, and the least solution is computed
// with a queue-based Bellman-Ford (same least fixed point as schedule()).
struct FastOrders {
    const OrderModel& M;
    int E;
    vector<int> agent_of, idx_of, onext, cnt;
    vector<int> oprev;            // inverse of onext: previous visitor event (-1: first)
    vector<int> onext0;           // onext of the base orders
    vector<long long> e, e0;      // current and base earliest schedules
    vector<char> inA, inA2;       // incremental evaluation: affected events
    vector<int> A, stk;
    // strongly connected components of the affected events (iterative Tarjan)
    vector<int> tidx, tlow, cid, tstack, order_c;
    vector<char> onst;
    vector<pair<int, int>> frames;
    vector<char> inq;
    vector<int> q;
    explicit FastOrders(const OrderModel& m) : M(m), E(m.E) {
        agent_of.resize(E); idx_of.resize(E);
        for (int k = 0; k < M.n; k++)
            for (int j = 0; j < (int)M.route[k].size(); j++) { agent_of[M.offset[k] + j] = k; idx_of[M.offset[k] + j] = j; }
        onext.assign(E, -1); oprev.assign(E, -1);
        for (auto& L : M.visits) set_node(L);
        onext0 = onext;
        e.resize(E); inq.resize(E); cnt.resize(E);
        inA.assign(E, 0); inA2.assign(E, 0);
        tidx.assign(E, -1); tlow.assign(E, 0); cid.assign(E, -1); onst.assign(E, 0);
        nexte.assign(E, -1);
        for (int x = 0; x < E; x++)
            if (idx_of[x] + 1 < (int)M.route[agent_of[x]].size()) nexte[x] = x + 1;
        A.reserve(E); stk.reserve(E); tstack.reserve(E); order_c.reserve(E); frames.reserve(E);
    }
    // successors of event x: (x + 1, weight 1) and (onext[x - 1], weight 0)
    vector<int> nexte;            // nexte[x] = x + 1 if the agent has a next event, else -1
    int succ1(int x) const { return nexte[x]; }
    int succ0(int x) const { return idx_of[x] >= 1 ? onext[x - 1] : -1; }
    int ev(const pair<int, int>& v) const { return M.offset[v.first] + v.second; }
    void set_node(const vector<pair<int, int>>& L) {
        for (size_t p = 0; p < L.size(); p++) {
            onext[ev(L[p])] = p + 1 < L.size() ? ev(L[p + 1]) : -1;
            oprev[ev(L[p])] = p > 0 ? ev(L[p - 1]) : -1;
        }
    }
    // Incremental evaluation (exact). The base orders (onext0) have the least
    // schedule e0. A candidate changes the orders only at some nodes, so only
    // the constraints entering the visit events of those nodes (the seeds)
    // change. An event that is not reachable from a seed, in the base graph or
    // in the candidate graph, has the same ancestors and constraints in both,
    // hence its base value; only the reachable events A are recomputed, from
    // their fixed predecessors outside A.
    void set_base() { e0 = e; onext0 = onext; }
    template <class F> void closure(const vector<int>& seeds, const vector<int>& nxt, F mark) {
        stk.clear();
        for (int x : seeds) if (mark(x)) stk.push_back(x);
        while (!stk.empty()) {
            int x = stk.back(); stk.pop_back();
            int a = agent_of[x];
            if (idx_of[x] + 1 < (int)M.route[a].size() && mark(x + 1)) stk.push_back(x + 1);
            if (idx_of[x] >= 1 && nxt[x - 1] >= 0 && mark(nxt[x - 1])) stk.push_back(nxt[x - 1]);
        }
    }
    bool solve_incremental(const vector<int>& seeds) {
        // structural feasibility of the patched nodes
        for (int x : seeds) {
            if (onext[x] >= 0) {
                if (idx_of[x] + 1 >= (int)M.route[agent_of[x]].size()) return false;
                if (idx_of[onext[x]] == 0) return false;
            }
        }
        A.clear();
        auto mark = [&](int x) { if (inA[x]) return false; inA[x] = 1; A.push_back(x); return true; };
        closure(seeds, onext0, mark);           // values that may decrease
        closure(seeds, onext, mark);            // values that may increase
        // lower bounds from the fixed predecessors outside A
        for (int x : A) {
            long long v = idx_of[x];
            if (idx_of[x] >= 1 && !inA[x - 1]) v = max(v, e[x - 1] + 1);
            int pv = oprev[x];                   // previous visitor (a, j): x >= entry of (a, j+1)
            if (pv >= 0) {
                int y = pv + 1;                  // exists: checked above for seeds, unchanged otherwise
                if (!inA[y]) v = max(v, e[y]);
            }
            e[x] = v; cnt[x] = 0;
        }
        // Least solution in linear time: the constraint graph restricted to A
        // (closed under successors) is condensed into strongly connected
        // components, processed in topological order. A component may contain
        // only weight-0 arcs (rotations): all its events take the same value;
        // a weight-1 arc inside a component is a positive cycle (infeasible).
        int counter = 0, ncomp = 0;
        tstack.clear(); order_c.clear();
        vector<int>& compnodes = stk;            // nodes grouped by component
        compnodes.clear();
        vector<int> cstart;
        for (int s0 : A) {
            if (tidx[s0] >= 0) continue;
            frames.clear();
            frames.push_back({s0, 0});
            tidx[s0] = tlow[s0] = counter++; tstack.push_back(s0); onst[s0] = 1;
            while (!frames.empty()) {
                int x = frames.back().first;
                int& ei = frames.back().second;
                if (ei < 2) {
                    int y = ei == 0 ? succ1(x) : succ0(x);
                    ei++;
                    if (y < 0) continue;
                    if (tidx[y] < 0) {
                        tidx[y] = tlow[y] = counter++; tstack.push_back(y); onst[y] = 1;
                        frames.push_back({y, 0});
                    } else if (onst[y]) tlow[x] = min(tlow[x], tidx[y]);
                } else {
                    if (tlow[x] == tidx[x]) {      // x is the root of a component
                        cstart.push_back(compnodes.size());
                        int y;
                        do {
                            y = tstack.back(); tstack.pop_back(); onst[y] = 0;
                            cid[y] = ncomp; compnodes.push_back(y);
                        } while (y != x);
                        ncomp++;
                    }
                    frames.pop_back();
                    if (!frames.empty()) { int p = frames.back().first; tlow[p] = min(tlow[p], tlow[x]); }
                }
            }
        }
        cstart.push_back(compnodes.size());
        bool ok = true;
        // Tarjan emits the components in reverse topological order
        for (int c = ncomp - 1; c >= 0 && ok; c--) {
            long long v = 0;
            for (int i = cstart[c]; i < cstart[c + 1]; i++) v = max(v, e[compnodes[i]]);
            for (int i = cstart[c]; i < cstart[c + 1] && ok; i++) {
                int x = compnodes[i];
                int y1 = succ1(x);
                if (y1 >= 0 && cid[y1] == c) ok = false;          // positive cycle
                else if (idx_of[x] == 0 && v != 0) ok = false;     // a start time would move
                else if (bound >= 0 && v >= limit_of(agent_of[x])) ok = false;   // cannot improve
                e[x] = v;
            }
            if (!ok) break;
            for (int i = cstart[c]; i < cstart[c + 1]; i++) {
                int x = compnodes[i];
                int y1 = succ1(x), y0 = succ0(x);
                if (y1 >= 0 && cid[y1] != c) e[y1] = max(e[y1], v + 1);
                if (y0 >= 0 && cid[y0] != c) e[y0] = max(e[y0], v);
            }
            if (deadline && (c & 0x3FFF) == 0 && Clock::now() > *deadline) ok = false;
        }
        for (int x : A) { tidx[x] = -1; cid[x] = -1; }
        return ok;
    }
    // Quick exact test for single-node swaps. After exchanging a and b at a
    // node (b now enters before a), the only constraint that goes backwards
    // in the base schedule e0 is "a enters after b leaves" (from b's leaving
    // event to a's entry); every other constraint, old or new, satisfies
    // e0[head] >= e0[tail]. So there is a positive cycle (the candidate is
    // infeasible) iff a's entry reaches b's leaving event by a path with at
    // least one weight-1 constraint, and every node of such a path has
    // e0 <= e0[b's leaving event]: the search explores only that window.
    // Paths of weight-0 constraints only (rotations) are not reported.
    vector<int> qc_vis0, qc_vis1, qc_stack;
    int qc_stamp = 0;
    bool quick_cycle(int av, int bv) {
        int target = bv + 1;
        if (idx_of[bv] + 1 >= (int)M.route[agent_of[bv]].size()) return false;   // left to the solver
        if (qc_vis0.size() != (size_t)E) { qc_vis0.assign(E, 0); qc_vis1.assign(E, 0); }
        if (++qc_stamp == 0) { fill(qc_vis0.begin(), qc_vis0.end(), 0); fill(qc_vis1.begin(), qc_vis1.end(), 0); qc_stamp = 1; }
        long long tmax = e0[target];
        qc_stack.clear();
        qc_stack.push_back(av * 2); qc_vis0[av] = qc_stamp;            // (event, flag) encoded as 2x + flag
        while (!qc_stack.empty()) {
            int code = qc_stack.back(); qc_stack.pop_back();
            int x = code >> 1, f = code & 1;
            int y1 = succ1(x), y0 = succ0(x);
            if (y1 >= 0 && e0[y1] <= tmax) {                       // weight 1: flag becomes 1
                if (y1 == target) return true;
                if (qc_vis1[y1] != qc_stamp) { qc_vis1[y1] = qc_stamp; qc_stack.push_back(y1 * 2 + 1); }
            }
            if (y0 >= 0 && e0[y0] <= tmax) {                       // weight 0: flag kept
                if (y0 == target && f) return true;
                vector<int>& vis = f ? qc_vis1 : qc_vis0;
                if (vis[y0] != qc_stamp) { vis[y0] = qc_stamp; qc_stack.push_back(y0 * 2 + f); }
            }
        }
        return false;
    }
    void restore() {                            // back to the base schedule
        for (int x : A) { e[x] = e0[x]; inA[x] = 0; }
        A.clear();
    }
    // Pruning (exact): with soc, a candidate can improve only if every arrival
    // time e_a satisfies e_a < bound - (lb_total - lb_a), where lb_a is the
    // number of moves of agent a (a lower bound on its arrival); every entry
    // time of a precedes its arrival, so the same bound applies to all of them.
    // With the makespan, e < bound. bound < 0: no pruning.
    long long bound = -1;
    bool soc_bound = true;
    long long lb_total = 0;
    vector<long long> lb;
    const Clock::time_point* deadline = nullptr;
    void set_bound(long long b, bool soc) {
        bound = b; soc_bound = soc;
        if (lb.empty()) {
            lb.resize(M.n);
            for (int k = 0; k < M.n; k++) { lb[k] = (long long)M.route[k].size() - 1; lb_total += lb[k]; }
        }
    }
    long long limit_of(int a) const {
        return soc_bound ? bound - (lb_total - lb[a]) : bound;
    }
    // earliest schedule for the current onext; false if infeasible, or (with a
    // bound) if it cannot improve, or if the deadline has passed
    bool solve() {
        for (int x = 0; x < E; x++) {
            int a = agent_of[x];
            if (onext[x] >= 0) {
                if (idx_of[x] + 1 >= (int)M.route[a].size()) return false;   // never leaves
                if (idx_of[onext[x]] == 0) return false;                    // next starts there
            }
            e[x] = idx_of[x]; inq[x] = 1; cnt[x] = 0;
        }
        q.clear();
        for (int x = 0; x < E; x++) q.push_back(x);
        size_t h = 0;
        while (h < q.size()) {
            int x = q[h++];
            inq[x] = 0;
            auto relax = [&](int y, long long val) -> bool {
                if (e[y] < val) {
                    e[y] = val;
                    if (idx_of[y] == 0) return false;       // a start time would move
                    if (bound >= 0 && val >= limit_of(agent_of[y])) return false;   // cannot improve
                    if (++cnt[y] > E + 1) return false;     // positive cycle
                    if (!inq[y]) { inq[y] = 1; q.push_back(y); }
                }
                return true;
            };
            int a = agent_of[x];
            if (idx_of[x] + 1 < (int)M.route[a].size() && !relax(x + 1, e[x] + 1)) return false;
            // x = entry of (a, j): it is also the leaving time of (a, j - 1)
            if (idx_of[x] >= 1 && onext[x - 1] >= 0 && !relax(onext[x - 1], e[x])) return false;
            if (h > 1u << 26) { q.erase(q.begin(), q.begin() + h); h = 0; }
            if (deadline && (h & 0xFFFF) == 0 && Clock::now() > *deadline) return false;
        }
        return true;
    }
    vector<Config> plan() const {
        long long Tn = 0;
        for (int k = 0; k < M.n; k++) Tn = max(Tn, e[M.offset[k + 1] - 1]);
        vector<Config> p(Tn + 1, Config(M.n));
        for (int k = 0; k < M.n; k++) {
            size_t j = 0;
            for (long long t = 0; t <= Tn; t++) {
                while (j + 1 < M.route[k].size() && e[M.offset[k] + j + 1] <= t) j++;
                p[t][k] = M.route[k][j];
            }
        }
        return p;
    }
    // objective without building the plan
    long long objective(const Config& goals, bool soc) const {
        long long v = 0;
        for (int k = 0; k < M.n; k++) {
            long long arr = e[M.offset[k + 1] - 1];
            if (soc) v += arr; else v = max(v, arr);
        }
        return v;
    }
};

// mode: "retime" or "swap" (single-node and pair-wide swaps, best improvement)
static SearchResult order_search(const Graph& G, const Config& goals, const vector<Config>& ref,
                                 const string& mode, bool soc, Clock::time_point deadline) {
    SearchResult res;
    OrderModel M;
    M.build(ref, G.N);
    auto obj = [&](const vector<Config>& p) {
        return soc ? sum_of_costs(p, goals) : (long long)makespan(p);
    };
    const Config& starts = ref[0];
    FastOrders F(M);
    bool base_ok = F.solve();
    vector<Config> best = base_ok ? F.plan() : vector<Config>();
    if (best.empty() || !valid_plan(G, best, starts, goals)) best = ref;
    long long bestv = obj(best);
    res.states = 1;
    if (mode == "swap") {
        // candidate: evaluate with the arrival-time objective (the plan's objective
        // equals it: arrivals are the last entry times), build and validate the
        // plan only when it improves
        F.deadline = &deadline;
        // incremental evaluation from the base schedule when the base orders
        // are feasible (always, for a feasible reference)
        if (base_ok) F.set_base();
        vector<int> seeds;
        auto evaluate = [&]() {
            res.states++;
            F.set_bound(bestv, soc);
            bool ok = base_ok ? F.solve_incremental(seeds) : F.solve();
            if (ok) {
                long long v = F.objective(goals, soc);
                if (v < bestv) {
                    auto q = F.plan();
                    if (obj(q) < bestv && valid_plan(G, q, starts, goals)) { best = q; bestv = obj(q); }
                }
            }
            if (base_ok) F.restore();
        };
        auto seeds_of = [&](const vector<pair<int, int>>& L) { for (auto& x : L) seeds.push_back(F.ev(x)); };
        // Exact pruning: exchanging consecutive visitors a, b of a node removes
        // the constraint "b enters after a leaves" and adds the opposite one.
        // If the removed constraint is not tight in the base schedule, no
        // longest path uses it, so removing it changes nothing, and adding a
        // constraint can only delay events: the candidate cannot improve.
        auto tight = [&](const pair<int, int>& a, const pair<int, int>& b) {
            if (!base_ok) return true;
            int ea = F.ev(a), eb = F.ev(b);
            if (a.second + 1 >= (int)M.route[a.first].size()) return true;   // let the solver reject it
            return F.e0[eb] == F.e0[ea + 1];
        };
        for (size_t v = 0; v < M.visits.size() && Clock::now() < deadline; v++) {
            auto L = M.visits[v];
            for (size_t p = 0; p + 1 < L.size(); p++) {
                if (L[p].first == L[p + 1].first) continue;
                if (!tight(L[p], L[p + 1])) continue;
                if (Clock::now() > deadline) break;
                swap(L[p], L[p + 1]);
                F.set_node(L);
                // b = L[p] now enters before a = L[p + 1]: reject at once if the
                // exchange creates a positive cycle (same decision as the solver)
                if (base_ok && F.quick_cycle(F.ev(L[p + 1]), F.ev(L[p]))) {
                    res.states++;
                } else {
                    seeds.clear(); seeds_of(L);
                    evaluate();
                }
                swap(L[p], L[p + 1]);
                F.set_node(L);
            }
        }
        set<pair<int, int>> pairs;
        for (auto& L : M.visits)
            for (size_t p = 0; p + 1 < L.size(); p++)
                if (L[p].first != L[p + 1].first) pairs.insert({L[p].first, L[p + 1].first});
        for (auto [a, b] : pairs) {
            if (Clock::now() > deadline) break;
            vector<size_t> touched;
            int changed = 0;
            bool any_tight = false;
            for (size_t v = 0; v < M.visits.size() && !any_tight; v++) {
                const auto& L = M.visits[v];
                for (size_t p = 0; p + 1 < L.size(); p++)
                    if (L[p].first == a && L[p + 1].first == b && tight(L[p], L[p + 1])) { any_tight = true; break; }
            }
            if (!any_tight) continue;                 // cannot improve (see tight)
            for (size_t v = 0; v < M.visits.size(); v++) {
                auto L = M.visits[v];
                bool mod = false;
                for (size_t p = 0; p + 1 < L.size(); p++)
                    if (L[p].first == a && L[p + 1].first == b) { swap(L[p], L[p + 1]); changed++; p++; mod = true; }
                if (mod) { F.set_node(L); touched.push_back(v); }
            }
            seeds.clear();
            for (size_t v : touched) seeds_of(M.visits[v]);
            if (changed >= 2) evaluate();          // single-node swaps already tried
            for (size_t v : touched) F.set_node(M.visits[v]);
        }
    }
    res.found = true;
    res.plan = best;
    return res;
}


// Distances to the goals, for the A* heuristic: dg[k][v] = dist_G(v, goal_k).
// Conflict term of the A* heuristic (Sum-of-Costs). Agent a "rests" if it is on
// its goal; a moving agent b is blocked by a if g_a lies on every shortest path
// from b's node to g_b. Then in every plan either b avoids g_a (at least one
// extra move) or a leaves its goal and returns (at least two): each pair costs
// at least one more than the sum of the distances. The pairs of a matching
// involve distinct agents, so sum_k d_k + |maximum matching| is a lower bound.
// It is consistent: one agent changes its node per step of the sequential
// search, which changes the maximum matching by at most one; along a shortest
// path the blocking nodes of an agent are kept (every shortest path from its
// next node extends to one from its node), and next to its goal it has none,
// so the matching can lose a pair only in steps that cost at least one and do
// not decrease the sum of the distances.
struct ConflictBound {
    int n = 0, N = 0, Wd = 0;
    vector<uint64_t> bits;            // bits[(b * N + v) * Wd + w]: goals on every shortest path
    const Config* goals = nullptr;
    const vector<vector<int>>* dg = nullptr;
    void build(const Graph& G, const Config& gl, const vector<vector<int>>& d) {
        n = gl.size(); N = G.N; Wd = (n + 63) / 64; goals = &gl; dg = &d;
        vector<int> goal_of(N, -1);
        for (int a = 0; a < n; a++) goal_of[gl[a]] = a;
        bits.assign((size_t)n * N * Wd, 0);
        vector<int> order(N);
        for (int b = 0; b < n; b++) {
            iota(order.begin(), order.end(), 0);
            sort(order.begin(), order.end(), [&](int u, int v) { return d[b][u] < d[b][v]; });
            for (int v : order) {
                int dv = d[b][v];
                if (dv == 0 || dv >= INT_MAX / 4) continue;
                uint64_t* bv = &bits[((size_t)b * N + v) * Wd];
                bool first = true;
                for (int w : G.out[v]) {
                    if (w == v || d[b][w] != dv - 1) continue;      // shortest-path successors
                    const uint64_t* bw = &bits[((size_t)b * N + w) * Wd];
                    vector<uint64_t> m(bw, bw + Wd);
                    int a = goal_of[w];
                    if (a >= 0 && a != b) m[a >> 6] |= 1ULL << (a & 63);
                    if (first) { for (int q = 0; q < Wd; q++) bv[q] = m[q]; first = false; }
                    else for (int q = 0; q < Wd; q++) bv[q] &= m[q];
                }
            }
        }
    }
    // maximum matching between blocked moving agents and resting agents
    vector<int> matchR, seen; vector<uint64_t> rest; vector<int> movers; int stamp = 0;
    long long value(const vector<int>& x) {
        rest.assign(Wd, 0); movers.clear();
        bool any_rest = false;
        for (int a = 0; a < n; a++) if (x[a] == (*goals)[a]) { rest[a >> 6] |= 1ULL << (a & 63); any_rest = true; }
        if (!any_rest) return 0;
        for (int b = 0; b < n; b++) {
            if (x[b] == (*goals)[b]) continue;
            const uint64_t* bb = &bits[((size_t)b * N + x[b]) * Wd];
            for (int q = 0; q < Wd; q++) if (bb[q] & rest[q]) { movers.push_back(b); break; }
        }
        if (movers.empty()) return 0;
        matchR.assign(n, -1); seen.assign(n, 0);
        long long m = 0;
        function<bool(int)> aug = [&](int b) -> bool {
            const uint64_t* bb = &bits[((size_t)b * N + x[b]) * Wd];
            for (int q = 0; q < Wd; q++) {
                uint64_t c = bb[q] & rest[q];
                while (c) {
                    int a = q * 64 + __builtin_ctzll(c); c &= c - 1;
                    if (seen[a] == stamp) continue;
                    seen[a] = stamp;
                    if (matchR[a] < 0 || aug(matchR[a])) { matchR[a] = b; return true; }
                }
            }
            return false;
        };
        for (int b : movers) { stamp++; if (aug(b)) m++; }
        return m;
    }
};

// --pair-h: pairwise pattern databases. For disjoint pairs (a, b), a < b, the
// exact cost-to-go of the two agents alone in the sequential automaton (a
// decides, then b; b may not move onto a's new node; swaps allowed; the same 0/1
// costs), as a table over (node of a, node of b, phase), phase 1 meaning that a
// has decided in the current round. The pair model is an abstraction of the
// search (other agents ignored), so its cost-to-go is consistent; with disjoint
// pairs, h = sum over the pairs + sum of the distances of the other agents.
static bool g_pair_h = false;
// --anytime: an exact search that exhausts its expansion budget does not give up;
// it continues as weighted A* (key g + W h, W = 3) on the states already
// generated, for a quarter of the budget more, and returns the first complete
// plan it finds (not proven optimal: the search still counts as over budget)
static bool g_anytime = false;
static atomic<long long> g_anytime_found{0};
static long long g_anytime_w = 3;
// --chain: a decision with a single feasible successor (an agent whose move is
// forced, e.g. frozen in a free-set neighborhood) is applied within the same
// expansion instead of creating and storing an intermediate state; chains stop
// at the end of a round (complete configurations are always stored)
static bool g_chain = false;
struct PairDB {
    int N = 0;
    vector<pair<int, int>> pairs;
    vector<int> partner;                  // partner[k] = index of k's pair, -1 if none
    vector<vector<uint16_t>> tab;         // tab[p][(ph * N + u) * N + v]
    static const uint16_t UNR = 65535;
    void build(const Graph& G, const Config& goals, const vector<Config>& ref) {
        int n = goals.size(); N = G.N;
        partner.assign(n, -1); pairs.clear(); tab.clear();
        // interaction: number of nodes shared by the two reference routes
        vector<vector<int>> on(n);
        for (int k = 0; k < n; k++) {
            vector<char> m(N, 0);
            for (auto& x : ref) m[x[k]] = 1;
            for (int v = 0; v < N; v++) if (m[v]) on[k].push_back(v);
        }
        vector<tuple<int, int, int>> cand;
        vector<int> mark(N, -1);
        for (int a = 0; a < n; a++) {
            for (int v : on[a]) mark[v] = a;
            for (int b = a + 1; b < n; b++) {
                int c = 0;
                for (int v : on[b]) if (mark[v] == a) c++;
                if (c > 0) cand.emplace_back(-c, a, b);
            }
        }
        sort(cand.begin(), cand.end());
        for (auto& [c, a, b] : cand) {
            if (partner[a] >= 0 || partner[b] >= 0) continue;
            partner[a] = partner[b] = (int)pairs.size();
            pairs.push_back({a, b});
        }
        for (auto& [a, b] : pairs) tab.push_back(solve(G, goals[a], goals[b]));
    }
    // backward 0-1 BFS from (g_a, g_b, phase 0)
    vector<uint16_t> solve(const Graph& G, int ga, int gb) {
        vector<uint16_t> d((size_t)2 * N * N, UNR);
        deque<pair<int, uint16_t>> q;     // (index, cost)
        auto idx = [&](int ph, int u, int v) { return ((size_t)ph * N + u) * N + v; };
        d[idx(0, ga, gb)] = 0; q.push_back({(int)idx(0, ga, gb), 0});
        while (!q.empty()) {
            auto [x, c] = q.front(); q.pop_front();
            if (c != d[x]) continue;
            int ph = x / (N * N), u = (x / N) % N, v = x % N;
            if (ph == 0) {
                // predecessors (u, vp, 1) -> b moves vp -> v, with v != u
                if (u == v) continue;
                for (int vp : G.inn[v]) {
                    int cb = (vp == gb && v == gb) ? 0 : 1;
                    size_t y = idx(1, u, vp);
                    if (d[y] > c + cb) { d[y] = c + cb; if (cb) q.push_back({(int)y, d[y]}); else q.push_front({(int)y, d[y]}); }
                }
            } else {
                // predecessors (up, v, 0) -> a moves up -> u
                for (int up : G.inn[u]) {
                    int ca = (up == ga && u == ga) ? 0 : 1;
                    size_t y = idx(0, up, v);
                    if (d[y] > c + ca) { d[y] = c + ca; if (ca) q.push_back({(int)y, d[y]}); else q.push_front({(int)y, d[y]}); }
                }
            }
        }
        return d;
    }
};

// value of pair p minus the distances of its agents, for the concrete state (z, y, i)
static long long pair_term(const PairDB& P, int p, const Config& z, const Config& y,
                           const vector<vector<int>>& d) {
    int a = P.pairs[p].first, b = P.pairs[p].second, i = y.size();
    int ph, u, v;
    if (i <= a) { ph = 0; u = z[a]; v = z[b]; }
    else if (i <= b) { ph = 1; u = y[a]; v = z[b]; }
    else { ph = 0; u = y[a]; v = y[b]; }
    uint16_t c = P.tab[p][((size_t)ph * P.N + u) * P.N + v];
    long long val = c == PairDB::UNR ? (long long)BIG : (long long)c;
    int xa = a < i ? y[a] : z[a], xb = b < i ? y[b] : z[b];
    return val - d[a][xa] - d[b][xb];
}

static vector<vector<int>> goal_distances(const Graph& G, const Config& goals) {
    vector<vector<int>> dg(goals.size(), vector<int>(G.N, INT_MAX / 4));
    for (size_t k = 0; k < goals.size(); k++) {
        vector<int> q{goals[k]};
        dg[k][goals[k]] = 0;
        for (size_t h = 0; h < q.size(); h++)
            for (int u : G.inn[q[h]])
                if (dg[k][u] == INT_MAX / 4) { dg[k][u] = dg[k][q[h]] + 1; q.push_back(u); }
    }
    return dg;
}

// Consistent lower bound on the cost-to-go of the sequential state (z, y),
// ignoring collisions and the locality constraint.
//   makespan: (n - i) + n * max(max_{k<i} d_k(y_k), max_{k>=i} d_k(z_k) - 1, 0),
//             0 if i = 0 and z is the goal configuration;
//   Sum-of-Costs (exact or proxy): sum_k d_k(x_k), x_k = y_k if k < i, else z_k.
static long long heuristic(const vector<vector<int>>& dg, const Config& z, const Config& y,
                           bool sum, const Neighborhood* nb = nullptr,
                           const vector<long long>* aux = nullptr) {
    int n = z.size(), i = y.size();
    if (nb && nb->goal_dist()) {           // tighter distances inside the neighborhood
        const vector<vector<int>>& d2 = *nb->goal_dist();
        long long base = heuristic(d2, z, y, sum);
        long long extra = (!sum && aux) ? nb->rounds_lb(*aux) : 0;
        return max(base, extra > 0 ? (n - i) + (long long)n * (extra - 1) : 0);
    }
    if (sum) {
        long long h = 0;
        for (int k = 0; k < n; k++) h += dg[k][k < i ? y[k] : z[k]];
        return h;
    }
    long long M = 0;
    bool at_goals = true;
    for (int k = 0; k < n; k++) {
        long long d = k < i ? dg[k][y[k]] : dg[k][z[k]] - 1;
        if (k >= i && dg[k][z[k]] != 0) at_goals = false;
        M = max(M, d);
    }
    if (i == 0 && at_goals) return 0;
    long long h = (n - i) + (long long)n * M;
    if (nb && aux) {
        // joint step c is in progress and agents outside D follow the reference
        // until time tau_k: at least `extra` more joint steps, the current one
        // included when i > 0
        long long extra = nb->rounds_lb(*aux);
        if (extra > 0) h = max(h, (n - i) + (long long)n * (extra - 1));
    }
    return h;
}

static SearchResult neighborhood_search(const Graph& G, const Config& goals, const vector<Config>& ref,
                                        Neighborhood& nb, bool tiebreak, long long max_exp,
                                        Clock::time_point deadline, bool soc, bool soc_proxy,
                                        const vector<vector<int>>* dg = nullptr,
                                        bool enumerate = false, long long beam = 0,
                                        long long incumbent = -1) {
    int n = ref[0].size();
    if (G.N < n + 1) { cerr << "Lemma lem_mapf_Si requires |V| >= n+1\n"; exit(2); }
    if (G.N >= 0xFFFF) { cerr << "graphs with more than 65534 nodes are not supported\n"; exit(2); }
    // tables larger than DPLS_MAX_TABLE_MB (default 2048) are not allocated: the
    // search ends as "budget exceeded", so the result is not certified
    static const size_t max_table = [] {
        const char* e = getenv("DPLS_MAX_TABLE_MB");
        return (size_t)(e ? atof(e) : 2048.0) * (size_t)(1 << 20);
    }();
    if (nb.table_bytes(G, ref) > max_table) {
        SearchResult res;
        res.budget = true;
        return res;
    }
    nb.prepare(G, ref);
    SumMin tb;
    if (tiebreak) tb.prepare(G, ref);
    vector<long long> aux0 = nb.init_aux();
    StateStore store;
    store.init(n, (int)aux0.size(), soc, 1 << 15);   // grows on demand
    vector<int> parent;
    vector<long long> dist;
    vector<char> closed;
    // heap entry: lexicographic key (g, tb, len, counter) packed in two words
    struct Entry { uint64_t k1, k2; int sid; long long g; };
    struct EntryCmp {                // a functor, so that the queue can be rebuilt (beam)
        bool operator()(const Entry& a, const Entry& b) const {
            if (a.k1 != b.k1) return a.k1 > b.k1;
            return a.k2 > b.k2;
        }
    };
    auto mk = [](long long g, long long t, int len, long long cnt, int sid) {
        // g < 2^40 and tb < 2^24, len < 2^16 and counter < 2^48 (asserted by bounds below)
        return Entry{((uint64_t)g << 24) | (uint64_t)t, ((uint64_t)len << 48) | (uint64_t)cnt, sid, g};
    };
    // binary heap on a vector (same order as priority_queue with EntryCmp: the
    // smallest key first); a vector lets the beam select the entries to keep
    // in linear time
    struct Heap {
        vector<Entry> v; EntryCmp c;
        bool empty() const { return v.empty(); }
        size_t size() const { return v.size(); }
        const Entry& top() const { return v.front(); }
        void push(const Entry& e) { v.push_back(e); push_heap(v.begin(), v.end(), c); }
        void pop() { pop_heap(v.begin(), v.end(), c); v.pop_back(); }
    } heap;
    heap.v.reserve(1 << 16);
    vector<uint16_t> buf(store.K);
    Config cnt0 = soc ? Config(n, 0) : Config();
    store.encode(ref[0], {}, aux0, cnt0, buf.data());
    store.find_or_insert(buf.data());
    parent.push_back(-1); dist.push_back(0); closed.push_back(0);
    const bool sum_h = soc || soc_proxy;
    // conflict term (only with the Sum-of-Costs and the distances in the whole graph)
    const vector<vector<int>>* dh0 = dg ? (nb.goal_dist() ? nb.goal_dist() : dg) : nullptr;
    bool use_conf = g_conflict_h && sum_h && dg && dh0 == dg;
    static ConflictBound CB;
    static const vector<vector<int>>* cb_for = nullptr;
    if (use_conf) { lock_guard<mutex> lk(g_build_mutex); if (cb_for != dg) { CB.build(G, goals, *dg); cb_for = dg; } }
    vector<int> X(n);
    static PairDB PDB;
    static bool pdb_built = false, pdb_ok = false;
    if (g_pair_h) {
        lock_guard<mutex> lk(g_build_mutex);
        if (!pdb_built) { pdb_built = true; if (G.N <= 400) { PDB.build(G, goals, ref); pdb_ok = true; } }
    }
    bool use_pair = g_pair_h && pdb_ok && sum_h && dg && dh0 == dg && !use_conf;
    {
    long long h0 = dg ? heuristic(*dg, ref[0], {}, sum_h, &nb, &aux0) : 0;
    if (use_conf) { for (int k = 0; k < n; k++) X[k] = ref[0][k]; h0 += CB.value(X); }
    if (use_pair) for (size_t p = 0; p < PDB.pairs.size(); p++) h0 += pair_term(PDB, p, ref[0], {}, *dg);
    Entry en = mk(h0, 0, 0, 0, 0);
        en.g = 0;
        heap.push(en);
    }
    long long counter = 1, expansions = 0;
    long long s_flast = -1, s_below = 0, s_at = 0;
    bool greedy = false;                 // weighted phase of an anytime search
    const long long W_ANY = g_anytime_w;
    long long stat_incumbent = 0;
    for (int k = 0; k < n; k++) stat_incumbent += 0;     // set below from the reference
    {
        long long c = 0;                                  // Sum-of-Costs of the reference
        for (int k = 0; k < n; k++) {
            int t = (int)ref.size() - 1;
            while (t > 0 && ref[t - 1][k] == goals[k]) t--;
            c += t;
        }
        stat_incumbent = c;
    }
    SearchResult res;
    State st, s2;                    // reused buffers
    Config z2, y2, cnt2;
    // per-expansion marks for the collision checks: occ[v] == stamp if a
    // decided agent moves to v (see the collision marks in the expansion)
    // v to the node of the agent deciding now (moving there would be a swap)
    vector<unsigned> occ(G.N, 0);
    vector<int> occa(G.N, -1);
    unsigned stamp = 0;
    // batch of the successors of one expansion
    vector<int> b_w; vector<long long> b_cost; vector<uint64_t> b_h;
    vector<vector<long long>> b_aux; vector<uint16_t> b_buf;
    vector<Config> b_cnt;               // counters of the successors (exact Sum-of-Costs only)
    vector<long long> aux2;
    z2.reserve(n); y2.reserve(n);
    while (!heap.empty()) {
        // Beam (bounded open list): keep only the `beam` most promising
        // entries, in the order of the A* key (cost + heuristic, then the
        // tie-breaks). The search is then no longer exact: it returns the best
        // plan among the states it kept, and the result is flagged as pruned.
        if (beam > 0 && (long long)heap.size() > 2 * beam) {
            // keep the `beam` valid entries with the smallest keys (keys are
            // unique: same set as popping them one by one)
            auto& v = heap.v;
            size_t m = 0;
            for (size_t q = 0; q < v.size(); q++)
                if (!closed[v[q].sid] && v[q].g == dist[v[q].sid]) v[m++] = v[q];
            v.resize(m);
            if ((long long)m > beam) {
                // the comparator orders a max-heap of "greater" keys: the
                // smallest keys come last in the sorted order of EntryCmp
                auto less_key = [&](const Entry& a, const Entry& b) { return heap.c(b, a); };
                nth_element(v.begin(), v.begin() + beam, v.end(), less_key);
                v.resize(beam);
                res.pruned = true;
            }
            make_heap(v.begin(), v.end(), heap.c);
            if (heap.empty()) break;
        }
        Entry e = heap.top(); heap.pop();
        int sid = e.sid;
        if (closed[sid]) continue;
        // Beam searches stop as soon as no plan they can still reach can be
        // accepted: the key f = g + h is a lower bound on the search cost of
        // every plan through this state, the search cost never exceeds the
        // Sum-of-Costs (or the makespan) of a plan, and a plan is accepted only
        // if its objective is below the incumbent's. Exact searches are left
        // unchanged.
        if (!greedy && incumbent >= 0 && (long long)(e.k1 >> 24) >= incumbent) break;
        closed[sid] = 1;
        long long fcur = (long long)(e.k1 >> 24);
        if (fcur > s_flast) { s_below += s_at; s_at = 0; s_flast = fcur; }
        store.decode(sid, st);
        if (!enumerate && st.y.empty() && st.z == goals && nb.marked(st)) {
            g_exp_below += s_below; g_exp_at += s_at; g_found++;
            vector<Config> plan;
            State tmp;
            for (int x = sid; x >= 0; x = parent[x]) {
                store.decode(x, tmp);
                if (tmp.y.empty()) plan.push_back(tmp.z);
            }
            reverse(plan.begin(), plan.end());
            res.found = true; res.plan = plan; res.expansions = expansions; res.states = store.count;
            if (greedy) { res.budget = true; g_anytime_found++; }   // found by the weighted phase
            return res;
        }
        expansions++; s_at++;
        if (g_anytime && dg && !greedy && expansions > max_exp && !(Clock::now() > deadline)) {
            // switch to weighted A*: re-key the open list with g + W h
            greedy = true;
            max_exp += max_exp / 4;
            for (auto& x : heap.v) {
                long long fx = (long long)(x.k1 >> 24), hx = fx - x.g;
                x.k1 = ((uint64_t)(x.g + W_ANY * hx) << 24) | (x.k1 & ((1ULL << 24) - 1));
            }
            make_heap(heap.v.begin(), heap.v.end(), heap.c);
        }
        if (expansions > max_exp || (expansions % 256 == 0 && Clock::now() > deadline)) {
            g_budget_n++; g_budget_gap += max(0LL, stat_incumbent - fcur);
            res.budget = true; res.expansions = expansions; res.states = store.count;
            return res;
        }
        long long chain_g = 0;               // cost of the forced moves applied so far
        bool chained = false;
        long long chain_h = 0, chain_tb = 0; // heuristic and tie-break key after the forced moves
    expand_again:
        nb.z_unchanged = chained;            // within a chain z does not change
        nb.expand(st);
        int i = st.y.size();
        int zi = st.z[i];
        bool done = (i == n - 1);
        // incremental A* heuristic (same values as heuristic(), O(1) per successor)
        const vector<vector<int>>* dh = dg ? (nb.goal_dist() ? nb.goal_dist() : dg) : nullptr;
        long long h_sum = 0, h_mexcl = 0, h_ydmax = 0, pair_sum = 0;
        if (use_pair) for (size_t p = 0; p < PDB.pairs.size(); p++) pair_sum += pair_term(PDB, p, st.z, st.y, *dg);
        bool h_ygoal = true;
        if (dh) {
            const vector<vector<int>>& d = *dh;
            if (sum_h) {
                // the sum of the distances of this state is f - g of its queue
                // entry (it was computed for it as a successor), except for the
                // start state, whose entry was built by heuristic()
                if (chained && !use_conf && !use_pair) h_sum = chain_h;      // updated along the chain
                else if (sid != 0 && !use_conf && !use_pair) h_sum = (long long)(e.k1 >> 24) - e.g;
                else for (int k = 0; k < n; k++) h_sum += d[k][k < i ? st.y[k] : st.z[k]];
            } else {
                for (int k = 0; k < n; k++) {
                    if (k == i) continue;
                    long long t = k < i ? d[k][st.y[k]] : d[k][st.z[k]] - 1;
                    h_mexcl = max(h_mexcl, t);
                }
                for (int k = 0; k < i; k++) {
                    h_ydmax = max<long long>(h_ydmax, d[k][st.y[k]]);
                    if (d[k][st.y[k]] != 0) h_ygoal = false;
                }
            }
        }
        auto h_succ = [&](int w, const vector<long long>& a2) -> long long {
            const vector<vector<int>>& d = *dh;
            if (sum_h) return h_sum - d[i][zi] + d[i][w];
            long long h;
            if (!done) h = (n - i - 1) + (long long)n * max(h_mexcl, (long long)d[i][w]);
            else if (h_ygoal && d[i][w] == 0) h = 0;
            else h = (long long)n + (long long)n * max(max(h_ydmax, (long long)d[i][w]) - 1, 0LL);
            long long extra = nb.rounds_lb(a2);
            if (extra > 0) {
                int i2 = done ? 0 : i + 1;
                h = max(h, (n - i2) + (long long)n * (extra - 1));
            }
            return h;
        };
        // incremental tie-break key (same values as tb.value(z2, y2)): a
        // successor changes only the term of agent i, from zi to w
        // the tie-break key of this state is the second field of its queue key,
        // unless it is the start state (key 0) or the key saturated
        long long tb_base = 0;
        if (tiebreak) {
            long long t0 = (long long)(e.k1 & ((1ULL << 24) - 1));
            if (chained) tb_base = chain_tb;                                  // updated along the chain
            else if (sid != 0 && t0 < (1LL << 24) - 1) tb_base = t0;
            else for (int k = 0; k < n; k++) tb_base += tb.mu[k][k < i ? st.y[k] : st.z[k]];
        }
        // collision marks: occ[v] == stamp if a decided agent moves to v, and
        // occa[v] is that agent; a move to w is a swap if the decided agent
        // moving to zi comes from w. Set once per popped state, extended by one
        // agent along a chain (same tests as scanning the decided agents).
        if (!chained) {
            if (++stamp == 0) { fill(occ.begin(), occ.end(), 0); stamp = 1; }
            for (int k = 0; k < i; k++) { occ[st.y[k]] = stamp; occa[st.y[k]] = k; }
        }
        // successor configurations built once; only the last element changes
        if (done) { z2.assign(st.y.begin(), st.y.end()); z2.push_back(-1); y2.clear(); }
        else { z2.assign(st.z.begin(), st.z.end()); y2.assign(st.y.begin(), st.y.end()); y2.push_back(-1); }
        // Two passes over the successors: the first encodes and hashes them and
        // prefetches their table slots, the second looks them up and inserts
        // them in the same order as a single pass. The successors of one
        // expansion are distinct states (they differ in the node of agent i),
        // so inserting one cannot change the lookup of another.
        int nb_succ = 0;
        for (int w : G.out[zi]) {
            if (occ[w] == stamp) continue;                               // vertex conflict
            if (occ[zi] == stamp && st.z[occa[zi]] == w) continue;       // swap
            if (done) z2.back() = w; else y2.back() = w;
            aux2.assign(st.aux.begin(), st.aux.end());
            if (!nb.step(st, i, w, z2, y2, done, aux2)) continue;
            long long cost = 1;
            if (soc_proxy && zi == goals[i] && w == goals[i]) cost = 0;
            if (soc) {
                cnt2.assign(st.cnt.begin(), st.cnt.end());
                if (zi == goals[i] && w == goals[i]) { cost = 0; cnt2[i] = st.cnt[i] + 1; }
                else if (zi == goals[i]) { cost = st.cnt[i] + 1; cnt2[i] = 0; }
                else { cost = 1; cnt2[i] = 0; }
            }
            if ((int)b_w.size() <= nb_succ) {
                b_w.resize(nb_succ + 1); b_cost.resize(nb_succ + 1); b_h.resize(nb_succ + 1);
                b_aux.resize(nb_succ + 1); b_buf.resize((size_t)(nb_succ + 1) * store.K);
            }
            b_w[nb_succ] = w; b_cost[nb_succ] = cost; b_aux[nb_succ] = aux2;
            if (soc) { if ((int)b_cnt.size() <= nb_succ) b_cnt.resize(nb_succ + 1); b_cnt[nb_succ] = cnt2; }
            nb_succ++;
        }
        if (g_chain && nb_succ == 1 && !done && !soc) {
            // forced move: apply it and decide the next agent in this expansion
            if (dh && sum_h) chain_h = h_sum - (*dh)[i][zi] + (*dh)[i][b_w[0]];
            if (tiebreak) chain_tb = tb_base - tb.mu[i][zi] + tb.mu[i][b_w[0]];
            occ[b_w[0]] = stamp; occa[b_w[0]] = i;                       // agent i is now decided
            st.y.push_back(b_w[0]);
            st.aux = b_aux[0];
            chain_g += b_cost[0];
            chained = true;
            expansions++;
            if (expansions > max_exp || (expansions % 256 == 0 && Clock::now() > deadline)) {
                res.budget = true; res.expansions = expansions; res.states = store.count;
                return res;
            }
            goto expand_again;
        }
        // encode, hash and prefetch only the successors that will be stored
        // (a forced move continuing the chain above is never encoded)
        for (int q = 0; q < nb_succ; q++) {
            if (done) z2.back() = b_w[q]; else y2.back() = b_w[q];
            uint16_t* bw = &b_buf[(size_t)q * store.K];
            store.encode(z2, y2, b_aux[q], soc ? b_cnt[q] : cnt2, bw);
            b_h[q] = StateStore::hash_words(bw, store.K);
            store.prefetch(b_h[q]);
        }
        for (int q = 0; q < nb_succ; q++) {
            int w = b_w[q];
            long long cost = b_cost[q];
            auto [sid2, inserted] = store.find_or_insert_h(&b_buf[(size_t)q * store.K], b_h[q]);
            if (inserted) { parent.push_back(-1); dist.push_back(LLONG_MAX); closed.push_back(0); }
            else if (closed[sid2]) continue;
            long long g2 = e.g + chain_g + cost;
            if (g2 < dist[sid2]) {
                dist[sid2] = g2; parent[sid2] = sid;
                long long t = tiebreak ? tb_base - tb.mu[i][zi] + tb.mu[i][w] : 0;
                if (t >= (1LL << 24)) t = (1LL << 24) - 1;   // sum-min key saturates (only tie-breaking)
                long long f2 = g2 + (dg ? h_succ(w, b_aux[q]) : 0);
                long long h2_base = f2 - g2;
                if (use_conf) {
                    for (int k = 0; k < n; k++) X[k] = k < i ? st.y[k] : st.z[k];
                    X[i] = w;
                    f2 += CB.value(X);
                }
                if (use_pair) {
                    // all pair terms of the parent, with the term of i's pair replaced
                    f2 += pair_sum;
                    int pi = PDB.partner[i];
                    if (pi >= 0) {
                        if (done) { z2.back() = w; f2 += pair_term(PDB, pi, z2, y2, *dg) - pair_term(PDB, pi, st.z, st.y, *dg); }
                        else { y2.back() = w; f2 += pair_term(PDB, pi, z2, y2, *dg) - pair_term(PDB, pi, st.z, st.y, *dg); }
                    }
                }
                if (greedy) f2 = g2 + W_ANY * (f2 - g2);     // weighted phase: g + W h
                (void)h2_base;
                Entry en = mk(f2, t, (int)y2.size(), counter++, sid2);
                en.g = g2;
                heap.push(en);
            }
        }
    }
    res.expansions = expansions; res.states = store.count;
    return res;
}

static unique_ptr<Neighborhood> parse_nb(const string& spec) {
    vector<string> f;
    stringstream ss(spec);
    string tok;
    while (getline(ss, tok, ':')) f.push_back(tok);
    if (f[0] == "path") return make_unique<PathDistance>(stod(f[1]));
    if (f[0] == "pointwise") return make_unique<Pointwise>(stoll(f[1]));
    if (f[0] == "agent") return make_unique<AgentDistance>(stoi(f[1]));
    if (f[0] == "combined") return make_unique<Combined>(stoi(f[1]), stoll(f[2]));
    if (f[0] == "window") {   // window:W[:K[:multi]], K < 0 = all agents may move
        int K = f.size() > 2 ? stoi(f[2]) : -1;
        bool multi = f.size() > 3 && f[3] == "multi";
        return make_unique<Window>(stoi(f[1]), K, multi);
    }
    if (f[0] == "gc") {
        int window = f.size() > 5 ? stoi(f[5]) : 2, mg = f.size() > 6 ? stoi(f[6]) : 4;
        int kf = f.size() > 7 ? stoi(f[7]) : 0, ci = f.size() > 8 ? stoi(f[8]) : 0;
        int cor = f.size() > 9 ? stoi(f[9]) : -1;
        return make_unique<GroupedClock>(stod(f[1]), stod(f[2]), f[3], f[4] == "1", window, mg, kf, ci, cor);
    }
    cerr << "unknown neighborhood " << spec << "\n";
    exit(2);
}

int main(int argc, char** argv) {
    if (argc < 3) { cerr << "usage: dpls input output [--nb SPEC]... [--tiebreak summin|none] [--time S] [--expansions K]\n"; return 2; }
    vector<string> specs;
    bool tiebreak = false;
    double time_limit = 60;
    long long max_exp = 300000, max_searches = 10000;
    long long beam = 0, pruned = 0;     // beam width (0: exact search), pruned searches
    bool early_stop = false;
    bool soc = false;
    bool soc_proxy = false;  // waiting at the goal costs 0, any other action 1;
                             // no counters, acceptance on the true Sum-of-Costs
    bool astar = false;        // A* with a consistent heuristic instead of Dijkstra
    bool enumerate = false;    // explore the whole neighborhood of the plan and
                               // report its number of states (no search)
    bool keep_radius = false;  // adaptive radii: keep the radius that improved
                               // instead of resetting all radii to the minimum
    bool vnd = false;   // variable neighborhood descent: restart from the first
                        // neighborhood after every improvement
    for (int a = 3; a < argc; a++) {
        string o = argv[a];
        if (o == "--nb") specs.push_back(argv[++a]);
        else if (o == "--tiebreak") tiebreak = string(argv[++a]) == "summin";
        else if (o == "--time") time_limit = stod(argv[++a]);
        else if (o == "--expansions") max_exp = stoll(argv[++a]);
        else if (o == "--beam") beam = stoll(argv[++a]);
        else if (o == "--early-stop") early_stop = true;   // also exact searches stop when no plan can be accepted
        else if (o == "--region-h") g_region_h = true;
        else if (o == "--conflict-h") g_conflict_h = true;
        else if (o == "--pair-h") g_pair_h = true;
        else if (o == "--anytime") g_anytime = true;
        else if (o == "--anytime-w") g_anytime_w = stoll(argv[++a]);
        else if (o == "--chain") g_chain = true;
        else if (o == "--threads") g_threads = max(1, atoi(argv[++a]));
        else if (o == "--max-searches") max_searches = stoll(argv[++a]);
        else if (o == "--objective") {
            string v = argv[++a];
            soc = v == "soc";
            soc_proxy = v == "soc_proxy";
        }
        else if (o == "--vnd") vnd = true;
        else if (o == "--keep-radius") keep_radius = true;
        else if (o == "--astar") astar = true;
        else if (o == "--enumerate") enumerate = true;
        else { cerr << "unknown option " << o << "\n"; return 2; }
    }
    if (specs.empty()) specs.push_back("combined:2:1");
    ifstream in(argv[1]);
    Graph G;
    int M;
    in >> G.N >> M;
    G.out.assign(G.N, {}); G.inn.assign(G.N, {});
    for (int e = 0; e < M; e++) { int u, v; in >> u >> v; if (u != v) { G.out[u].push_back(v); G.inn[v].push_back(u); } }
    G.finish();
    int n; in >> n;
    Config starts(n), goals(n);
    for (auto& x : starts) in >> x;
    for (auto& x : goals) in >> x;
    int T; in >> T;
    vector<Config> plan(T + 1, Config(n));
    for (auto& x : plan) for (auto& v : x) in >> v;

    auto t0 = Clock::now();
    auto deadline = t0 + chrono::duration_cast<Clock::duration>(chrono::duration<double>(time_limit));
    // A slot is a neighborhood with a fixed spec, or a template with "{R}"
    // and a radius range "template@rmin-rmax" (adaptive radius).
    struct Slot { string tmpl; int rmin = 0, rmax = 0, r = 0; bool adaptive = false;
                  long long budget = -1; bool cycle = false;
                  long long beam = -1;    // per-neighborhood beam width ("SPEC%W"); -1: global
                  // geometric growth ("template@rmin-rmax^"): after a search that
                  // ends without improvement and without pruning (its neighborhood
                  // was searched completely), the next radius is r + step and the
                  // step doubles; after a pruned search the step returns to 1
                  bool geometric = false; int step = 1;
                  // extension ("template@rmin-rmax+cap"): beyond rmax, the radius
                  // keeps growing by one, up to cap, as long as the searches end
                  // without improvement, pruning or exceeding their budget
                  int rcap = -1;
                  // geometric extension ("template@rmin-rmax+cap*"): beyond rmax the
                  // radius grows by a doubling step up to cap, also after pruned
                  // searches (it stops when a search exceeds its budget)
                  bool gext = false; };
    vector<Slot> slots;
    for (auto sp : specs) {
        Slot sl;
        size_t hash_pos = sp.find('#');          // per-neighborhood state budget: SPEC#K
        if (hash_pos != string::npos) {
            sl.budget = stoll(sp.substr(hash_pos + 1));
            sp = sp.substr(0, hash_pos);
        }
        size_t pct_pos = sp.find('%');           // per-neighborhood beam width: SPEC%W
        if (pct_pos != string::npos) {
            sl.beam = stoll(sp.substr(pct_pos + 1));
            sp = sp.substr(0, pct_pos);
        }
        size_t at = sp.find('@');
        if (at == string::npos) { sl.tmpl = sp; }
        else {
            sl.tmpl = sp.substr(0, at);
            string rng = sp.substr(at + 1);
            while (!rng.empty() && (rng.back() == '!' || rng.back() == '^' || rng.back() == '*')) {
                if (rng.back() == '!') sl.cycle = true;
                else if (rng.back() == '^') sl.geometric = true;
                else sl.gext = true;
                rng.pop_back();
            }
            size_t plus = rng.find('+');
            if (plus != string::npos) { sl.rcap = stoi(rng.substr(plus + 1)); rng = rng.substr(0, plus); }
            size_t dash = rng.find('-');
            sl.rmin = stoi(rng.substr(0, dash));
            sl.rmax = stoi(rng.substr(dash + 1));
            sl.r = sl.rmin;
            sl.adaptive = true;
        }
        slots.push_back(sl);
    }
    auto make_nb = [&](const Slot& sl) {
        string sp = sl.tmpl;
        size_t p = sp.find("{R}");
        if (p != string::npos) sp.replace(p, 3, to_string(sl.r));
        return parse_nb(sp);
    };
    long long increases = 0;
    int max_radius = 0;
    // per-neighborhood statistics: searches, improvements, objective gain, time
    vector<long long> sl_searches(slots.size(), 0), sl_impr(slots.size(), 0), sl_gain(slots.size(), 0);
    vector<double> sl_time(slots.size(), 0.0);
    vector<Config> cur = trim(plan);
    vector<vector<int>> dgoal;
    if (enumerate) {
        string sp = specs[0];
        if (sp.find('#') != string::npos) sp = sp.substr(0, sp.find('#'));
        int r0 = 0;
        if (sp.find('@') != string::npos) {
            r0 = stoi(sp.substr(sp.find('@') + 1));
            sp = sp.substr(0, sp.find('@'));
        }
        if (sp.find("{R}") != string::npos) sp.replace(sp.find("{R}"), 3, to_string(r0));
        auto nbp = parse_nb(sp);
        SearchResult r = neighborhood_search(G, goals, cur, *nbp, false, max_exp,
                                             Clock::now() + chrono::hours(24), false, false,
                                             nullptr, true);
        printf("{\"states\": %lld, \"budget_exceeded\": %s, \"T\": %d, \"n\": %d}\n",
               r.states, r.budget ? "true" : "false", (int)cur.size() - 1, n);
        return 0;
    }
    int mk0 = makespan(cur);
    long long soc0 = sum_of_costs(cur, goals);
    auto obj = [&](const vector<Config>& p) {
        return (soc || soc_proxy) ? sum_of_costs(p, goals) : (long long)makespan(p);
    };
    long long searches = 0, improvements = 0, budget = 0, maxstates = 0;
    string impr_at;              // [slot, radius, gain] of every accepted improvement
    int stale = 0;
    size_t j = 0;
    bool timeout = false;
    while (stale < (int)slots.size() && searches < max_searches) {
        if (Clock::now() > deadline) { timeout = true; break; }
        size_t sidx = j % slots.size();
        Slot& sl = slots[sidx];
        auto ts = Clock::now();
        if (sl.tmpl == "retime" || sl.tmpl == "swap") {
            SearchResult r = order_search(G, goals, cur, sl.tmpl, soc || soc_proxy, deadline);
            sl_time[sidx] += chrono::duration<double>(Clock::now() - ts).count();
            sl_searches[sidx]++;
            searches++;
            if (r.found && obj(r.plan) < obj(cur)) {
                sl_impr[sidx]++;
                sl_gain[sidx] += obj(cur) - obj(r.plan);
                cur = trim(r.plan); improvements++; stale = 0;
                if (!keep_radius) for (auto& x : slots) x.r = x.rmin;
                if (vnd) j = 0; else j++;
            } else { stale++; j++; }
            continue;
        }
        if (astar && dgoal.empty()) dgoal = goal_distances(G, goals);
        if (g_threads > 1 && sl.adaptive && sl.cycle && sl.r < sl.rmax) {
            // Parallel batch: the next T candidates of this cyclic neighborhood are
            // searched at the same time, each with its own neighborhood object; the
            // improvement with the lowest objective (then the lowest index) is
            // accepted, so the result does not depend on the thread timing. With no
            // improvement this is equivalent to T failed sequential searches.
            int nbatch = min(g_threads, sl.rmax - sl.r + 1);
            vector<SearchResult> R(nbatch);
            long long inc = obj(cur);
            {
                vector<thread> th;
                for (int b = 0; b < nbatch; b++) {
                    Slot s2 = sl; s2.r = sl.r + b;
                    th.emplace_back([&, s2, b]() {
                        auto nbq = make_nb(s2);
                        long long bw = s2.beam >= 0 ? s2.beam : beam;
                        R[b] = neighborhood_search(G, goals, cur, *nbq, tiebreak,
                                                   s2.budget > 0 ? min(s2.budget, max_exp) : max_exp,
                                                   deadline, soc, soc_proxy, astar ? &dgoal : nullptr,
                                                   false, bw, ((bw > 0 || early_stop) && (soc || soc_proxy)) ? inc : -1);
                    });
                }
                for (auto& t : th) t.join();
            }
            sl_time[sidx] += chrono::duration<double>(Clock::now() - ts).count();
            int best = -1; long long bestv = inc;
            for (int b = 0; b < nbatch; b++) {
                sl_searches[sidx]++; searches++;
                maxstates = max(maxstates, R[b].states);
                if (R[b].budget) budget++;
                if (R[b].pruned) pruned++;
                if (R[b].found) { long long v = obj(R[b].plan); if (v < bestv) { bestv = v; best = b; } }
            }
            max_radius = max(max_radius, sl.r + nbatch - 1);
            if (best >= 0) {
                impr_at += (impr_at.empty() ? "" : ", ") + string("[") + to_string(sidx) + ", " +
                           to_string(sl.r + best) + ", " + to_string(inc - bestv) + "]";
                sl_impr[sidx]++;
                sl_gain[sidx] += inc - bestv;
                cur = trim(R[best].plan); improvements++; stale = 0;
                sl.step = 1;
                sl.r = sl.r + best;                      // the candidate that improved
                if (!keep_radius) for (auto& x : slots) x.r = x.rmin;
                if (vnd) j = 0; else j++;
                continue;
            }
            int last = sl.r + nbatch - 1;
            increases += nbatch - (last >= sl.rmax ? 1 : 0);
            if (last < sl.rmax) { sl.r = last + 1; continue; }
            sl.r = sl.rmin; sl.step = 1; stale++; j++;
            continue;
        }
        auto nbp = make_nb(sl);
        SearchResult r = neighborhood_search(G, goals, cur, *nbp, tiebreak,
                                             sl.budget > 0 ? min(sl.budget, max_exp) : max_exp,
                                             deadline, soc, soc_proxy, astar ? &dgoal : nullptr,
                                             false, sl.beam >= 0 ? sl.beam : beam,
                                             // the search cost is in Sum-of-Costs units only
                                             // with these objectives (with the makespan it
                                             // is n times the makespan): stop only then
                                             (((sl.beam >= 0 ? sl.beam : beam) > 0 || early_stop) && (soc || soc_proxy))
                                                 ? obj(cur) : -1);
        sl_time[sidx] += chrono::duration<double>(Clock::now() - ts).count();
        sl_searches[sidx]++;
        searches++;
        maxstates = max(maxstates, r.states);
        if (sl.adaptive) max_radius = max(max_radius, sl.r);
        if (r.budget) budget++;
        if (r.pruned) pruned++;
        if (r.found && obj(r.plan) < obj(cur)) {
            impr_at += (impr_at.empty() ? "" : ", ") + string("[") + to_string(sidx) + ", " +
                       to_string(sl.adaptive ? sl.r : -1) + ", " + to_string(obj(cur) - obj(r.plan)) + "]";
            sl_impr[sidx]++;
            sl_gain[sidx] += obj(cur) - obj(r.plan);
            cur = trim(r.plan); improvements++; stale = 0;
            sl.step = 1;
            if (!keep_radius)
                for (auto& x : slots) x.r = x.rmin;      // back to small radii
            if (vnd) j = 0; else j++;
            continue;
        }
        if (sl.adaptive && sl.rcap > sl.rmax && sl.r >= sl.rmax && sl.r < sl.rcap
            && !r.budget && (sl.gext || !r.pruned)) {    // extension beyond rmax
            if (sl.gext) { sl.r = min(sl.rcap, sl.r + sl.step); sl.step *= 2; }
            else sl.r++;
            increases++;
            continue;
        }
        if (sl.adaptive && (sl.cycle || !r.budget) && sl.r < sl.rmax) {  // same neighborhood, next radius / candidate
            if (sl.geometric) {
                int inc = r.pruned ? 1 : sl.step;
                sl.r = min(sl.rmax, sl.r + inc);
                sl.step = r.pruned ? 1 : 2 * sl.step;
            } else sl.r++;
            increases++;
            continue;
        }
        sl.r = sl.rmin;
        sl.step = 1;
        stale++;
        j++;
    }
    double dt = chrono::duration<double>(Clock::now() - t0).count();
    auto vec = [](auto& v) {
        string o = "[";
        for (size_t k = 0; k < v.size(); k++) { o += to_string(v[k]); if (k + 1 < v.size()) o += ", "; }
        return o + "]";
    };
    ofstream out(argv[2]);
    for (auto& x : cur) { for (int k = 0; k < n; k++) out << x[k] << (k + 1 < n ? ' ' : '\n'); }
    printf("{\"objective\": \"%s\", \"makespan0\": %d, \"makespan\": %d, \"soc0\": %lld, "
           "\"soc\": %lld, \"searches\": %lld, \"improvements\": %lld, "
           "\"budget_exceeded\": %lld, \"pruned\": %lld, \"states\": %lld, \"time\": %.4f, \"timeout\": %s, "
           "\"radius_increases\": %lld, \"max_radius\": %d, \"slot_searches\": %s, "
           "\"slot_improvements\": %s, \"slot_gain\": %s, \"slot_time\": %s, \"impr_at\": [%s], "
           "\"exp_below\": %lld, \"exp_at\": %lld, \"found_searches\": %lld, \"budget_searches\": %lld, \"budget_gap\": %lld}\n",
           soc ? "soc" : (soc_proxy ? "soc_proxy" : "makespan"), mk0, makespan(cur), soc0, sum_of_costs(cur, goals),
           searches, improvements, budget, pruned, maxstates, dt, timeout ? "true" : "false",
           increases, max_radius, vec(sl_searches).c_str(), vec(sl_impr).c_str(),
           vec(sl_gain).c_str(), vec(sl_time).c_str(), impr_at.c_str(),
           g_exp_below.load(), g_exp_at.load(), g_found.load(), g_budget_n.load(), g_budget_gap.load());
    (void)g_anytime_found;
    return 0;
}
