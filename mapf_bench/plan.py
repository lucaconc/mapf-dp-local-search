# SPDX-License-Identifier: GPL-3.0-or-later
"""Plans as sequences of joint configurations, with validation and metrics."""


class InvalidPlan(Exception):
    pass


def validate(graph, starts, goals, plan):
    """Check that a plan (list of configurations x^0..x^T) is feasible.

    Agents move along arcs or wait, never share a node, never swap,
    start at `starts` and end at `goals`.
    """
    if tuple(plan[0]) != tuple(starts):
        raise InvalidPlan("plan does not start at the initial configuration")
    if tuple(plan[-1]) != tuple(goals):
        raise InvalidPlan("plan does not end at the goal configuration")
    n = len(starts)
    for t, x in enumerate(plan):
        if len(set(x)) != n:
            raise InvalidPlan(f"vertex collision at time {t}")
    for t in range(1, len(plan)):
        z, y = plan[t - 1], plan[t]
        for k in range(n):
            if y[k] not in graph.out[z[k]]:
                raise InvalidPlan(f"agent {k} uses a missing arc at time {t}")
        pos = {v: k for k, v in enumerate(z)}
        for k in range(n):
            j = pos.get(y[k])
            if j is not None and j != k and y[j] == z[k]:
                raise InvalidPlan(f"swap between agents {k} and {j} at time {t}")
    return True


def trim(plan):
    """Remove trailing steps in which every agent waits at its final node."""
    plan = [tuple(x) for x in plan]
    while len(plan) > 1 and plan[-1] == plan[-2]:
        plan.pop()
    return plan


def makespan(plan):
    return len(trim(plan)) - 1


def arrival_times(plan, goals):
    """Arrival time of each agent: 1 + last time it is not at its goal."""
    T = len(plan) - 1
    arr = []
    for k, g in enumerate(goals):
        a = 0
        for t in range(T, -1, -1):
            if plan[t][k] != g:
                a = t + 1
                break
        arr.append(a)
    return arr


def sum_of_costs(plan, goals):
    return sum(arrival_times(plan, goals))


def fuel(plan):
    return sum(1 for t in range(1, len(plan)) for k in range(len(plan[0]))
               if plan[t][k] != plan[t - 1][k])


def paths_to_plan(paths):
    """Per-agent node sequences -> joint configurations (padding at goals)."""
    T = max(len(p) for p in paths) - 1
    return [tuple(p[min(t, len(p) - 1)] for p in paths) for t in range(T + 1)]


def plan_to_paths(plan, goals):
    """Joint configurations -> per-agent node sequences, cut at arrival."""
    arr = arrival_times(plan, goals)
    return [[plan[t][k] for t in range(arr[k] + 1)] for k in range(len(goals))]


def serialize(graph, plan):
    """Stretch a plan so that, as in rule-based solvers, one agent moves at a time.

    Within each joint step, agents whose target node is free move alone, one
    after the other; agents left in a rotation move together. The result is a
    valid plan through the same nodes, typically much longer.
    """
    plan = [tuple(x) for x in plan]
    out = [plan[0]]
    cur = list(plan[0])
    for t in range(1, len(plan)):
        movers = [k for k in range(len(cur)) if plan[t][k] != cur[k]]
        while movers:
            occupied = set(cur)
            free = [k for k in movers if plan[t][k] not in occupied]
            if free:
                k = free[0]
                cur[k] = plan[t][k]
                movers.remove(k)
            else:  # remaining movers form rotations: move them together
                for k in movers:
                    cur[k] = plan[t][k]
                movers = []
            out.append(tuple(cur))
    return out
