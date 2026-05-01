from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import random

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl


@dataclass(frozen=True)
class FuzzyParams:
    queue_low: tuple[float, float, float]
    queue_medium: tuple[float, float, float]
    queue_high: tuple[float, float, float, float]
    wait_short: tuple[float, float, float]
    wait_medium: tuple[float, float, float]
    wait_long: tuple[float, float, float, float]
    occ_low: tuple[float, float, float]
    occ_medium: tuple[float, float, float]
    occ_high: tuple[float, float, float, float]
    green_short: tuple[float, float, float]
    green_medium: tuple[float, float, float]
    green_long: tuple[float, float, float, float]


BASELINE_PARAMS = FuzzyParams(
    queue_low=(0, 5, 14), queue_medium=(8, 20, 34), queue_high=(26, 40, 60, 60),
    wait_short=(0, 12, 35), wait_medium=(20, 50, 90), wait_long=(70, 110, 180, 180),
    occ_low=(0, 18, 40), occ_medium=(30, 55, 78), occ_high=(70, 86, 100, 100),
    green_short=(10, 16, 24), green_medium=(22, 34, 45), green_long=(40, 53, 60, 60),
)


def build_fuzzy_control_system(params: FuzzyParams) -> ctrl.ControlSystem:
    queue = ctrl.Antecedent(np.arange(0, 61, 1), "queue")
    wait_time = ctrl.Antecedent(np.arange(0, 181, 1), "wait_time")
    occupancy = ctrl.Antecedent(np.arange(0, 101, 1), "occupancy")
    green_time = ctrl.Consequent(np.arange(10, 61, 1), "green_time")

    queue["low"] = fuzzy.trimf(queue.universe, params.queue_low)
    queue["medium"] = fuzzy.trimf(queue.universe, params.queue_medium)
    queue["high"] = fuzzy.trapmf(queue.universe, params.queue_high)

    wait_time["short"] = fuzzy.trimf(wait_time.universe, params.wait_short)
    wait_time["medium"] = fuzzy.trimf(wait_time.universe, params.wait_medium)
    wait_time["long"] = fuzzy.trapmf(wait_time.universe, params.wait_long)

    occupancy["low"] = fuzzy.trimf(occupancy.universe, params.occ_low)
    occupancy["medium"] = fuzzy.trimf(occupancy.universe, params.occ_medium)
    occupancy["high"] = fuzzy.trapmf(occupancy.universe, params.occ_high)

    green_time["short"] = fuzzy.trimf(green_time.universe, params.green_short)
    green_time["medium"] = fuzzy.trimf(green_time.universe, params.green_medium)
    green_time["long"] = fuzzy.trapmf(green_time.universe, params.green_long)

    rules = [
        ctrl.Rule(queue["high"] | occupancy["high"], green_time["long"]),
        ctrl.Rule(wait_time["long"], green_time["long"]),
        ctrl.Rule(queue["medium"] & occupancy["medium"], green_time["medium"]),
        ctrl.Rule(queue["low"] & wait_time["short"] & occupancy["low"], green_time["short"]),
        ctrl.Rule(queue["high"] & wait_time["short"], green_time["medium"]),
        ctrl.Rule(queue["low"] & wait_time["long"], green_time["medium"]),
        ctrl.Rule(wait_time["medium"] & occupancy["high"], green_time["long"]),
    ]
    return ctrl.ControlSystem(rules)


def evaluate_green_time(system: ctrl.ControlSystem, queue: float, wait_time: float, occupancy: float) -> float:
    sim = ctrl.ControlSystemSimulation(system, cache=False)
    sim.input["queue"] = float(np.clip(queue, 0, 60))
    sim.input["wait_time"] = float(np.clip(wait_time, 0, 180))
    sim.input["occupancy"] = float(np.clip(occupancy, 0, 100))
    sim.compute()
    out = sim.output.get("green_time", np.nan)
    if np.isnan(out):
        out = 12 + 0.55 * queue + 0.07 * wait_time + 0.13 * occupancy
    return float(np.clip(out, 10, 60))


def _fitness(params: FuzzyParams, samples: List[Dict[str, float]]) -> float:
    system = build_fuzzy_control_system(params)
    mse = 0.0
    for s in samples:
        pred = evaluate_green_time(system, s["queue"], s["wait_time"], s["occupancy"])
        mse += (pred - s["target"]) ** 2
    return mse / len(samples)


def _mutate_value(v: float, lo: float, hi: float, sigma: float = 2.5) -> float:
    return float(np.clip(v + random.gauss(0, sigma), lo, hi))


def optimize_membership_with_ga(generations: int = 60, pop_size: int = 40, seed: int = 42) -> FuzzyParams:
    random.seed(seed)
    np.random.seed(seed)

    samples = [
        {
            "queue": q,
            "wait_time": w,
            "occupancy": o,
            "target": float(np.clip(10 + 0.52 * q + 0.08 * w + 0.18 * o, 10, 60)),
        }
        for q in [0, 8, 16, 24, 35, 45, 60]
        for w in [0, 10, 20, 40, 70, 120, 180]
        for o in [0, 20, 40, 60, 80, 100]
    ]

    def random_individual() -> FuzzyParams:
        return FuzzyParams(
            queue_low=(0, random.uniform(3, 8), random.uniform(10, 18)),
            queue_medium=(random.uniform(6, 12), random.uniform(16, 24), random.uniform(28, 38)),
            queue_high=(random.uniform(22, 30), random.uniform(36, 45), 60, 60),
            wait_short=(0, random.uniform(8, 16), random.uniform(28, 45)),
            wait_medium=(random.uniform(16, 30), random.uniform(38, 60), random.uniform(75, 100)),
            wait_long=(random.uniform(55, 90), random.uniform(95, 130), 180, 180),
            occ_low=(0, random.uniform(12, 24), random.uniform(32, 45)),
            occ_medium=(random.uniform(26, 38), random.uniform(48, 62), random.uniform(70, 84)),
            occ_high=(random.uniform(62, 74), random.uniform(82, 92), 100, 100),
            green_short=(10, random.uniform(14, 18), random.uniform(22, 26)),
            green_medium=(random.uniform(20, 26), random.uniform(30, 38), random.uniform(42, 48)),
            green_long=(random.uniform(36, 44), random.uniform(50, 56), 60, 60),
        )

    def crossover(a: FuzzyParams, b: FuzzyParams) -> FuzzyParams:
        fields = []
        for k in a.__dataclass_fields__:
            av, bv = getattr(a, k), getattr(b, k)
            fields.append(tuple((x + y) / 2 for x, y in zip(av, bv)))
        return FuzzyParams(*fields)

    def mutate(p: FuzzyParams) -> FuzzyParams:
        ql = (0, _mutate_value(p.queue_low[1], 2, 9), _mutate_value(p.queue_low[2], 8, 20))
        qm = (_mutate_value(p.queue_medium[0], 5, 15), _mutate_value(p.queue_medium[1], 14, 27), _mutate_value(p.queue_medium[2], 24, 42))
        qh = (_mutate_value(p.queue_high[0], 18, 34), _mutate_value(p.queue_high[1], 32, 48), 60, 60)
        ws = (0, _mutate_value(p.wait_short[1], 6, 20), _mutate_value(p.wait_short[2], 20, 55))
        wm = (_mutate_value(p.wait_medium[0], 12, 34), _mutate_value(p.wait_medium[1], 30, 70), _mutate_value(p.wait_medium[2], 60, 120))
        wl = (_mutate_value(p.wait_long[0], 50, 100), _mutate_value(p.wait_long[1], 80, 145), 180, 180)
        ol = (0, _mutate_value(p.occ_low[1], 8, 30), _mutate_value(p.occ_low[2], 25, 50))
        om = (_mutate_value(p.occ_medium[0], 22, 42), _mutate_value(p.occ_medium[1], 40, 68), _mutate_value(p.occ_medium[2], 64, 90))
        oh = (_mutate_value(p.occ_high[0], 56, 80), _mutate_value(p.occ_high[1], 76, 96), 100, 100)
        gs = (10, _mutate_value(p.green_short[1], 12, 20), _mutate_value(p.green_short[2], 20, 30))
        gm = (_mutate_value(p.green_medium[0], 18, 30), _mutate_value(p.green_medium[1], 26, 42), _mutate_value(p.green_medium[2], 38, 52))
        gl = (_mutate_value(p.green_long[0], 34, 46), _mutate_value(p.green_long[1], 46, 58), 60, 60)
        return FuzzyParams(ql, qm, qh, ws, wm, wl, ol, om, oh, gs, gm, gl)

    population = [BASELINE_PARAMS] + [random_individual() for _ in range(pop_size - 1)]
    for _ in range(generations):
        ranked = sorted(population, key=lambda p: _fitness(p, samples))
        elites = ranked[: max(4, pop_size // 5)]
        next_population = elites.copy()
        while len(next_population) < pop_size:
            a, b = random.sample(elites, 2)
            next_population.append(mutate(crossover(a, b)))
        population = next_population
    return min(population, key=lambda p: _fitness(p, samples))


def plot_membership_functions(params: FuzzyParams, out_dir: str = "plots") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        ("queue", np.arange(0, 61, 1), [params.queue_low, params.queue_medium, params.queue_high], ["low", "medium", "high"]),
        ("wait_time", np.arange(0, 181, 1), [params.wait_short, params.wait_medium, params.wait_long], ["short", "medium", "long"]),
        ("occupancy", np.arange(0, 101, 1), [params.occ_low, params.occ_medium, params.occ_high], ["low", "medium", "high"]),
        ("green_time", np.arange(10, 61, 1), [params.green_short, params.green_medium, params.green_long], ["short", "medium", "long"]),
    ]

    for name, universe, mfs, labels in specs:
        plt.figure(figsize=(8, 4))
        for mf, label in zip(mfs, labels):
            y = fuzzy.trimf(universe, mf) if len(mf) == 3 else fuzzy.trapmf(universe, mf)
            plt.plot(universe, y, linewidth=2, label=label)
        plt.title(f"Membership Functions - {name}")
        plt.xlabel(name)
        plt.ylabel("Membership")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"{name}_mfs.png", dpi=140)
        plt.close()
