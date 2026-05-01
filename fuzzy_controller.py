import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


OPTIMIZED_PARAMS = FuzzyParams(
    queue_low=(0.0, 3.0, 10.0), queue_medium=(7.0, 18.0, 30.0), queue_high=(24.0, 35.0, 60.0, 60.0),
    wait_short=(0.0, 7.0, 20.0), wait_medium=(12.0, 30.0, 48.0), wait_long=(40.0, 55.0, 120.0, 120.0),
    occ_low=(0.0, 12.0, 35.0), occ_medium=(25.0, 50.0, 72.0), occ_high=(65.0, 82.0, 100.0, 100.0),
    green_short=(10.0, 14.0, 24.0), green_medium=(18.0, 32.0, 44.0), green_long=(36.0, 50.0, 60.0, 60.0),
)


def build_fuzzy_system(params: FuzzyParams = OPTIMIZED_PARAMS) -> ctrl.ControlSystemSimulation:
    queue = ctrl.Antecedent(np.arange(0, 61, 1), "queue")
    wait_time = ctrl.Antecedent(np.arange(0, 121, 1), "wait_time")
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
        ctrl.Rule(queue["high"] | wait_time["long"] | occupancy["high"], green_time["long"]),
        ctrl.Rule(queue["medium"] & wait_time["medium"] & occupancy["medium"], green_time["medium"]),
        ctrl.Rule(queue["low"] & wait_time["short"] & occupancy["low"], green_time["short"]),
        ctrl.Rule(queue["high"] & wait_time["short"], green_time["medium"]),
        ctrl.Rule(queue["low"] & wait_time["long"], green_time["medium"]),
        ctrl.Rule(occupancy["high"] & wait_time["medium"], green_time["long"]),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def _simulate_controller(params: FuzzyParams, samples: List[Dict[str, float]]) -> float:
    errors = []
    for sample in samples:
        sim = build_fuzzy_system(params)
        sim.input["queue"] = sample["queue"]
        sim.input["wait_time"] = sample["wait_time"]
        sim.input["occupancy"] = sample["occupancy"]
        sim.compute()
        errors.append((sim.output.get("green_time", 25.0) - sample["target_green"]) ** 2)
    return float(np.mean(errors))


def optimize_membership_with_ga(generations: int = 60, pop_size: int = 40, seed: int = 42) -> FuzzyParams:
    random.seed(seed)
    np.random.seed(seed)
    samples = [
        {"queue": q, "wait_time": w, "occupancy": o, "target_green": np.clip(10 + 0.55 * q + 0.18 * w + 0.2 * o, 10, 60)}
        for q in [0, 10, 20, 35, 50, 60] for w in [0, 10, 20, 40, 80, 120] for o in [0, 20, 40, 60, 80, 100]
    ]

    def random_params() -> FuzzyParams:
        return FuzzyParams(
            queue_low=(0.0, random.uniform(1, 5), random.uniform(7, 14)),
            queue_medium=(random.uniform(6, 12), random.uniform(14, 22), random.uniform(25, 35)),
            queue_high=(random.uniform(18, 28), random.uniform(30, 40), 60.0, 60.0),
            wait_short=(0.0, random.uniform(4, 12), random.uniform(15, 25)),
            wait_medium=(random.uniform(10, 20), random.uniform(24, 36), random.uniform(42, 60)),
            wait_long=(random.uniform(30, 45), random.uniform(50, 70), 120.0, 120.0),
            occ_low=(0.0, random.uniform(8, 20), random.uniform(30, 40)),
            occ_medium=(random.uniform(20, 35), random.uniform(45, 58), random.uniform(65, 78)),
            occ_high=(random.uniform(58, 72), random.uniform(75, 90), 100.0, 100.0),
            green_short=(10.0, random.uniform(12, 18), random.uniform(22, 30)),
            green_medium=(random.uniform(16, 24), random.uniform(28, 36), random.uniform(40, 48)),
            green_long=(random.uniform(32, 42), random.uniform(46, 55), 60.0, 60.0),
        )

    def crossover(a: FuzzyParams, b: FuzzyParams) -> FuzzyParams:
        child = []
        for key in a.__dataclass_fields__:
            av, bv = getattr(a, key), getattr(b, key)
            child.append(tuple((x + y) / 2 for x, y in zip(av, bv)))
        return FuzzyParams(*child)

    def mutate(p: FuzzyParams, rate: float = 0.15) -> FuzzyParams:
        out = []
        for key in p.__dataclass_fields__:
            vals = list(getattr(p, key))
            for i in range(len(vals)):
                if random.random() < rate and vals[i] not in (0.0, 10.0, 60.0, 100.0, 120.0):
                    vals[i] += random.uniform(-3, 3)
            out.append(tuple(vals))
        return FuzzyParams(*out)

    population = [random_params() for _ in range(pop_size)]
    for _ in range(generations):
        ranked = sorted(population, key=lambda p: _simulate_controller(p, samples))
        elites = ranked[: max(4, pop_size // 5)]
        population = elites + [mutate(crossover(*random.sample(elites, 2))) for _ in range(pop_size - len(elites))]
    return min(population, key=lambda p: _simulate_controller(p, samples))


def save_membership_plots(params: FuzzyParams = OPTIMIZED_PARAMS, out_dir: str = "plots") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    defs = [
        (np.arange(0, 61, 1), [params.queue_low, params.queue_medium, params.queue_high], ["low", "medium", "high"], "queue"),
        (np.arange(0, 121, 1), [params.wait_short, params.wait_medium, params.wait_long], ["short", "medium", "long"], "wait_time"),
        (np.arange(0, 101, 1), [params.occ_low, params.occ_medium, params.occ_high], ["low", "medium", "high"], "occupancy"),
        (np.arange(10, 61, 1), [params.green_short, params.green_medium, params.green_long], ["short", "medium", "long"], "green_time"),
    ]
    for universe, mfs, labels, name in defs:
        lines = []
        for mf, label in zip(mfs, labels):
            y = fuzzy.trimf(universe, mf) if len(mf) == 3 else fuzzy.trapmf(universe, mf)
            pairs = ";".join(f"{x},{float(v):.4f}" for x, v in zip(universe, y))
            lines.append(f"{label}: {pairs}")
        (out / f"{name}_mfs.csv").write_text("\n".join(lines))
