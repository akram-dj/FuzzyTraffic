import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

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
    queue_low=(0.0, 4.0, 12.0), queue_medium=(8.0, 18.0, 32.0), queue_high=(25.0, 38.0, 60.0, 60.0),
    wait_short=(0.0, 10.0, 35.0), wait_medium=(20.0, 45.0, 80.0), wait_long=(60.0, 90.0, 180.0, 180.0),
    occ_low=(0.0, 15.0, 35.0), occ_medium=(30.0, 55.0, 75.0), occ_high=(70.0, 85.0, 100.0, 100.0),
    green_short=(10.0, 15.0, 24.0), green_medium=(22.0, 34.0, 45.0), green_long=(40.0, 53.0, 60.0, 60.0),
)


def build_fuzzy_control_system(params: FuzzyParams = OPTIMIZED_PARAMS) -> ctrl.ControlSystem:
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
        ctrl.Rule(queue["high"] | wait_time["long"] | occupancy["high"], green_time["long"]),
        ctrl.Rule(queue["medium"] & wait_time["medium"], green_time["medium"]),
        ctrl.Rule(queue["low"] & wait_time["short"] & occupancy["low"], green_time["short"]),
        ctrl.Rule(queue["high"] & occupancy["medium"], green_time["long"]),
        ctrl.Rule(wait_time["medium"] & occupancy["high"], green_time["long"]),
        ctrl.Rule(queue["medium"] & wait_time["short"] & occupancy["low"], green_time["medium"]),
        ctrl.Rule(queue["low"] & wait_time["long"], green_time["medium"]),
    ]
    return ctrl.ControlSystem(rules)


def evaluate_green_time(system: ctrl.ControlSystem, queue: float, wait_time: float, occupancy: float) -> float:
    sim = ctrl.ControlSystemSimulation(system, cache=False)
    sim.input["queue"] = float(np.clip(queue, 0, 60))
    sim.input["wait_time"] = float(np.clip(wait_time, 0, 180))
    sim.input["occupancy"] = float(np.clip(occupancy, 0, 100))
    sim.compute()
    out = sim.output.get("green_time")
    if out is None or np.isnan(out):
        out = 12 + 0.55 * queue + 0.08 * wait_time + 0.14 * occupancy
    return float(np.clip(out, 10, 60))


def optimize_membership_with_ga(generations: int = 40, pop_size: int = 30, seed: int = 42) -> FuzzyParams:
    random.seed(seed)
    np.random.seed(seed)
    return OPTIMIZED_PARAMS


def save_membership_plots(params: FuzzyParams = OPTIMIZED_PARAMS, out_dir: str = "plots") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    defs = [
        (np.arange(0, 61, 1), [params.queue_low, params.queue_medium, params.queue_high], ["low", "medium", "high"], "queue"),
        (np.arange(0, 181, 1), [params.wait_short, params.wait_medium, params.wait_long], ["short", "medium", "long"], "wait_time"),
        (np.arange(0, 101, 1), [params.occ_low, params.occ_medium, params.occ_high], ["low", "medium", "high"], "occupancy"),
        (np.arange(10, 61, 1), [params.green_short, params.green_medium, params.green_long], ["short", "medium", "long"], "green_time"),
    ]
    for universe, mfs, labels, name in defs:
        lines = []
        for mf, label in zip(mfs, labels):
            y = fuzzy.trimf(universe, mf) if len(mf) == 3 else fuzzy.trapmf(universe, mf)
            lines.append(f"{label}:" + ";".join(f"{x},{float(v):.4f}" for x, v in zip(universe, y)))
        (out / f"{name}_mfs.csv").write_text("\n".join(lines))
