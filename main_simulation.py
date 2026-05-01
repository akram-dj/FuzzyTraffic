import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set the environment variable 'SUMO_HOME'")

import numpy as np
import traci

from fuzzy_controller import (
    BASELINE_PARAMS,
    build_fuzzy_control_system,
    evaluate_green_time,
    optimize_membership_with_ga,
    plot_membership_functions,
)

TLS_ID = "J1"
PHASE_EW_GREEN = 0
PHASE_NS_GREEN = 2
MIN_GREEN = 10.0
MAX_GREEN = 60.0
DECISION_HORIZON_SEC = 2.0


def _get_opposite_detector_ids(current_phase: int) -> tuple[list[str], str]:
    if current_phase == PHASE_NS_GREEN:
        return ["e2_0", "e2_1"], "EW (East-West)"
    return ["e2_2", "e2_3"], "NS (North-South)"


def _get_metrics(detector_ids: list[str]) -> tuple[float, float, float]:
    queue = sum(traci.lanearea.getJamLengthVehicle(det_id) for det_id in detector_ids)
    occupancy = float(np.mean([traci.lanearea.getLastStepOccupancy(det_id) for det_id in detector_ids]))

    vehicle_ids = set()
    for det_id in detector_ids:
        vehicle_ids.update(traci.lanearea.getLastStepVehicleIDs(det_id))

    wait_time = max((traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids), default=0.0)
    return float(queue), float(wait_time), occupancy


def run(use_ga_optimization: bool = False, use_gui: bool = True) -> None:
    print("🚦 Starting Intelligent Traffic Light Control Simulation...\n")
    if use_ga_optimization:
        print("[Init] Running GA optimization (fast mode)...")
        params = optimize_membership_with_ga(generations=12, pop_size=16, seed=42, verbose=True)
    else:
        print("[Init] Using baseline params (skip GA for fast startup).")
        params = BASELINE_PARAMS

    plot_membership_functions(params)
    fuzzy_system = build_fuzzy_control_system(params)

    print(f"[Init] Controller ready. GA optimized={use_ga_optimization}")
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    print(f"[Init] Launching {sumo_binary}...")
    traci.start([sumo_binary, "-c", "config.sumocfg", "--start", "true", "--quit-on-end", "true"])
    print(f"[Init] {sumo_binary} started. You can now control/observe the simulation.")

    next_green_duration = 20.0
    last_phase = -1
    decision_made = False

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        now = traci.simulation.getTime()
        time_left = traci.trafficlight.getNextSwitch(TLS_ID) - now

        if current_phase != last_phase:
            decision_made = False
            if current_phase in (PHASE_EW_GREEN, PHASE_NS_GREEN):
                traci.trafficlight.setPhaseDuration(TLS_ID, next_green_duration)
                dir_name = "EW" if current_phase == PHASE_EW_GREEN else "NS"
                print(f"[{now:.1f}s] {dir_name} GREEN start -> {next_green_duration:.1f}s")
            last_phase = current_phase

        if not decision_made and current_phase in (PHASE_EW_GREEN, PHASE_NS_GREEN) and time_left <= DECISION_HORIZON_SEC:
            detector_ids, next_direction = _get_opposite_detector_ids(current_phase)
            queue, wait_time, occupancy = _get_metrics(detector_ids)

            next_green_duration = evaluate_green_time(fuzzy_system, queue, wait_time, occupancy)
            next_green_duration = float(np.clip(next_green_duration, MIN_GREEN, MAX_GREEN))

            print(
                f"[{now:.1f}s] {next_direction} demand | queue={queue:.0f}, wait={wait_time:.1f}s, "
                f"occupancy={occupancy:.1f}% -> next_green={next_green_duration:.1f}s"
            )
            decision_made = True

    traci.close()
    print("\nSimulation finished successfully.")


if __name__ == "__main__":
    run(use_ga_optimization=False, use_gui=True)
