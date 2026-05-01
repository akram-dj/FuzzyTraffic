import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set the environment variable 'SUMO_HOME'")

import numpy as np
import traci

from fuzzy_controller import OPTIMIZED_PARAMS, build_fuzzy_control_system, evaluate_green_time, save_membership_plots

TLS_ID = "J1"
EW_PHASE = 0
NS_PHASE = 2
MIN_GREEN = 10.0
MAX_GREEN = 60.0


def _get_opposite_metrics(current_phase: int) -> tuple[float, float, float, str]:
    if current_phase == NS_PHASE:
        detector_ids = ["e2_0", "e2_1"]
        direction = "EW (East-West)"
    else:
        detector_ids = ["e2_2", "e2_3"]
        direction = "NS (North-South)"

    queue = sum(traci.lanearea.getJamLengthVehicle(det) for det in detector_ids)
    occupancy = float(np.mean([traci.lanearea.getLastStepOccupancy(det) for det in detector_ids]))

    vehicle_ids = set()
    for det in detector_ids:
        vehicle_ids.update(traci.lanearea.getLastStepVehicleIDs(det))
    wait_time = max((traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids), default=0.0)

    return queue, wait_time, occupancy, direction


def run() -> None:
    print("🚦 Starting Intelligent Traffic Light Control Simulation...\n")
    print(f"Using optimized fuzzy params: {OPTIMIZED_PARAMS}\n")
    save_membership_plots(OPTIMIZED_PARAMS)

    traci.start(["sumo-gui", "-c", "config.sumocfg"])
    fuzzy_system = build_fuzzy_control_system()
    next_green_duration = 20.0
    last_phase = -1
    decision_made = False

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_phase = traci.trafficlight.getPhase(TLS_ID)
        time_now = traci.simulation.getTime()
        time_left = traci.trafficlight.getNextSwitch(TLS_ID) - time_now

        if current_phase != last_phase:
            decision_made = False
            if current_phase in [EW_PHASE, NS_PHASE]:
                traci.trafficlight.setPhaseDuration(TLS_ID, next_green_duration)
                direction = "EW" if current_phase == EW_PHASE else "NS"
                print(f"[{time_now:.1f}s] {direction} GREEN started -> {next_green_duration:.1f}s")
            last_phase = current_phase

        if (not decision_made) and (time_left <= 2.0) and (current_phase in [EW_PHASE, NS_PHASE]):
            queue, wait_time, occupancy, next_direction = _get_opposite_metrics(current_phase)
            next_green_duration = evaluate_green_time(fuzzy_system, queue, wait_time, occupancy)
            next_green_duration = max(MIN_GREEN, min(MAX_GREEN, next_green_duration))
            print(
                f"[{time_now:.1f}s] {next_direction}: queue={queue:.0f}, wait={wait_time:.1f}s, "
                f"occ={occupancy:.1f}% -> green={next_green_duration:.1f}s"
            )
            decision_made = True

    traci.close()
    print("\nSimulation finished successfully.")


if __name__ == "__main__":
    run()
