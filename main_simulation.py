import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set the environment variable 'SUMO_HOME'")

import numpy as np
import traci

from fuzzy_controller import OPTIMIZED_PARAMS, build_fuzzy_system, save_membership_plots

TLS_ID = "J1"
EW_PHASE = 0
NS_PHASE = 2
MIN_GREEN = 10.0
MAX_GREEN = 60.0


def _get_opposite_metrics(current_phase: int) -> tuple[float, float, float, str]:
    if current_phase == NS_PHASE:
        detector_ids = ["e2_0", "e2_1"]
        next_direction = "EW (East-West)"
    else:
        detector_ids = ["e2_2", "e2_3"]
        next_direction = "NS (North-South)"

    queue = sum(traci.lanearea.getJamLengthVehicle(det) for det in detector_ids)
    waiting_time = sum(traci.lanearea.getIntervalMeanTimeLoss(det) for det in detector_ids)
    occupancy = float(np.mean([traci.lanearea.getLastStepOccupancy(det) for det in detector_ids]))
    return queue, waiting_time, occupancy, next_direction


def run() -> None:
    print("🚦 Starting Intelligent Traffic Light Control Simulation...\n")
    print(f"Using optimized fuzzy params: {OPTIMIZED_PARAMS}\n")
    save_membership_plots(OPTIMIZED_PARAMS)

    traci.start(["sumo-gui", "-c", "config.sumocfg"])

    next_green_duration = 20.0
    last_phase = -1
    decision_made = False
    fuzzy_sim = build_fuzzy_system()

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
                print(f"[{time_now:.1f}s] {direction} GREEN started → {next_green_duration:.1f}s")
            last_phase = current_phase

        if (not decision_made) and (time_left <= 2.0) and (current_phase in [EW_PHASE, NS_PHASE]):
            try:
                queue_opposite, wait_opposite, occ_opposite, next_direction = _get_opposite_metrics(current_phase)

                fuzzy_sim.input["queue"] = queue_opposite
                fuzzy_sim.input["wait_time"] = wait_opposite
                fuzzy_sim.input["occupancy"] = occ_opposite
                fuzzy_sim.compute()

                new_duration = fuzzy_sim.output.get("green_time", 25.0)
                next_green_duration = max(MIN_GREEN, min(MAX_GREEN, new_duration))

                print(
                    f"[{time_now:.1f}s] Measuring {next_direction} | queue={queue_opposite:.0f}, "
                    f"wait={wait_opposite:.1f}s, occ={occ_opposite:.1f}% -> next_green={next_green_duration:.1f}s"
                )
            except Exception as exc:
                print(f"[{time_now:.1f}s] Fuzzy Error: {exc} → Using default 25s")
                next_green_duration = 25.0
            decision_made = True

    traci.close()
    print("\nSimulation finished successfully.")


if __name__ == "__main__":
    run()
