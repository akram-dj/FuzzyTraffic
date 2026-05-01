import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set the environment variable 'SUMO_HOME'")

import numpy as np
import traci

from fuzzy_controller import BASELINE_PARAMS, build_fuzzy_control_system, evaluate_green_time

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


def _get_metrics(conn: traci.connection.Connection, detector_ids: list[str]) -> tuple[float, float, float]:
    queue = sum(conn.lanearea.getJamLengthVehicle(det_id) for det_id in detector_ids)
    occupancy = float(np.mean([conn.lanearea.getLastStepOccupancy(det_id) for det_id in detector_ids]))

    vehicle_ids = set()
    for det_id in detector_ids:
        vehicle_ids.update(conn.lanearea.getLastStepVehicleIDs(det_id))
    wait_time = max((conn.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids), default=0.0)
    return float(queue), float(wait_time), occupancy


def _step_controller(conn: traci.connection.Connection, fuzzy_system, state: dict, fuzzy_enabled: bool) -> None:
    current_phase = conn.trafficlight.getPhase(TLS_ID)
    now = conn.simulation.getTime()
    time_left = conn.trafficlight.getNextSwitch(TLS_ID) - now

    if current_phase != state["last_phase"]:
        state["decision_made"] = False
        if current_phase in (PHASE_EW_GREEN, PHASE_NS_GREEN) and fuzzy_enabled:
            conn.trafficlight.setPhaseDuration(TLS_ID, state["next_green_duration"])
        state["last_phase"] = current_phase

    if fuzzy_enabled and (not state["decision_made"]) and current_phase in (PHASE_EW_GREEN, PHASE_NS_GREEN) and time_left <= DECISION_HORIZON_SEC:
        detector_ids, _ = _get_opposite_detector_ids(current_phase)
        queue, wait_time, occupancy = _get_metrics(conn, detector_ids)
        state["next_green_duration"] = float(np.clip(evaluate_green_time(fuzzy_system, queue, wait_time, occupancy), MIN_GREEN, MAX_GREEN))
        state["decision_made"] = True


def _accumulate_waiting_seconds(conn: traci.connection.Connection) -> float:
    waiting = 0
    for veh_id in conn.vehicle.getIDList():
        if conn.vehicle.getSpeed(veh_id) < 0.1:
            waiting += 1
    return float(waiting)


def run_parallel_comparison() -> None:
    print("🚦 Starting parallel comparison: fuzzy vs standard controller")

    # identical scenario and demand for both instances
    common_args = ["-c", "config.sumocfg", "--start", "true", "--quit-on-end", "true"]
    traci.start(["sumo", *common_args], label="fuzzy")
    traci.start(["sumo", *common_args], label="standard")

    conn_fuzzy = traci.getConnection("fuzzy")
    conn_standard = traci.getConnection("standard")

    fuzzy_params = BASELINE_PARAMS
    fuzzy_system = build_fuzzy_control_system(fuzzy_params)

    fuzzy_state = {"next_green_duration": 20.0, "last_phase": -1, "decision_made": False}
    standard_state = {"next_green_duration": 20.0, "last_phase": -1, "decision_made": False}

    total_wait_fuzzy = 0.0
    total_wait_standard = 0.0

    while conn_fuzzy.simulation.getMinExpectedNumber() > 0 or conn_standard.simulation.getMinExpectedNumber() > 0:
        if conn_fuzzy.simulation.getMinExpectedNumber() > 0:
            conn_fuzzy.simulationStep()
            _step_controller(conn_fuzzy, fuzzy_system, fuzzy_state, fuzzy_enabled=True)
            total_wait_fuzzy += _accumulate_waiting_seconds(conn_fuzzy)

        if conn_standard.simulation.getMinExpectedNumber() > 0:
            conn_standard.simulationStep()
            _step_controller(conn_standard, None, standard_state, fuzzy_enabled=False)
            total_wait_standard += _accumulate_waiting_seconds(conn_standard)

    conn_fuzzy.close()
    conn_standard.close()

    print("\n=== Comparison Result ===")
    print(f"Fuzzy total waiting time (veh*s):    {total_wait_fuzzy:.1f}")
    print(f"Standard total waiting time (veh*s): {total_wait_standard:.1f}")
    if total_wait_fuzzy < total_wait_standard:
        print("Winner: Fuzzy controller (lower total waiting time).")
    elif total_wait_fuzzy > total_wait_standard:
        print("Winner: Standard controller (lower total waiting time).")
    else:
        print("Result: Tie.")


if __name__ == "__main__":
    run_parallel_comparison()
