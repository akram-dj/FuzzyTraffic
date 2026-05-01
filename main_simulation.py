import os
import sys
import traci
from fuzzy_controller import get_fuzzy_simulation
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set the environment variable 'SUMO_HOME'")
def run():
    print("🚦 Starting Intelligent Traffic Light Control Simulation...\n")

    traci.start(["sumo-gui", "-c", "config.sumocfg"])

    next_green_duration = 20.0 
    last_phase = -1
    decision_made = False

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        current_phase = traci.trafficlight.getPhase("J1")
        time_now = traci.simulation.getTime()
        next_switch = traci.trafficlight.getNextSwitch("J1")
        time_left = next_switch - time_now

        if current_phase != last_phase:
            decision_made = False

            if current_phase == 2:      
                traci.trafficlight.setPhaseDuration("J1", next_green_duration)
                print(f"[{time_now:.1f}s] NS (North-South) GREEN started → {next_green_duration:.1f}s")

            elif current_phase == 0:    
                traci.trafficlight.setPhaseDuration("J1", next_green_duration)
                print(f"[{time_now:.1f}s] EW (East-West) GREEN started → {next_green_duration:.1f}s")

            last_phase = current_phase

        if (not decision_made) and (time_left <= 2.0) and (current_phase in [0, 2]):

            try:
                
                fuzzy_sim = get_fuzzy_simulation()

                if current_phase == 2:        
                    queue_opposite = (
                        traci.lanearea.getJamLengthVehicle("e2_0") +
                        traci.lanearea.getJamLengthVehicle("e2_1")
                    )
                    next_direction = "EW (East-West)"

                else:                         # EW green → measure opposite (NS)
                    queue_opposite = (
                        traci.lanearea.getJamLengthVehicle("e2_2") +
                        traci.lanearea.getJamLengthVehicle("e2_3")
                    )
                    next_direction = "NS (North-South)"

                # Run Fuzzy Controller
                fuzzy_sim.input['queue'] = queue_opposite
                fuzzy_sim.compute()

                new_duration = fuzzy_sim.output.get('green_time', 25.0)

                next_green_duration = max(10.0, min(60.0, new_duration))

                print(f"[{time_now:.1f}s] Measuring {next_direction} | Queued vehicles = {queue_opposite:.0f} "
                      f"→ Next green = {next_green_duration:.1f}s")

            except Exception as e:
                print(f"[{time_now:.1f}s] Fuzzy Error: {e} → Using default 25s")
                next_green_duration = 25.0

            decision_made = True

    traci.close()
    print("\nSimulation finished successfully.")


if __name__ == "__main__":
    run()