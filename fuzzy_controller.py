import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
def get_fuzzy_simulation():
    queue = ctrl.Antecedent(np.arange(0, 61, 1), 'queue')
    green_time = ctrl.Consequent(np.arange(10, 61, 1), 'green_time')
    queue['low']    = fuzzy.trimf(queue.universe, [0, 2, 5])
    queue['medium'] = fuzzy.trimf(queue.universe, [4, 6, 15])
    queue['high']   = fuzzy.trapmf(queue.universe, [12, 16, 30, 60])
    green_time['short']  = fuzzy.trimf(green_time.universe, [10, 18, 26])
    green_time['medium'] = fuzzy.trimf(green_time.universe, [24, 32, 42])
    green_time['long']   = fuzzy.trimf(green_time.universe, [40, 52, 60])
    rule1 = ctrl.Rule(queue['low'],    green_time['short'])
    rule2 = ctrl.Rule(queue['medium'], green_time['medium'])
    rule3 = ctrl.Rule(queue['high'],   green_time['long'])
    rule4 = ctrl.Rule(queue['low'] & queue['medium'], green_time['short'])
    rule5 = ctrl.Rule(queue['medium'] & queue['high'], green_time['medium'])
    system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(system)