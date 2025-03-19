#!/usr/bin/env python

import logging, traceback, sys, os, inspect
logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from .checks import if_neutral_planet_available, is_under_attack, have_vulnerable_enemy_planet
from .behaviors import defend_weakest_planet, spread_to_strongest_neutral_planet, attack_enemy_weakpoint
from .bt_nodes import Selector, Sequence, Action, Check
import PlanetWars

class Bt_Bot():
    def __init__(self):
        self.root = Selector(name='High Level Strategy')

        aggressive_plan = Sequence(name='Aggressive Strategy')
        enemy_vulnerable = Check(have_vulnerable_enemy_planet)
        attack = Action(attack_enemy_weakpoint)
        aggressive_plan.child_nodes = [enemy_vulnerable, attack]

        production_plan = Sequence(name='Production Strategy')
        neutral_available = Check(if_neutral_planet_available)
        get_neutral = Action(spread_to_strongest_neutral_planet)
        production_plan.child_nodes = [neutral_available, get_neutral]

        defensive_plan = Sequence(name='Defensive Strategy')
        under_attack = Check(is_under_attack)
        defend = Action(defend_weakest_planet)
        defensive_plan.child_nodes = [under_attack, defend]

        self.root.child_nodes = [aggressive_plan, production_plan, defensive_plan]

    def do_turn(self, state):
        self.root.execute(state)

if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    behavior_tree = setup_behavior_tree()
    try:
        map_data = ''
        while True:
            current_line = input()
            if len(current_line) >= 2 and current_line.startswith("go"):
                planet_wars = PlanetWars(map_data)
                do_turn(planet_wars)
                finish_turn()
                map_data = ''
            else:
                map_data += current_line + '\n'

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error in bot.")
