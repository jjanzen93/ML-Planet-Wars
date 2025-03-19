#!/usr/bin/env python
#

"""
// There is already a basic strategy in place here. You can use it as a
// starting point, or you can throw it out entirely and replace it with your
// own.
"""
import logging, traceback, sys, os, inspect
logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from .behaviors import *
from .checks import *
from .bt_nodes import Selector, Sequence, Action, Check

import PlanetWars


# You have to improve this tree or create an entire new one that is capable
# of winning against all the 5 opponent bots
class Bt_Bot():
    def __init__(self):
        # Top-down construction of behavior tree
        self.root = Selector(name='High Level Ordering of Strategies')
        

        defensive_plan = Sequence(name='Defensive Strategy')
        defend_check = Check(if_been_attacked_without_reinforcement)
        defend_action = Action(defend_incoming_attacks)
        defensive_plan.child_nodes = [defend_check, defend_action]

        combat_selector = Selector(name='Spread and Offense Strategy')
        attack_sequence = Sequence(name='Attacking')
        attack_check = Check(if_possible_victory)
        attack_action = Action(attack_upgrade)
        attack_sequence.child_nodes = [attack_check, attack_action]

        weakening_sequence = Sequence(name="Weakening")
        weakening_check = Check(if_enemy_planet_available)
        weakening_action = Action(attack_weakest_enemy_planet_upgrade)
        weakening_sequence.child_nodes = [weakening_check, weakening_action]

        combat_selector.child_nodes = [attack_sequence, weakening_sequence]

        spread_plan = Sequence(name='Spread Strategy')
        neutral_planet_check = Check(if_neutral_planet_available)
        spread_action = Action(attack_cheapest_neutral)
        spread_plan.child_nodes = [neutral_planet_check, spread_action]
        
        self.root.child_nodes = [defensive_plan, combat_selector, spread_plan]

    # You don't need to change this function
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
