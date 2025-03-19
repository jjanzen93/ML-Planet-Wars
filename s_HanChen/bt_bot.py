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
        planet_under_attack_check = Check(is_friendly_planet_under_attack)
        defend = Action(send_many_reinforcements_to_planets_under_attack)
        defensive_plan.child_nodes = [planet_under_attack_check, defend]
        
        # Hijack against enemys attacking neutral planets 
        hijack_plan = Sequence(name="Hijack Plan")
        neutral_planet_under_attack_check = Check(is_neutral_planet_under_attack)
        hijack_neutral = Action(send_reinforcements_to_neutral_planet_under_attack)
        hijack_plan.child_nodes = [neutral_planet_under_attack_check, hijack_neutral]

        offensive_plan = Sequence(name='Offensive Strategy')
        largest_fleet_check = Check(if_enemy_planet_snipable)
        attack = Action(like_agressive)
        offensive_plan.child_nodes = [largest_fleet_check, attack]

        spread_sequence = Sequence(name='Spread Strategy')
        neutral_planet_check = Check(if_neutral_planet_available)
        spread_action = Action(spread_many_to_closest_planet)
        spread_sequence.child_nodes = [neutral_planet_check, spread_action]
        
        conquest_sequence = Sequence(name='Conquest')
        fleet_check = Check(have_largest_fleet)
        conquer = Action(all_out_attack)
        conquest_sequence.child_nodes = [fleet_check, conquer]
        
        self.root.child_nodes = [offensive_plan, hijack_plan, defensive_plan, spread_sequence, conquest_sequence, attack.copy()]

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
