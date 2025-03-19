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

PlanetWars


# You have to improve this tree or create an entire new one that is capable
# of winning against all the 5 opponent bots
class Bt_Bot():
    def __init__(self):

        # Top-down construction of behavior tree
        self.root = Selector(name='High Level Ordering of Strategies')

        early_occupation = Sequence(name = 'Early Populating Strategy')
        worthwhile_neighbors = Check(CloseOccupation)
        occupy = Action(SendToBestNeutralPlanet)
        early_occupation.child_nodes = [worthwhile_neighbors, occupy]

        steal_time = Sequence(name = 'Time to Steal')
        is_stealable = Check(IsStealable)
        steal = Action(StealPlanet)
        steal_time.child_nodes = [is_stealable, steal]

        defensive_plan = Sequence(name = 'Defend My Planets')
        defendable = Check(IsDefendable)
        defense = Action(Reinforce)
        defensive_plan.child_nodes = [defendable, defense]

        #premade nodes, not very good AI
        offensive_plan = Sequence(name='Offensive Strategy')
        largest_fleet_check = Check(have_largest_fleet)
        attack = Action(attack_weakest_enemy_planet)
        offensive_plan.child_nodes = [largest_fleet_check, attack]

        spread_sequence = Sequence(name='Spread Strategy')
        neutral_planet_check = Check(if_neutral_planet_available)
        spread_action = Action(spread_to_weakest_neutral_planet)
        spread_sequence.child_nodes = [neutral_planet_check, spread_action]

        # root.child_nodes = [early_occupation, steal_time, offensive_plan, spread_sequence, attack.copy()]
        self.root.child_nodes = [early_occupation, steal_time, defensive_plan, offensive_plan, spread_sequence, attack.copy()]

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
