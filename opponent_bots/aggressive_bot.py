#!/usr/bin/env python
#
import logging, traceback, sys, os, inspect
logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import PlanetWars

class Aggressive_Bot():
    def __init__(self):
        pass
    def spread(self, state):
        my_planets = iter(sorted(state.EnemyPlanets(), key=lambda p: p.NumShips()))

        neutral_planets = [planet for planet in state.NeutralPlanets()
                        if not any(fleet.DestinationPlanet() == planet.PlanetID() for fleet in state.EnemyFleets())]
        neutral_planets.sort(key=lambda p: p.NumShips())

        target_planets = iter(neutral_planets)

        try:
            my_planet = next(my_planets)
            target_planet = next(target_planets)
            while True:
                required_ships = target_planet.NumShips() + 1

                if my_planet.NumShips() > required_ships:
                    state.IssueOrder(my_planet.PlanetID(), target_planet.PlanetID(), required_ships)
                    my_planet = next(my_planets)
                    target_planet = next(target_planets)
                else:
                    my_planet = next(my_planets)

        except StopIteration:
            return


    def attack(self, state):
        my_planets = iter(sorted(state.EnemyPlanets(), key=lambda p: p.NumShips()))

        enemy_planets = [planet for planet in state.MyPlanets()
                        if not any(fleet.DestinationPlanet() == planet.PlanetID() for fleet in state.EnemyFleets())]
        enemy_planets.sort(key=lambda p: p.NumShips())

        target_planets = iter(enemy_planets)

        try:
            my_planet = next(my_planets)
            target_planet = next(target_planets)
            while True:
                required_ships = target_planet.NumShips() + \
                                    state.Distance(my_planet.PlanetID(), target_planet.PlanetID()) * target_planet.GrowthRate() + 1

                if my_planet.NumShips() > required_ships:
                    state.IssueOrder(my_planet.PlanetID(), target_planet.PlanetID(), required_ships)
                    my_planet = next(my_planets)
                    target_planet = next(target_planets)
                else:
                    my_planet = next(my_planets)

        except StopIteration:
            return


    def do_turn(self, state):
        self.attack(state)
        self.spread(state)

if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)

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
    except:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error in bot.")
        raise
