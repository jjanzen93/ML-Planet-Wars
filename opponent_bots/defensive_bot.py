#!/usr/bin/env python
#
import logging, traceback, sys, os, inspect
logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import PlanetWars

class Defensive_Bot:
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


    def defend(self, state):
        my_planets = [planet for planet in state.EnemyPlanets()]
        if not my_planets:
            return

        def strength(p):
            return p.NumShips() \
                + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
                - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

        avg = sum(strength(planet) for planet in my_planets) / len(my_planets)

        weak_planets = [planet for planet in my_planets if strength(planet) < avg]
        strong_planets = [planet for planet in my_planets if strength(planet) > avg]

        if (not weak_planets) or (not strong_planets):
            return

        weak_planets = iter(sorted(weak_planets, key=strength))
        strong_planets = iter(sorted(strong_planets, key=strength, reverse=True))

        try:
            weak_planet = next(weak_planets)
            strong_planet = next(strong_planets)
            while True:
                need = int(avg - strength(weak_planet))
                have = int(strength(strong_planet) - avg)

                if have >= need > 0:
                    state.IssueOrder(strong_planet.PlanetID(), weak_planet.PlanetID(), need)
                    weak_planet = next(weak_planets)
                elif have > 0:
                    state.IssueOrder(strong_planet.PlanetID(), weak_planet.PlanetID(), have)
                    strong_planet = next(strong_planets)
                else:
                    strong_planet = next(strong_planets)

        except StopIteration:
            return


    def do_turn(self, state):
        self.spread(state)
        self.defend(state)


if __name__ == '__main__':
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

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
