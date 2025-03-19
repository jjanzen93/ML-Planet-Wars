import logging, traceback, sys, os, inspect
logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)


def if_been_attacked_without_reinforcement(state):
    enemy_targets = [fleet.DestinationPlanet() for fleet in state.MyFleets()]
    if not enemy_targets:
        return False

    def strength(p):
        return p.NumShips() \
               + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    for p in state.EnemyPlanets():
        if p.PlanetID() in enemy_targets:
            if strength(p) <= 0:
                return True

    return False


def if_possible_victory(state):
    attack_unit = max(state.EnemyPlanets(), key=lambda x: x.NumShips(), default=None)
    enemy_weakest = min(state.Planets(), key=lambda x: x.NumShips(), default=None)

    if attack_unit == None or enemy_weakest == None:
        return False
    
    return attack_unit.NumShips() > enemy_weakest.NumShips()


def if_neutral_planet_available(state):
    return any(state.NeutralPlanets())


def if_enemy_planet_available(state):
    return any(state.Planets())


def have_largest_fleet(state):
    return sum(planet.NumShips() for planet in state.EnemyPlanets()) \
             + sum(fleet.NumShips() for fleet in state.EnemyFleets()) \
           > sum(planet.NumShips() for planet in state.Planets()) \
             + sum(fleet.NumShips() for fleet in state.MyFleets())