import sys
sys.path.insert(0, '../')

import logging, traceback, sys, os, inspect
logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)

import PlanetWars
from math import inf

def attack_weakest_enemy_planet(state):
    # (1) If we currently have a fleet in flight, abort plan.
    if len(state.EnemyFleets()) >= 1:
        return False

    # (2) Find my strongest planet.
    strongest_planet = max(state.EnemyPlanets(), key=lambda t: t.NumShips(), default=None)

    # (3) Find the weakest enemy planet.
    weakest_planet = min(state.MyPlanets(), key=lambda t: t.NumShips(), default=None)

    if not strongest_planet or not weakest_planet:
        # No legal source or destination
        return False
    else:
        # (4) Send half the ships from my strongest planet to the weakest enemy planet.
        return state.IssueOrder(strongest_planet.PlanetID(), weakest_planet.PlanetID(), strongest_planet.NumShips() / 2)


def attack_weakest_enemy_planet_upgrade(state):
    def strength(p):
        return p.NumShips() \
               + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    def enemy_strength(p):
        return p.NumShips() \
               - sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               + sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    # (1) If we currently have a fleet in flight, abort plan.
    if len(state.EnemyFleets()) >= 1:
        return False

    # (2) Find my strongest planet.
    my_planets = [planet for planet in state.EnemyPlanets() if strength(planet) > 0]
    strongest_planet = max(my_planets, key=lambda t: t.NumShips(), default=None)

    # (3) Find the weakest enemy planet.
    weakest_planet = min(state.MyPlanets(), key=enemy_strength, default=None)

    if not strongest_planet or not weakest_planet:
        return False
    
    ships_required = strongest_planet.NumShips() / 2
    return state.IssueOrder(strongest_planet.PlanetID(), weakest_planet.PlanetID(), int(ships_required))


def spread_to_weakest_neutral_planet(state):
    # (1) If we currently have a fleet in flight, just do nothing.
    if len(state.EnemyFleets()) >= 1:
        return False

    # (2) Find my strongest planet.
    strongest_planet = max(state.EnemyPlanets(), key=lambda p: p.NumShips(), default=None)

    # (3) Find the weakest neutral planet.
    weakest_planet = min(state.NeutralPlanets(), key=lambda p: p.NumShips(), default=None)

    if not strongest_planet or not weakest_planet:
        # No legal source or destination
        return False
    else:
        # (4) Send half the ships from my strongest planet to the weakest enemy planet.
        return state.IssueOrder(strongest_planet.PlanetID(), weakest_planet.PlanetID(), int(strongest_planet.NumShips() / 2))

def attack(state):
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


def attack_upgrade(state):
    def strength(p):
        return p.NumShips() \
               + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    enemy_planets = [planet for planet in state.MyPlanets()
                      if not any(fleet.DestinationPlanet() == planet.PlanetID() for fleet in state.EnemyFleets())]
    if not enemy_planets:
        return False

    target_planets = iter(sorted(enemy_planets, key=lambda p: p.NumShips()))
    target_planet = next(target_planets)

    my_planets = [planet for planet in state.EnemyPlanets() if strength(planet) > 0]
    if not my_planets:
        return False

    my_planets = iter(sorted(my_planets, key=lambda p: state.Distance(target_planet.PlanetID(), p.PlanetID())))
    my_planet = next(my_planets)

    try:
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
        return False


def weakening(state):
    def strength(p):
        return p.NumShips() \
               + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    def enemy_strength(p):
        return p.NumShips() \
               - sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               + sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    enemy_planets = [planet for planet in state.MyPlanets()
                      if not any(fleet.DestinationPlanet() == planet.PlanetID() for fleet in state.EnemyFleets())]
    if not enemy_planets:
        return False

    target_planets = iter(sorted(enemy_planets, key=enemy_strength, reverse=True))
    target_planet = next(target_planets)

    my_planets = [planet for planet in state.EnemyPlanets()
                      if not any(fleet.DestinationPlanet() == planet.PlanetID() for fleet in state.MyFleets())]
    if not my_planets:
        return False
    my_planets = iter(sorted(my_planets, key=lambda x: (strength(x) / (state.Distance(target_planet.PlanetID(), x.PlanetID())))))

    try:
        my_planet = next(my_planets)
        
        while True:
            required_ships = target_planet.NumShips() + \
                                 state.Distance(my_planet.PlanetID(), target_planet.PlanetID()) * target_planet.GrowthRate() + 1

            if my_planet.NumShips() > required_ships:
                return state.IssueOrder(my_planet.PlanetID(), target_planet.PlanetID(), required_ships)
            else:
                my_planet = next(my_planets)

    except StopIteration:
        return False

def spread(state):
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

def defend_incoming_attacks(state):
    def strength(p):
        return p.NumShips() \
               + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               + min(fleet._turns_remaining  for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID()) * p.GrowthRate()
                   
    def distance(p):
        return state.Distance(p.PlanetID(), protect_planet.PlanetID())

    enemy_targets = [fleet.DestinationPlanet() for fleet in state.MyFleets()]

    protect_planets = []
    for planet in state.EnemyPlanets():
        if planet.PlanetID() in enemy_targets:
            if strength(planet) <= 0:
                protect_planets.append(planet)

    if not protect_planets:
        return False

    protect_planet = min(protect_planets, key=strength)
    enemy_incoming = max((fleet for fleet in state.MyFleets() if fleet.DestinationPlanet() == protect_planet.PlanetID()), key=lambda x: x.NumShips())

    combat_units = state.EnemyPlanets()
    combat_units.remove(protect_planet)
    combat_units = iter(sorted(combat_units, key=distance))

    try:
        unit = next(combat_units)
        
        while True:
            ships_required = 1 - strength(protect_planet)
            if ships_required <= 0:
                return False

            if unit.NumShips() > ships_required:
                return state.IssueOrder(unit.PlanetID(), protect_planet.PlanetID(), ships_required)
            else:
                unit = next(combat_units)

    except StopIteration:
        return False


def attack_nearest_neutral(state):
    enemy_targets = [fleet.DestinationPlanet() for fleet in state.MyFleets()]

    def neutral_strength(p):
        return -p.NumShips() \
               + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    neutral_planets = [p for p in state.NeutralPlanets() if neutral_strength(p) <= 0]
    if not neutral_planets:
        return False

    neutral_planets = sorted(neutral_planets, key=neutral_strength)

    check = 0
    for my_planet in state.EnemyPlanets():

        if my_planet.PlanetID() in enemy_targets:
            continue

        distance = inf
        nearest_target = None
        for target in neutral_planets:
            target_distance = state.Distance(my_planet.PlanetID(), target.PlanetID())
            if target_distance < distance:
                shipWeNeed = target.NumShips() + 1
                if my_planet.NumShips() > shipWeNeed:
                    distance = target_distance
                    nearest_target = target

        if nearest_target:
            check += 1
            state.IssueOrder(my_planet.PlanetID(), nearest_target.PlanetID(), shipWeNeed)

    if check is not 0:
        return True
    else:
        return False


def attack_cheapest_neutral(state):

    def strength(p):
        return p.NumShips() \
               + sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               - sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())

    def neutral_strength(p):
        return p.NumShips() \
               - sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == p.PlanetID()) \
               + sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == p.PlanetID())
    
    neutral_planets = [p for p in state.NeutralPlanets() if neutral_strength(p) > 0]
    if not neutral_planets:
        return False

    enemy_targets = [fleet.DestinationPlanet() for fleet in state.MyFleets()]
    combat_units = [p for p in state.EnemyPlanets() if p.PlanetID() not in enemy_targets]
    if not combat_units:
        return False

    combat_unit = max(combat_units, key=strength)

    target_planet = min(neutral_planets, key=lambda x: (neutral_strength(x) * state.Distance(combat_unit.PlanetID(), x.PlanetID())))

    ships_required = neutral_strength(target_planet) + 1

    if combat_unit.NumShips() > ships_required:
        return state.IssueOrder(combat_unit.PlanetID(), target_planet.PlanetID(), ships_required)
    else:
        return False
