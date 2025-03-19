

import logging


def if_neutral_planet_available(state):
    return any(state.NeutralPlanets())
    # needs to check if neutral planets already has fleets on the way and is able to conquer planet

def if_enemy_planet_snipable(state):
  # check if my strongest planet can conquer the weakest enemy planet
  strongest_planet = max(state.EnemyPlanets(), key=lambda p: p.NumShips(), default=None)

  # Find the weakest enemy planet (one with the fewest ships)
  weakest_enemy_planet = min(state.MyPlanets(), key=lambda p: p.NumShips(), default=None)

  if not strongest_planet or not weakest_enemy_planet:
      # If there are no valid strongest or weakest planets, return False
      return False
  enemy_planet_reinforcements = sum(fleet.NumShips() for fleet in state.MyFleets() if fleet.DestinationPlanet() == weakest_enemy_planet.PlanetID())

  # Check if the strongest planet can conquer the weakest enemy planet
  return strongest_planet.NumShips() > (weakest_enemy_planet.NumShips() + enemy_planet_reinforcements)

def have_largest_fleet(state):
    #do not use this when neutral planets exist.
    if(any(state.NeutralPlanets())):
       return False
    return sum(planet.NumShips() for planet in state.EnemyPlanets()) \
             + sum(fleet.NumShips() for fleet in state.EnemyFleets()) \
           > sum(planet.NumShips() for planet in state.MyPlanets()) \
             + sum(fleet.NumShips() for fleet in state.MyFleets())

def is_friendly_planet_under_attack(state):
  # Checks if any friendly planet is being targeted by enemy fleets.
  planets_under_attack = [
      fleet.DestinationPlanet() for fleet in state.MyFleets()
      if fleet.DestinationPlanet() in [planet.PlanetID() for planet in state.EnemyPlanets()]  # Compare IDs
  ]
  
  if not planets_under_attack:
      return False
  else:
    return True

def is_neutral_planet_under_attack(state):
    planets_under_attack = [
        fleet.DestinationPlanet() for fleet in state.MyFleets()
        if state._planets[fleet.DestinationPlanet()].Owner() == 0  # Only consider neutral planets
    ]
    if not planets_under_attack:
      return False
    else:
      return True