def if_neutral_planet_available(state):
    return any(p for p in state.NeutralPlanets()
              if not any(f.DestinationPlanet() == p.PlanetID() for f in state.EnemyFleets()))

def is_under_attack(state):
    return any(fleet.DestinationPlanet() == planet.PlanetID() 
              for planet in state.EnemyPlanets() 
              for fleet in state.MyFleets())

def have_vulnerable_enemy_planet(state):
    my_strongest = max(state.EnemyPlanets(), key=lambda p: p.NumShips(), default=None)
    if not my_strongest:
        return False

    enemy_planets = state.MyPlanets()
    
    for target in enemy_planets:
        distance = state.Distance(my_strongest.PlanetID(), target.PlanetID())
        required_ships = target.NumShips() + distance * target.GrowthRate() + 1
        if my_strongest.NumShips() > required_ships * 1.2:  # 20% safety margin
            return True
    return False

