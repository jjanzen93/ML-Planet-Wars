import sys
sys.path.insert(0, '../')
import PlanetWars

def defend_weakest_planet(state):
    my_planets = list(state.EnemyPlanets())
    if not my_planets:
        return False

    # Calculate total incoming threat to each planet
    def planet_threat(planet):
        incoming_friendly = sum(f.NumShips() for f in state.EnemyFleets()
                              if f.DestinationPlanet() == planet.PlanetID())
        incoming_enemy = sum(f.NumShips() for f in state.MyFleets()
                              if f.DestinationPlanet() == planet.PlanetID())
        # Consider growth over time until enemy arrives
        min_turns = min((f._turns_remaining for f in state.MyFleets()
                        if f.DestinationPlanet() == planet.PlanetID()), default=0)
        future_growth = planet.GrowthRate() * min_turns
        return planet.NumShips() + incoming_friendly + future_growth - incoming_enemy

    weakest = min(my_planets, key=planet_threat)
    strongest = max(my_planets, key=lambda p: p.NumShips())

    threat_level = planet_threat(weakest)
    if threat_level < 20:  # Only defend if actually threatened
        needed = abs(threat_level) + 15  # Dynamic defense size
        if strongest.NumShips() > needed * 1.5:  # Ensure we keep enough reserve
            return state.IssueOrder(strongest.PlanetID(), weakest.PlanetID(), needed)
    return False

def attack_enemy_weakpoint(state):
    my_planets = sorted(state.EnemyPlanets(), key=lambda p: p.NumShips(), reverse=True)
    if not my_planets:
        return False

    target_planets = [p for p in state.MyPlanets()
                     if not any(f.DestinationPlanet() == p.PlanetID() for f in state.EnemyFleets())]
    if not target_planets:
        return False

    # Sort targets by lowest (ships_needed / growth_rate) ratio
    def target_value(target):
        distance = state.Distance(my_planets[0].PlanetID(), target.PlanetID())
        ships_needed = target.NumShips() + distance * target.GrowthRate() + 1
        return ships_needed / (target.GrowthRate() + 1)  # +1 to avoid division by zero

    target_planets.sort(key=target_value)

    success = False
    for source in my_planets[:2]:  # Try to use top 2 strongest planets
        for target in target_planets[:2]:  # Consider top 2 most efficient targets
            distance = state.Distance(source.PlanetID(), target.PlanetID())
            needed = target.NumShips() + distance * target.GrowthRate() + 1

            if source.NumShips() > needed * 1.3:  # Only attack with 30% more ships than needed
                success = True
                return state.IssueOrder(source.PlanetID(), target.PlanetID(), needed)

    return success

def spread_to_strongest_neutral_planet(state):
    my_planets = sorted(state.EnemyPlanets(), key=lambda p: p.NumShips(), reverse=True)
    if not my_planets:
        return False

    neutral_planets = [p for p in state.NeutralPlanets()
                      if not any(f.DestinationPlanet() == p.PlanetID()
                               for f in state.EnemyFleets() + state.MyFleets())]
    if not neutral_planets:
        return False

    # Value = growth_rate / (ships_needed * distance)
    def planet_value(neutral):
        distance = min(state.Distance(p.PlanetID(), neutral.PlanetID()) for p in my_planets)
        return neutral.GrowthRate() / (neutral.NumShips() * distance + 1)

    neutral_planets.sort(key=planet_value, reverse=True)

    success = False
    for source in my_planets[:2]:  # Try using top 2 strongest planets
        for target in neutral_planets[:2]:  # Try capturing top 2 most valuable planets
            needed = target.NumShips() + 2
            if source.NumShips() > needed * 1.3:
                success = True
                state.IssueOrder(source.PlanetID(), target.PlanetID(), needed)

    return success


