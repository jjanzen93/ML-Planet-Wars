import sys
sys.path.insert(0, '../')
import PlanetWars

def attack_weakest_enemy_planet(state):
    strongest_planet = max(state.EnemyPlanets(), key=lambda t: t.NumShips(), default=None)
    weakest_planet = min(state.MyPlanets(), key=lambda t: t.NumShips(), default=None)

    if not strongest_planet or not weakest_planet:
        return False
    enemy_support = sum([fleet.NumShips() for fleet in state.MyFleets()
        if fleet.DestinationPlanet() == weakest_planet.PlanetID()
    ])

    min_amount = weakest_planet.NumShips() + enemy_support + (state.Distance(strongest_planet.PlanetID(), weakest_planet.PlanetID()) * weakest_planet.GrowthRate()) + 1
    
    # If already sniping don't snipe
    if any(weakest_planet.PlanetID() == fleet.DestinationPlanet() for fleet in state.EnemyFleets()):
        return False
    
    return state.IssueOrder(strongest_planet.PlanetID(), weakest_planet.PlanetID(), min_amount)

def like_agressive(state):

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
            if(state.Distance(my_planet.PlanetID(), target_planet.PlanetID()) > 30):
                return False
            if my_planet.NumShips() > required_ships:
                state.IssueOrder(my_planet.PlanetID(), target_planet.PlanetID(), required_ships)
                my_planet = next(my_planets)
                target_planet = next(target_planets)
            else:
                my_planet = next(my_planets)

    except StopIteration:
        return True

def spread_to_weakest_neutral_planet(state):
    # Find the weakest neutral planet that isn't already being conquered
    weakest_neutral = min(
        (
            neutral for neutral in state.NeutralPlanets()
            if sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == neutral.PlanetID()) <= neutral.NumShips()
        ),
        key=lambda neutral: neutral.NumShips(),  # Target the planet with the fewest ships
        default=None
    )
    
    if not weakest_neutral:
        return False

    strongest_planet = max(state.EnemyPlanets(), key=lambda p: p.NumShips(), default=None)

    if not strongest_planet or strongest_planet.NumShips() <= 1:
        return False

    result = state.IssueOrder(strongest_planet.PlanetID(), weakest_neutral.PlanetID(), weakest_neutral.NumShips() +  (state.Distance(strongest_planet.PlanetID(), weakest_neutral.PlanetID()) * weakest_neutral.GrowthRate()) + 1)
    return result


def spread_to_most_growth_neutral_planet(state):
    strongest_planet = max(state.EnemyPlanets(), key=lambda p: p.NumShips(), default=None)

    growth_planet = min(state.NeutralPlanets(), key=lambda p: p.GrowthRate(), default=None)

    if not strongest_planet or not growth_planet or strongest_planet.NumShips() < growth_planet.NumShips() + 1:
        return False
    else:
        return state.IssueOrder(strongest_planet.PlanetID(), growth_planet.PlanetID(), growth_planet.NumShips())


def spread_many_to_closest_planet(state):
    # Iterates through all my._planets() and spread to the closest values it can take. Similar to spread.bot, but focuses on Distance.
    strong_to_weak_planet = iter(sorted(state.EnemyPlanets(), key=lambda p: p.NumShips()))

    if not strong_to_weak_planet:
        return False
    try:
        curr_strong = next(strong_to_weak_planet)
    except StopIteration:
        return False
    
    neutral_planets = [planet for planet in state.NeutralPlanets()
                      if not any(fleet.DestinationPlanet() == planet.PlanetID() for fleet in state.EnemyFleets())]
    
    if not neutral_planets:
        return False
    
    neutral_planets.sort(key=lambda p: state.Distance(p.PlanetID(), curr_strong.PlanetID()))
    target = iter(neutral_planets)
    count = 0

    try:
        target_planet = next(target)
        while True:
            required_ships = target_planet.NumShips() + 1
            
            if curr_strong.NumShips() > required_ships:
                state.IssueOrder(curr_strong.PlanetID(), target_planet.PlanetID(), required_ships)
                curr_strong = next(strong_to_weak_planet)
                target_planet = next(target)
                count = 0
            elif required_ships > curr_strong.NumShips() and count < 3:
                #This is a general check of the other 3 closest nodes. 
                target_planet = next(target)    
                count += 1
            else:
                curr_strong = next(strong_to_weak_planet)
                count = 0            
                neutral_planets = [planet for planet in state.NeutralPlanets()
                                if not any(fleet.DestinationPlanet() == planet.PlanetID() for fleet in state.EnemyFleets())]
                
                if not neutral_planets:
                    return False
    
                neutral_planets.sort(key=lambda p: state.Distance(p.PlanetID(), curr_strong.PlanetID()))
                target = iter(neutral_planets)

    except StopIteration:
        return False
    


def spread_to_closest_netural_planet(state):
    # Find the closest neutral planet to any of my _planets that isn't already being conquered
    closest_neutral = min(
        (
            neutral for neutral in state.NeutralPlanets()
            if sum(fleet.NumShips() for fleet in state.EnemyFleets() if fleet.DestinationPlanet() == neutral.PlanetID()) <= neutral.NumShips()
        ),
        key=lambda neutral: min(state.Distance(neutral.PlanetID(), my_planet.PlanetID()) for my_planet in state.EnemyPlanets()),
        default=None
    )
    
    if not closest_neutral:
        return False

    # Find the strongest allied planet to neutral planet
    strongest_planet = max(state.EnemyPlanets(), key=lambda p: p.NumShips(), default=None)


    if not strongest_planet or strongest_planet.NumShips() <= 1:
        # No friendly planet with enough ships to send
        return False

    # Issue the order to send ships to the closest neutral planet
    result = state.IssueOrder(strongest_planet.PlanetID(), closest_neutral.PlanetID(), closest_neutral.NumShips() + 1)
    return result

def send_reinforcements_to_weakest_planet_under_attack(state):
    # Find all _planets under attack
    # *** fleet.DestinationPlanet() is the planet PlanetID() NOT the planet itself...
    planets_under_attack = [
        fleet.DestinationPlanet() for fleet in state.MyFleets()
        if state._planets[fleet.DestinationPlanet()].Owner() == 1  # Only consPlanetID()er your _planets
        and (
            # Case 1: No incoming friendly fleets
            sum(ally_fleet.NumShips() for ally_fleet in state.EnemyFleets()
                if ally_fleet.DestinationPlanet() == fleet.DestinationPlanet()) == 0
            or
            # Case 2: Incoming friendly fleets are insufficient
            sum(ally_fleet.NumShips() for ally_fleet in state.EnemyFleets()
                if ally_fleet.DestinationPlanet() == fleet.DestinationPlanet())
            < sum(enemy_fleet.NumShips() for enemy_fleet in state.MyFleets()
                if enemy_fleet.DestinationPlanet() == fleet.DestinationPlanet())
        )
    ]

    if not planets_under_attack:
        return False
    
    # Find my weakest_planet under attack
    weakest_planet = min(
        (planet for planet in state.EnemyPlanets() if planet.PlanetID() in planets_under_attack),
        key=lambda p: p.NumShips(),
        default=None
    )
    
    # Find all fleets that are attacking weakest_planet
    attacking_fleets = [fleet for fleet in state.MyFleets()
         if fleet.DestinationPlanet() in [planet.PlanetID() for planet in state.EnemyPlanets()]
    ]

    # Find the max fleet size
    min_req = sum([fleet.NumShips() for fleet in attacking_fleets]) + 1
    
    # Find the closest planet that can send out reinforcements
    closest_ally_planet = min(
        (planet for planet in state.EnemyPlanets() if planet.PlanetID() != weakest_planet.PlanetID() and planet.NumShips() > min_req * (state.Distance(planet.PlanetID(), weakest_planet.PlanetID()) * weakest_planet.GrowthRate())),
        key=lambda p: state.Distance(p.PlanetID(), weakest_planet.PlanetID()),
        default=None
    )

    # If cannot defend abandon the planet
    if not closest_ally_planet:
        return False
    else:
        min_req += (state.Distance(closest_ally_planet.PlanetID(), weakest_planet.PlanetID()) * weakest_planet.GrowthRate())
        return state.IssueOrder(closest_ally_planet.PlanetID(), weakest_planet.PlanetID(), min_req)

def send_many_reinforcements_to_planets_under_attack(state):
    #Defends against attacks using multiple _planets, makes sure it doesn't get captured.
    #is based on the code above, as it can defend but only with one planet.
    planets_under_attack = [
        state._planets[fleet.DestinationPlanet()] for fleet in state.MyFleets()
        if state._planets[fleet.DestinationPlanet()].Owner() == 1  # Only consPlanetID()er your _planets
        and (
            # Case 1: No incoming friendly fleets
            sum(ally_fleet.NumShips() for ally_fleet in state.EnemyFleets()
                if ally_fleet.DestinationPlanet() == fleet.DestinationPlanet()) == 0
            or
            # Case 2: Incoming friendly fleets are insufficient
            sum(ally_fleet.NumShips() for ally_fleet in state.EnemyFleets()
                if ally_fleet.DestinationPlanet() == fleet.DestinationPlanet())
            < sum(enemy_fleet.NumShips() for enemy_fleet in state.MyFleets()
                if enemy_fleet.DestinationPlanet() == fleet.DestinationPlanet())
        )
    ]

    if not planets_under_attack:
        return False
    
    planets_under_attack.sort(key=lambda p: p.NumShips())
    under_attack = iter(planets_under_attack)
    count = 0
    try:
        attacked = next(under_attack)
        while True:
            #how many many ships are being sent to planet under attack
            enemy_support = sum([fleet.NumShips() for fleet in state.MyFleets()
                if fleet.DestinationPlanet() == attacked.PlanetID()
            ])

            #how long it takes for the furthest enemy fleet to arrive. 
            #Using furtherest fleet allows for more fleets to respond. 
            enemy_time = max([fleet._total_trip_length for fleet in state.MyFleets()
                if fleet.DestinationPlanet() == attacked.PlanetID()
            ])
            
            #all ally _planets that can react before all enemy fleets can attack a planet
            closest_allies = [planet for planet in state.EnemyPlanets()
                if (state.Distance(attacked.PlanetID(), planet.PlanetID()) <= enemy_time)]  
            #cannot counter attack a planet that cannot be reacted to.
            if not closest_allies:
                return False
            
            closest_allies.sort(key=lambda p: p.NumShips(), reverse=True)
            
            #this check allows to see if a planet can be reasonably defended. If not, send all of it's ships to the strongest planet.
            check = sum(allies.NumShips() for allies in closest_allies) / 2
            if(check < enemy_support):
                state.IssueOrder(attacked.PlanetID(), closest_allies[0].PlanetID(), attacked.NumShips())
                return False
            
            ally_support = sum([fleet.NumShips() for fleet in state.EnemyFleets()
                if fleet.DestinationPlanet() == attacked.PlanetID()
            ])
            #have all allies send half of their ships to attacked planet until no longer needed. Will not return true, as this will repeat for other _planets
            #until iteration stops.
            for allies in closest_allies:
                count = ally_support + attacked.NumShips()
                if count > enemy_support:
                    return False
                elif enemy_support > count:
                    state.IssueOrder(allies.PlanetID(), attacked.PlanetID(), int(allies.NumShips() / 2))
            return False
            
    except StopIteration:
        return False
    


def send_reinforcements_to_neutral_planet_under_attack(state):
    # Find all neutral _planets that are about to be taken over by enemy fleets
    planets_under_attack = [
        fleet.DestinationPlanet() for fleet in state.MyFleets()
        if state._planets[fleet.DestinationPlanet()].Owner() == 0  # Neutral _planets
        and sum(f.NumShips() for f in state.MyFleets() if f.DestinationPlanet() == fleet.DestinationPlanet()) >= state._planets[fleet.DestinationPlanet()].NumShips()
    ]

    if not planets_under_attack:
        return False

    # Find the planet that has the most growth rate
    neutral_planet = max(
        (state._planets[planet_id] for planet_id in planets_under_attack),
        key=lambda p: p.GrowthRate(),
        default=None
    )

    # Calculate the time for the enemy fleet to arrive
    enemy_arrival_time = max(
        fleet._turns_remaining for fleet in state.MyFleets() if fleet.DestinationPlanet() == neutral_planet.PlanetID()
    )

    # Find the friendly _planets thats have enough ships and arrival time is > enemy_arrival_time
    strongest_planet = max(
        (planet for planet in state.EnemyPlanets() 
         if state.Distance(planet.PlanetID(), neutral_planet.PlanetID()) > enemy_arrival_time
         and planet.NumShips() > (state.Distance(planet.PlanetID(), neutral_planet.PlanetID()) - enemy_arrival_time) * neutral_planet.GrowthRate()),
        key=lambda p: p.NumShips(),
        default=None
    )

    if not strongest_planet:
        return False
    
    # Calculate Distance / turns it takes between my planet and neutral
    friendly_arrival_time = state.Distance(strongest_planet.PlanetID(), neutral_planet.PlanetID())

    ships_to_send = ((friendly_arrival_time - enemy_arrival_time) * neutral_planet.GrowthRate() + 2)

    if(friendly_arrival_time > 20):
        #tries not to send fleets too far away or too expensive.
        return False
    else:
        return state.IssueOrder(strongest_planet.PlanetID(), neutral_planet.PlanetID(), ships_to_send)


def all_out_attack(state):
    #sends 3 ships constantly from many different _planets to one until conquered, and moves to the next one.
    strong_to_weak_planet = iter(sorted(state.EnemyPlanets(), key=lambda p: p.NumShips(), reverse=True))

    if not strong_to_weak_planet:
        return False

    enemy_planets = [planet for planet in state.MyPlanets()]

    if not enemy_planets:
        return False

    enemy_planets.sort(key=lambda p: p.NumShips())
    target = iter(enemy_planets)
    
    try:
        curr_strong = next(strong_to_weak_planet)
        target_planet = next(target)
        while True:
            #though 3 is slower, it is less risky than any other number.
            required_ships = 3
            enemy_support = sum([fleet.NumShips() for fleet in state.MyFleets()
                if fleet.DestinationPlanet() == target_planet.PlanetID()
            ])
            ally_support = sum([fleet.NumShips() for fleet in state.EnemyFleets()
                if fleet.DestinationPlanet() == target_planet.PlanetID()
            ])
            planet_gone = target_planet.NumShips() + enemy_support - ally_support
            if(target_planet.NumShips() > 0 or planet_gone >= 0):
                state.IssueOrder(curr_strong.PlanetID(), target_planet.PlanetID(), required_ships)
                curr_strong = next(strong_to_weak_planet)
            else:
                target_planet = next(target)

    except StopIteration:
        return False