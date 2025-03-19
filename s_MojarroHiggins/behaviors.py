import sys
sys.path.insert(0, '../')
import PlanetWars
#my actions
def heuristic(state, myplanet): #returns the best planet to populate and the heuristic value from one of my planets
    dweight = 2
    pweight = 1
    gweight = 10

    bestpc = 100000
    bestp = myplanet

    for nplanet in state.NeutralPlanets():
        dist = dweight * state.Distance(myplanet.PlanetID(), nplanet.PlanetID())
        pop = pweight * nplanet.NumShips()
        grow = gweight * nplanet.GrowthRate()
        total = dist + pop - grow
        if(total < bestpc):
            bestp = nplanet
            bestpc = total
    return bestp, bestpc

def SendToBestNeutralPlanet(state):
    # Use a generator to find the best neutral planet and the corresponding myplanet
    result = min(
        ((myplanet, *heuristic(state, myplanet)) for myplanet in state.EnemyPlanets()), 
        key=lambda x: x[2],  # Compare based on the heuristic value (bestpc, which is the third element in the tuple)
        default=(None, None, float('inf'))  # Handle cases where there are no planets
    )
    myplanet, best_planet, bestpc = result  # Unpack the result
    
    if best_planet:
        # Ensure the source planet can send enough ships
        if (state.Distance(myplanet.PlanetID(), best_planet.PlanetID()) + best_planet.NumShips() <= myplanet.NumShips() and not any(fleet.DestinationPlanet() == best_planet.PlanetID() for fleet in state.EnemyFleets())):
            return state.IssueOrder(myplanet.PlanetID(), best_planet.PlanetID(), best_planet.NumShips() + 1)
    return False

def StealPlanet(state):
    for efleet in state.MyFleets():
        if(state._planets[efleet.DestinationPlanet()] in state.NeutralPlanets() or state.EnemyPlanets()):
            for mplanet in state.EnemyPlanets():
                total = efleet.NumShips() - state._planets[efleet.DestinationPlanet()].NumShips() + (state._planets[efleet.DestinationPlanet()].GrowthRate() * (state.Distance(mplanet.PlanetID(), efleet.DestinationPlanet()) - efleet._turns_remaining)) + 5 #how many ships there will be by the time our fleet arrives + 5
                isfleet = any(fleet.DestinationPlanet() == efleet.DestinationPlanet() for fleet in state.EnemyFleets()) #is there already a fleet going there
                if(efleet._turns_remaining < state.Distance(mplanet.PlanetID(), efleet.DestinationPlanet()) and total < mplanet.NumShips()) and total > 0 and not isfleet:
                    return state.IssueOrder(mplanet.PlanetID(), efleet.DestinationPlanet(), total)
    return False

            
def Reinforce(state):
    for efleet in state.MyFleets():
        if(state._planets[efleet.DestinationPlanet()] in state.EnemyPlanets()):
            for mplanet in state.EnemyPlanets():
                total = efleet.NumShips() - (state._planets[efleet.DestinationPlanet()].NumShips() + state._planets[efleet.DestinationPlanet()].GrowthRate() * efleet._turns_remaining) + 5 #how many ships there will be by the time our fleet arrives + 5
                isfleet = any(fleet.DestinationPlanet() == efleet.DestinationPlanet() for fleet in state.EnemyFleets()) #is there already a fleet going there
                if(efleet._turns_remaining > state.Distance(mplanet.PlanetID(), efleet.DestinationPlanet()) and total < mplanet.NumShips()) and total > 0 and not isfleet:
                    return state.IssueOrder(mplanet.PlanetID(), efleet.DestinationPlanet(), total)
    return False



def attack_value(state, myplanet, eplanet):
    return myplanet.NumShips() - (eplanet.NumShips() + eplanet.GrowthRate() * state.Distance(myplanet.PlanetID(), eplanet.PlanetID()))

#premades
def attack_weakest_enemy_planet(state):
    myp = None
    ep = None
    bestvalue = 0
    for myplanet in state.EnemyPlanets():
        for eplanet in state.MyPlanets():
            if(attack_value(state, myplanet, eplanet) > bestvalue):
                myp = myplanet
                ep = eplanet
                bestvalue = attack_value(state, myplanet, eplanet)
    if(myp == None or ep == None):
        return False
    total = ep.NumShips() + (ep.GrowthRate() * state.Distance(myp.PlanetID(), ep.PlanetID())) + 5
    if(myp.NumShips() > total and total > 0):
        return state.IssueOrder(myp.PlanetID(), ep.PlanetID(), total)
    return False


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
