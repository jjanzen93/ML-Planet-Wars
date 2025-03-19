#premades
def if_neutral_planet_available(state):
    return any(state.NeutralPlanets())


def have_largest_fleet(state): #not actually the right name
    for myplanet in state.EnemyPlanets():
        for eplanet in state.MyPlanets():
            if(eplanet.NumShips() < myplanet.NumShips()/2):
                return True
    return False

#my checks
def CloseOccupation(state):
    if(not state.EnemyPlanets()):
        return False
    divider = 2 #change if you want to send ships even though you don't have that many
    for myplanet in state.EnemyPlanets():
        for nplanet in state.NeutralPlanets():
            if(state.Distance(myplanet.PlanetID(), nplanet.PlanetID()) < 5 and myplanet.NumShips()/divider > nplanet.NumShips()):
                return True
    return False


def IsStealable(state):
    for efleet in state.MyFleets():
        if (state._planets[efleet.DestinationPlanet()] in state.NeutralPlanets() or state.EnemyPlanets()):
            return True
    return False

def IsDefendable(state):
    for efleet in state.MyFleets():
        if (state._planets[efleet.DestinationPlanet()] in state.EnemyPlanets()):
            return True
    return False