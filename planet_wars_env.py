import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from PlanetWars import PlanetWars
from PlanetWars import Fleet
from map_generator import generate_map

class PlanetWarsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_turns=1000, opponent_model=None, map_size=23, single_map=False, visualize=False):
        super(PlanetWarsEnv, self).__init__()
        self.max_turns = max_turns
        self.opponent_model = opponent_model  #starts out nulll then switches to copy
        self.visualize = visualize
        if self.visualize:
            from visualize import Visualizer
            self.visualizer = Visualizer()
        self.current_turn = 0
        self.last_score = 0.0
        self.episode_results = []
        self.last_planets_owned = 0
        self.last_planets_owned_opponent = 0
        self.planets_lost = 0
        self.planets_opponent_lost = 0
        self.max_agent_planets = 0
        self.max_enemy_planets = 0
        self.owned = [1]
        self.src = 1

        self.map_size = map_size
        self.single_map = single_map
        if single_map:
            self.map_data = generate_map(self.map_size)
        self._load_map()

    def _load_map(self):
        if not self.single_map:
            map_data = generate_map(self.map_size)
        else: map_data = self.map_data
        self.pw = PlanetWars(map_data)
        self.num_planets = len(self.pw._planets)
        self.observation_dim = self.num_planets*4 + (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([self.num_planets+1, 101])
        self.action_space = spaces.Discrete(self.num_planets+1)
        self.current_turn = 0
        self.last_planets_owned = 0
        self.last_planets_owned_opponent = 0
        self.planets_lost = 0
        self.planets_opponent_lost = 0
        self.max_agent_planets = 0
        self.max_enemy_planets = 0
        self.last_score = self._compute_score((0,0,0))

    def reset(self, seed=42):
        self._load_map()
        self.owned = [1]
        self.src = 1
        return self._get_obs(), seed
    
    def opp_move(self):
        
        if self.opponent_model is not None:
            try:
                self.opponent_model.do_turn(self.pw)
            except:
                #obs = self._get_obs()
                flipped_obs = self._flip_obs(obs)
                opp_action, _ = self.opponent_model.predict(flipped_obs, deterministic=False)
                self._process_action(opp_action, player=2)

    def step(self, action):
        #Hold all current planets in a queue. each step deque and make move from src. then check if queue is empty and make opponent move and requeue all planets owned.
        ships = 0

        #print("MOVE FOR ", self.src)
        dst = action
        #ships = int((pct/100)*self.pw._planets[self.src]._num_ships)
        valid = True
        if dst != self.map_size:
            ships, valid = self.move_heuristic(self.src, dst)
        
            self._process_action(self.src, dst, ships, player=1)
        
        if len(self.owned) == 0:
            self.opp_move()

            #only simulate at the end
            self._simulate_turn()
            self.current_turn += 1

            #re-queue
            self.owned = [p._planet_id for p in self.pw._planets if p.Owner() == 1]
        action = (self.src, dst, ships)
        current_score = self._compute_score(action)
        
        reward = current_score - self.last_score
        if not valid: reward -= 0
        self.last_score = current_score

        # Check for terminal condition (one side is eliminated or max_turns reached)
        if dst != self.map_size and ships > 0 and self.src != dst:
            reward += (20 * self.pw._planets[dst]._growth_rate / (0.66 * self.pw._planets[dst].NumShips() + 1)) * (10 / self.pw.Distance(self.src, dst))
            #print((20 * self.pw._planets[dst]._growth_rate / (0.66 * self.pw._planets[dst].NumShips()) + 1) * (10 / self.pw.Distance(self.src, dst)))
        done = self._check_done()

        # If game over, add additional terminal reward and record win/loss.
        if done:
            # Determine if agent and enemy are still alive.
            agent_alive = any(p.Owner() == 1 for p in self.pw._planets) or any(f._owner == 1 for f in self.pw._fleets)
            enemy_alive = any(p.Owner() == 2 for p in self.pw._planets) or any(f._owner == 2 for f in self.pw._fleets)
            # Initialize win flag as False.
            win = False
            if not agent_alive:
                reward -= 1000 + (1000 * (self.current_turn / 100)) - 400 * (self.planets_lost / self.max_agent_planets) + 400 * (self.planets_opponent_lost / self.max_enemy_planets)
            elif not enemy_alive:
                reward += 1000 + (1000 * (100 / self.current_turn)) - 400 * (self.planets_lost / self.max_agent_planets) + 400 * (self.planets_opponent_lost / self.max_enemy_planets) # Enemy eliminated
                win = True
            elif self.current_turn >= self.max_turns:
                # Timeout: determine win by total ships.
                agent_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 1) + sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1)
                enemy_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 2) + sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2)
                if agent_total_ships > enemy_total_ships:
                    reward += 500 + 100 * (agent_total_ships / enemy_total_ships)  # Smaller bonus for timeout win.
                    win = True
                elif agent_total_ships < enemy_total_ships:
                    reward += -500 + 100 * (enemy_total_ships / agent_total_ships)
            self.episode_results.append(win)

        if self.visualize:
            self.render()
        #print(self.src, action, reward)
        #print(valid, reward)
        return self._get_obs(), reward, done, False, {}

    def move_heuristic(self, src, dst):
        num_ships = 0
        src_planet = self.pw._planets[src]
        dst_planet = self.pw._planets[dst]
        if self.pw._planets[dst].Owner() == 2:
            delta = sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1 and f.DestinationPlanet()==dst and f._turns_remaining < self.pw.Distance(src, dst)) - sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2 and f.DestinationPlanet()==dst and f._turns_remaining < self.pw.Distance(src, dst))
            num_ships = dst_planet.NumShips() + dst_planet.GrowthRate() * self.pw.Distance(src, dst) - delta + 5

        elif dst_planet.Owner() == 0:
            delta = sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1 and f.DestinationPlanet()==dst and f._turns_remaining < self.pw.Distance(src, dst)) - sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2 and f.DestinationPlanet()==dst and f._turns_remaining < self.pw.Distance(src, dst))
            num_ships = dst_planet.NumShips() + 1 - delta
        else:
            num_ships = sum(f.NumShips() for f in self.pw.EnemyFleets() if f.DestinationPlanet() == dst) - (dst_planet.NumShips() + dst_planet.GrowthRate() * self.pw.Distance(src, dst) + 1)
        valid = True
        if num_ships > src_planet.NumShips():
            num_ships = 0
            valid = False
        if num_ships < 0:
            num_ships = 0
        return num_ships, valid

    def _process_action(self, src, dst, ships, player):
        if src == dst:
            return False
        if dst == self.num_planets:
            return True
        # For player 1 use owner==1, for opponent use owner==2.
        valid_owner = 1 if player == 1 else 2
        if src < 0 or src >= self.num_planets:
            return False
        planet = self.pw._planets[src]
        if planet.Owner() != valid_owner:
            return False
        available = planet.NumShips()
        if available <= 0:
            return False


        if ships < 1:
            return False
        # Remove ships from the source planet and create a new fleet.
        planet.RemoveShips(ships)
        trip_length = self.pw.Distance(src, dst)
        new_fleet = Fleet(player, ships, src, dst, trip_length, trip_length)
        self.pw._fleets.append(new_fleet)
        return True

    def _simulate_turn(self):
        # Update fleets: decrement turns and process arrivals.
        remaining_fleets = []
        for fleet in self.pw._fleets:
            fleet._turns_remaining -= 1
            if fleet._turns_remaining <= 0:
                planet = self.pw._planets[fleet._destination_planet]
                if planet.Owner() == fleet.Owner():
                    # Reinforce planet if same owner.
                    planet.AddShips(fleet._num_ships)
                else:
                    if fleet._num_ships > planet.NumShips():
                        # Capture the planet.
                        planet.NumShips(fleet._num_ships - planet.NumShips())
                        planet.Owner(fleet.Owner())
                    else:
                        planet.RemoveShips(fleet._num_ships)
            else:
                remaining_fleets.append(fleet)
        self.pw._fleets = remaining_fleets

        # All non-neutral planets receive growth.
        for planet in self.pw._planets:
            if planet.Owner() != 0:
                planet.AddShips(planet.GrowthRate())

    def _compute_score(self, action):
        # Compute a score from the agent’s perspective:
        # Score = (# agent planets) + 0.1*(agent ships, including fleets) + 0.5*(agent growth)
        # minus the corresponding enemy totals.

        src, dst, num_ships = action
        agent_planets = [p for p in self.pw._planets if p.Owner() == 1]
        enemy_planets = [p for p in self.pw._planets if p.Owner() == 2]
        agent_ships = sum(p.NumShips() for p in agent_planets)
        enemy_ships = sum(p.NumShips() for p in enemy_planets)
        agent_growth = sum(p.GrowthRate() for p in agent_planets)
        enemy_growth = sum(p.GrowthRate() for p in enemy_planets)
        agent_fleet_ships = sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1)
        enemy_fleet_ships = sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2)
        #Keeps track of the max number of planets acquired throughout the game, to make sure the bot doesn't just stay on one planet
        if self.max_agent_planets < len(agent_planets):
            self.max_agent_planets = len(agent_planets)
        #Keeps track of the max number of enemy planets throughout the game, to make sure that the bot doesn't evaluate based on taking enemy planets over and over again
        if self.max_enemy_planets < len(enemy_planets):
            self.max_enemy_planets = len(enemy_planets)
        #this is to keep track of how many planets the player loses over the course of the game
        if len(agent_planets) < self.last_planets_owned:
            self.planets_lost += self.last_planets_owned - len(agent_planets)
        #updates how many planets the player owns from last turn
        self.last_planets_owned = len(agent_planets)
        #this is to keep track of how many planets the enemy loses over the course of the game
        if len(enemy_planets) < self.last_planets_owned_opponent:
            self.planets_opponent_lost += self.last_planets_owned_opponent - len(enemy_planets)
        #updates how many planets the enemy owns from last turn
        self.last_planets_owned_opponent = len(enemy_planets)
        
        agent_total = len(agent_planets) + 0.3 * (agent_ships + agent_fleet_ships) + 0.5 * agent_growth
        enemy_total = len(enemy_planets) + 0.3 * (enemy_ships + enemy_fleet_ships) + 0.5 * enemy_growth
        return agent_total - enemy_total

    def _get_obs(self):
        if len(self.owned) == 0:
            self.src = 0
        else:
            self.src = self.owned.pop(0)
        obs = [self.src, self.pw._planets[self.src].NumShips(), self.pw._planets[self.src].Owner()] #
        #print("OBS FOR", self.src, self.owned)
        for planet in self.pw._planets: #ADD BINARY CAN CAP
            delta = sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1 and f.DestinationPlanet()==planet._planet_id) - sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2 and f.DestinationPlanet()==planet._planet_id)
            obs.extend([float(planet.Owner()), float(planet.NumShips()), float(planet.GrowthRate()), float(delta)]) #dist
        return np.array(obs, dtype=np.float32)

    def _check_done(self):
        # Terminal if the agent or enemy has no planets/fleets or if max_turns reached.
        agent_alive = any(p.Owner() == 1 for p in self.pw._planets) or any(f._owner == 1 for f in self.pw._fleets)
        enemy_alive = any(p.Owner() == 2 for p in self.pw._planets) or any(f._owner == 2 for f in self.pw._fleets)
        if not agent_alive or not enemy_alive or self.current_turn >= self.max_turns:
            return True
        return False

    def render(self, mode="human"):
        if self.visualize:
            self.visualizer.draw(self)
            self.visualizer.update()
        else:
            print(f"Turn: {self.current_turn}")
            for i, planet in enumerate(self.pw._planets):
                print(f"Planet {i}: Owner {planet.Owner()}, Ships {planet.NumShips()}, Growth {planet.GrowthRate()}")
            print("Fleets:")
            for fleet in self.pw._fleets:
                print(f"Fleet from {fleet._source_planet} to {fleet._destination_planet}, Owner {fleet._owner}, Ships {fleet._num_ships}, Turns remaining {fleet._turns_remaining}")
            print("-" * 40)

    def close(self):
        if self.visualize:
            self.visualizer.close()

if __name__ == "__main__":
    pass