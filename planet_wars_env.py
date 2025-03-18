import gym
from gym import spaces
import numpy as np
import random
from PlanetWars import PlanetWars
from PlanetWars import Fleet

class PlanetWarsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_turns=1000, opponent_model=None, visualize=False):
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
        self._load_map()

    def _load_map(self):
        map_index = random.randint(1, 100) #maybe change to a map generator function??
        with open(f"maps/map{map_index}.txt", "r") as f:
            map_data = f.read()
        self.pw = PlanetWars(map_data)
        self.num_planets = len(self.pw._planets)
        # observation: flatten each planet’s features: [x, y, owner, num_ships, growth_rate]
        self.observation_dim = self.num_planets * 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        # action: (source_planet, target_planet, fraction_index [0..10])
        self.action_space = spaces.MultiDiscrete([self.num_planets, self.num_planets, 11]) #chunked % of troops SHOULD BE IMPROVED
        self.current_turn = 0
        self.last_planets_owned = 0
        self.last_planets_owned_opponent = 0
        self.planets_lost = 0
        self.planets_opponent_lost = 0
        self.max_agent_planets = 0
        self.max_enemy_planets = 0
        self.last_score = self._compute_score()

    def reset(self):
        self._load_map()
        return self._get_obs()

    def _flip_obs(self, obs):
        # Reshape the flat observation to (num_planets, 5)
        reshaped = obs.reshape(-1, 5).copy()
        # Swap owner labels: 1 becomes 2 and 2 becomes 1.
        mask_agent = reshaped[:, 2] == 1.0
        mask_enemy = reshaped[:, 2] == 2.0
        reshaped[mask_agent, 2] = 2.0
        reshaped[mask_enemy, 2] = 1.0
        return reshaped.flatten()
    
    def step(self, action):
        # Process the agent’s action (player 1)
        valid_move = self._process_action(action, player=1)
        
        # Prepare observation for opponent: flip the owner labels.
        obs = self._get_obs()  # original observation from the agent's perspective
        if self.opponent_model is not None:
            try:
                self.opponent_model.do_turn(self.pw)
            except:
                flipped_obs = self._flip_obs(obs)
                opp_action, _ = self.opponent_model.predict(flipped_obs, deterministic=False)
                self._process_action(opp_action, player=2)
            #print(obs)
            #print(flipped_obs)
            #print("AGENT:", action)
            #print("OPP:", opp_action)
        # Simulate the turn: move fleets and resolve battles/growth
        self._simulate_turn()
        self.current_turn += 1

        current_score = self._compute_score()
        src, dst, frac = action
        fraction = frac / 10.0
        distance = self.pw.Distance(src, dst)
        if self.pw._planets[dst].NumShips() + self.pw._planets[dst].GrowthRate() * distance > self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() == 2:
            current_score -= 10
        elif self.pw._planets[dst].NumShips() + self.pw._planets[dst].GrowthRate() * distance < self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() == 2:
            current_score += 10
        if self.pw._planets[dst].NumShips() > self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() == 0:
            current_score -= 10
        elif self.pw._planets[dst].NumShips() < self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() == 0:
            current_score += 10
        if self.pw._planets[dst].GrowthRate() < 1 and self.pw._planets[dst].Owner != 1:
            current_score -= 10
        elif self.pw._planets[dst].GrowthRate() >= 1 and self.pw._planets[dst].Owner != 1:
            current_score += 10
        if valid_move:
            current_score += 10
        else:
            current_score -= 10
        reward = current_score - self.last_score
        self.last_score = current_score

        # Check for terminal condition (one side is eliminated or max_turns reached)
        done = self._check_done()

        # If game over, add additional terminal reward and record win/loss.
        if done:
            # Determine if agent and enemy are still alive.
            agent_alive = any(p.Owner() == 1 for p in self.pw._planets) or any(f._owner == 1 for f in self.pw._fleets)
            enemy_alive = any(p.Owner() == 2 for p in self.pw._planets) or any(f._owner == 2 for f in self.pw._fleets)
            # Initialize win flag as False.
            win = False
            if not agent_alive:
                reward += -100 + (50 * (100 / self.current_turn))# Agent lost
            elif not enemy_alive:
                reward += 100 + (50 * (100 / self.current_turn)) - 40 * (self.planets_lost / self.max_agent_planets) + 40 * (self.planets_opponent_lost / self.max_enemy_planets) # Enemy eliminated
                win = True
            elif self.current_turn >= self.max_turns:
                # Timeout: determine win by total ships.
                agent_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 1) + \
                                    sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1)
                enemy_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 2) + \
                                    sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2)
                if agent_total_ships > enemy_total_ships:
                    reward += 50 + 10 * (agent_total_ships / enemy_total_ships)  # Smaller bonus for timeout win.
                    win = True
                elif agent_total_ships < enemy_total_ships:
                    reward += -50 - 10 * (enemy_total_ships / agent_total_ships)
            self.episode_results.append(win)

        if self.visualize:
            self.render()
        
        return self._get_obs(), reward, done, {}

    def _process_action(self, action, player):
        # Action is a tuple: (source, target, fraction_index)
        src, dst, frac = action
        
        if src == dst:
            return False
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
        # fraction index from 0 to 10: 0 means no ships; 10 means 100%
        fraction = frac / 10.0
        num_ships = int(available * fraction)
        if num_ships < 1:
            return False
        # Remove ships from the source planet and create a new fleet.
        planet.RemoveShips(num_ships)
        trip_length = self.pw.Distance(src, dst)
        new_fleet = Fleet(player, num_ships, src, dst, trip_length, trip_length)
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

    def _compute_score(self):
        # Compute a score from the agent’s perspective:
        # Score = (# agent planets) + 0.1*(agent ships, including fleets) + 0.5*(agent growth)
        # minus the corresponding enemy totals.
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

        agent_total = len(agent_planets) + 0.1 * (agent_ships + agent_fleet_ships) + 0.5 * agent_growth
        enemy_total = len(enemy_planets) + 0.1 * (enemy_ships + enemy_fleet_ships) + 0.5 * enemy_growth
        return agent_total - enemy_total

    def _get_obs(self):
        # Return a flattened observation: for each planet [x, y, owner, num_ships, growth_rate].
        obs = []
        for planet in self.pw._planets:
            obs.extend([planet.X(), planet.Y(), float(planet.Owner()), float(planet.NumShips()), float(planet.GrowthRate())])
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
