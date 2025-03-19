import gym
from gym import spaces
import numpy as np
import random
from PlanetWars import PlanetWars
from PlanetWars import Fleet
import heapq
import copy

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
        self.observation_dim = 570 #self.num_planets * 5           now 80 for current planet, 20 for all 23 other planets, and 30 for extra game data 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        # action: (source_planet, target_planet, fraction_index [0..10])
        self.action_space = spaces.Discrete(24) #chunked % of troops SHOULD BE IMPROVED
        self.current_turn = 0
        self.last_planets_owned = 0
        self.last_planets_owned_opponent = 0
        self.planets_lost = 0
        self.planets_opponent_lost = 0
        self.max_agent_planets = 0
        self.max_enemy_planets = 0
        self.last_score = self._compute_score()

        self.opponents_state = copy.deepcopy(self.pw)
        #self.moves_remaining = 1
        self.pppq = []
        self.oppq = []
        heapq.heappush(self.pppq, (-100, 1))
        heapq.heappush(self.oppq, (-100, 2))
        self.distances = []
        for i in range(23):
            l = []
            for j in range(23):
                if i >= self.num_planets or j >= self.num_planets or i == j:
                    l.append(99999)
                else:
                    l.append(self.pw.Distance(i, j))
            self.distances.append(l)

    def reset(self):
        self._load_map()
        return self._get_obs()
    
    def step(self, dst):
        # Process the agent’s action (player 1)
        #print("STEPPING")
        #print(self.pppq)
        #print(self.oppq)
        src = self.src
        new_surplus, fleet_size = self._process_action(src, dst, self.surplus, player=1)
        # make process action return new surplus

        if new_surplus:
            heapq.heappush(self.pppq, (new_surplus * -1, src))
        done = False
        while (not self.pppq) and (not done):
            # q empty, no more moves to make.
            # make opponents moves
            if self.opponent_model != None:
                try:
                    self.opponent_model.do_turn(self.pw)
                    self._simulate_turn()
                    self.current_turn += 1
                    done = self._check_done()
                except:
                    while self.oppq:
                        opp_obs, opp_src, opp_surplus = self._get_opp_obs()
                        opp_action, _ = self.opponent_model.predict(opp_obs, deterministic=False)
                        #print(f"normal dst: {dst}\nopp action: {opp_action}")
                        new_surplus = self._opp_process_action(opp_src, opp_action, opp_surplus, player=2)
                        if new_surplus:
                            heapq.heappush(self.oppq, (opp_surplus * -1, opp_src))
                    self._simulate_turn()
                    self.current_turn += 1
                    done = self._check_done()
            else:
                self._simulate_turn()
                self.current_turn += 1
                done = self._check_done()
            

            # Check for terminal condition (one side is eliminated or max_turns reached)

            # If game over, add additional terminal reward and record win/loss.
            

            if self.visualize:
                self.render()

            # proceed to next turn
            # create new q
            # repeat until q not empty
        current_score = self._compute_score()
        """if self.pw._planets[dst].NumShips() > self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() != 1:
            current_score -= 10
        elif self.pw._planets[dst].NumShips() < self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() != 1:
            current_score += 10"""
        #if self.pw._planets[dst].GrowthRate() < 1 and self.pw._planets[dst].Owner != 1:
        #    current_score -= 10
        #elif self.pw._planets[dst].GrowthRate() > 1 and self.pw._planets[dst].Owner != 1:
        #    current_score += 10 * self.pw._planets[dst].GrowthRate()
        
        reward = current_score - self.last_score
        self.last_score = current_score

        reward = 0
        if dst != 23 and fleet_size > 0:
            reward += 20 * ((self.pw._planets[dst]._growth_rate / fleet_size) - self.distances[src][dst])
            if fleet_size > 0 and self.pw._planets[dst].Owner() == 0 and fleet_size > self.pw._planets[dst].NumShips():
                reward += 10
            elif fleet_size > 0 and self.pw._planets[dst].Owner() == 0 and fleet_size <= self.pw._planets[dst].NumShips():
                reward -= 10
            target_planet = self.pw._planets[dst]
            if len(self.pppq) > 1 and fleet_size > 0 and self.pw._planets[dst].Owner() == 2 and fleet_size + \
                sum(f.Owner() == 1 and f.DestinationPlanet() == dst for f in self.pw._fleets) > target_planet.NumShips() + target_planet.GrowthRate() * self.distances[src][dst]:
                reward += 10
            if fleet_size > 0 and fleet_size > self.pw._planets[dst].NumShips() / 2:
                reward += 10
            elif fleet_size > 0 and fleet_size < self.pw._planets[dst].NumShips() / 2:
                reward -= 10

        if done:
                # Determine if agent and enemy are still alive.
                agent_alive = any(p.Owner() == 1 for p in self.pw._planets) or any(f._owner == 1 for f in self.pw._fleets)
                enemy_alive = any(p.Owner() == 2 for p in self.pw._planets) or any(f._owner == 2 for f in self.pw._fleets)
                # Initialize win flag as False.
                win = False
                if not agent_alive:
                    reward += -100 + (100 * (self.current_turn / 100)) - 40 * (self.planets_lost / self.max_agent_planets) + 40 * (self.planets_opponent_lost / self.max_enemy_planets)# Agent lost
                elif not enemy_alive:
                    reward += 100 + (100 * (100 / self.current_turn)) - 40 * (self.planets_lost / self.max_agent_planets) + 40 * (self.planets_opponent_lost / self.max_enemy_planets)
                    win = True
                elif self.current_turn >= self.max_turns:
                    # Timeout: determine win by total ships.
                    agent_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 1) + \
                                        sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1)
                    enemy_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 2) + \
                                        sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2)
                    if agent_total_ships > enemy_total_ships:
                        reward += 50  # Smaller bonus for timeout win.
                        win = True
                    elif agent_total_ships < enemy_total_ships:
                        reward += -50
                self.episode_results.append(win)
                #obs = self._get_obs()
                #mask = self.mask
                #info = {"action_mask": mask}
                return np.zeros(570), reward, done, {}
        obs = self._get_obs()
        mask = self.mask
        info = {"action_mask": mask}
        return obs, reward, done, info





        
        """# Prepare observation for opponent: flip the owner labels.
        obs = self._get_obs()  # original observation from the agent's perspective
        if self.opponent_model is not None:
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
        if self.pw._planets[dst].NumShips() > self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() != 1:
            current_score -= 10
        elif self.pw._planets[dst].NumShips() < self.pw._planets[src].NumShips() * fraction and self.pw._planets[src].Owner() != 1:
            current_score += 10
        if self.pw._planets[dst].GrowthRate() < 1 and self.pw._planets[dst].Owner != 1:
            current_score -= 10
        elif self.pw._planets[dst].GrowthRate() > 1 and self.pw._planets[dst].Owner != 1:
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
        
        return self._get_obs(), reward, done, {}"""

    def _process_action(self, src, dst, surplus, player=1):
        # Action is a tuple: (source, target, fraction_index)
        #src, dst, frac = action
        if dst == 23:
            return 0, 0
        if src == dst:
            return 0, 0
        # For player 1 use owner==1, for opponent use owner==2.
        valid_owner = 1 if player == 1 else 2
        if src < 0 or src >= self.num_planets:
            return 0, 0
        planet = self.pw._planets[src]
        if planet.Owner() != valid_owner:
            return 0, 0
        available = surplus
        if available <= 0:
            return 0, 0
        # fraction index from 0 to 10: 0 means no ships; 10 means 100%
        num_ships = available

        #if self.p_needs[dst] > 0:
        #    num_ships = min(surplus, (self.p_needs[dst] + 1))

        #num_ships = int(available * fraction)
        if num_ships < 1 or num_ships > planet._num_ships:
            return 0, 0
        remaining = 0
        future = self.futures[dst][-1]
        if future[1] == 0:
            num_ships = future[2] + 1
            if surplus < num_ships:
                return 0, 0
            remaining = surplus - num_ships
        elif future[1] == 1:
            return 0, 0
        else:
            if self.distances[src][dst] >= future[0]:
                num_ships = min(surplus, future[2] + 1 + self.pw._planets[dst]._growth_rate * (self.distances[src][dst] - future[0]))
                remaining = surplus - num_ships

        # Remove ships from the source planet and create a new fleet.
        planet.RemoveShips(num_ships)
        trip_length = self.pw.Distance(src, dst)
        new_fleet = Fleet(player, num_ships, src, dst, trip_length, trip_length)
        self.pw._fleets.append(new_fleet)
        return remaining, num_ships

    def _opp_process_action(self, src, dst, surplus, player=2):
        # Action is a tuple: (source, target, fraction_index)
        #src, dst, frac = action
        if dst == 23:
            return 0
        if src == dst:
            return 0
        # For player 1 use owner==1, for opponent use owner==2.
        valid_owner = 1 if player == 1 else 2
        if src < 0 or src >= self.num_planets:
            return 0
        planet = self.pw._planets[src]
        o_planet = self.opponents_state._planets[src]
        if planet.Owner() != valid_owner:
            return 0
        available = surplus
        if available <= 0:
            return 0
        if available > planet._num_ships:
            return 0
        # fraction index from 0 to 10: 0 means no ships; 10 means 100%
        num_ships = available
        #num_ships = int(available * fraction)
        if num_ships < 1 or num_ships > planet._num_ships:
            return 0
        future = self.futures[dst][-1]
        if future[1] == 0:
            num_ships = future[2] + 1
            if surplus < num_ships:
                return 0
            remaining = surplus - num_ships
        elif future[1] == 2:
            return 0
        else:
            if self.distances[src][dst] >= future[0]:
                num_ships = min(surplus, future[2] + 1 + self.pw._planets[dst]._growth_rate * (self.distances[src][dst] - future[0]))
                remaining = surplus - num_ships
            else:
                remaining = 0

        # Remove ships from the source planet and create a new fleet.
        planet.RemoveShips(num_ships)
        o_planet.RemoveShips(num_ships)
        trip_length = self.pw.Distance(src, dst)
        new_fleet = Fleet(player, num_ships, src, dst, trip_length, trip_length)
        self.pw._fleets.append(new_fleet)
        self.opponents_state._fleets.append(new_fleet)
        return remaining

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

        self.opponents_state = copy.deepcopy(self.pw)
        self.my_planets = self.pw.MyPlanets()
        for planet in self.my_planets:
            heapq.heappush(self.pppq, (self.p_surpluses[planet.PlanetID()] * -1, planet.PlanetID()))
        self.enemy_planets = self.pw.EnemyPlanets()
        for planet in self.enemy_planets:
            heapq.heappush(self.oppq, (self.o_surpluses[planet.PlanetID()] * -1, planet.PlanetID()))

    def _process_fleets(self):
        self.sorted_fleets = [[],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[]] # 23 empty lists
        for fleet in self.pw._fleets:
            heapq.heappush(self.sorted_fleets[fleet._destination_planet], (fleet._turns_remaining, (fleet._owner, fleet._num_ships)))
        self.futures = [[],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[]] # 23 empty lists
        self.p_needs = []
        self.p_surpluses = []
        self.o_needs = []
        self.o_surpluses = []
        for i in range(self.num_planets):
            planet = self.pw._planets[i]
            original_owner = planet._owner
            current_owner = original_owner
            growth = planet._growth_rate
            current_ships = planet._num_ships
            if current_owner == 1:
                p_need = 0
                p_surplus = current_ships
                o_need = current_ships
                o_surplus = 0
            elif current_owner == 2:
                p_need = current_ships
                p_surplus = 0
                o_need = 0
                o_surplus = current_ships
            else:
                p_need = current_ships
                p_surplus = 0
                o_need = current_ships
                o_surplus = 0
            time = 0
            #self.futures[i].append(time, current_owner, current_ships)
            while self.sorted_fleets[i]:
                delay, fleet = heapq.heappop(self.sorted_fleets[i])
                if time != delay:
                    self.futures[i].append([time, current_owner, current_ships])
                    p_need, p_surplus, o_need, o_surplus = self._n_and_s(current_owner, current_ships, p_need, p_surplus, o_need, o_surplus)
                    if current_owner != 0:
                        current_ships += (delay - time) * growth
                    time = delay
                if current_owner == fleet[0]:
                    current_ships += fleet[1]
                else:
                    if current_ships < fleet[1]:
                        current_owner = fleet[0]
                        current_ships = fleet[1] - current_ships
                    else:
                        current_ships -= fleet[1]

            self.futures[i].append([time, current_owner, current_ships])
            p_need, p_surplus, o_need, o_surplus = self._n_and_s(current_owner, current_ships, p_need, p_surplus, o_need, o_surplus)
            self.p_needs.append(p_need)
            self.p_surpluses.append(p_surplus)
            self.o_needs.append(o_need)
            self.o_surpluses.append(o_surplus)


    def _n_and_s(self, current_owner, current_ships, p_need, p_surplus, o_need, o_surplus):
        if current_owner == 1:
            p_need = 0
            p_surplus = min(current_ships, p_surplus)
            o_need = current_ships
            o_surplus = 0
        elif current_owner == 2:
            p_need = current_ships
            p_surplus = 0
            o_need = 0
            o_surplus = min(current_ships, o_surplus)
        else:
            p_need = current_ships
            o_need = current_ships
        return p_need, p_surplus, o_need, o_surplus

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

    def _predict_attack_success(self, surplus, dist, dst, futures, enemy=2):

        future = futures[0]
        for i in range(len(futures)):
            if dist < futures[i][0]:
                break
            future = futures[i]

        if future[1] != enemy:
            return 0

        ships = future[2] + ((dist - future[0]) * dst._growth_rate)
        if surplus > ships:
            return 1
        return 0

    def _get_obs(self):
        # Return a flattened observation: for each planet [x, y, owner, num_ships, growth_rate].
        negative_surplus, self.src = heapq.heappop(self.pppq)
        self.surplus = negative_surplus * -1
        src_planet = self.pw._planets[self.src]

        self._process_fleets()
        obs = np.array(self.distances[self.src]) / 5
        growth = np.zeros(5)
        growth[max(0, src_planet._growth_rate - 1)] = 1
        features = np.array([self.p_needs[self.src], self.p_surpluses[self.src], src_planet._num_ships]) / 100
        filler = np.zeros(26)
        atk_pred = np.zeros(23)
        for i in range(self.num_planets):
            if self.src != i:
                atk_pred[i] = self._predict_attack_success(self.surplus, self.distances[self.src][i], self.pw._planets[i], self.futures[i])
        obs = np.concatenate((obs, growth, features, filler, atk_pred))

        #filler = np.zeros(490)
        #return np.concatenate((obs, filler))

        for i in range(self.num_planets):
            planet = self.pw._planets[i]
            growth = np.zeros(5)
            growth[max(0, planet._growth_rate - 1)] = 1
            owner = np.zeros(3)
            owner[planet._owner] = 1
            features = np.array([self.p_needs[i], self.p_surpluses[i], planet._num_ships]) / 100
            filler = np.zeros(9)
            obs = np.concatenate((obs, growth, owner, features, filler))
        if self.num_planets < 23:
            filler = np.zeros((23 - self.num_planets) * 20)
            obs = np.concatenate((obs, filler))
        filler = np.zeros(30)
        obs = np.concatenate((obs, filler))
        

        self.mask = np.ones(24)
        self.mask[self.src] = 0
        for i in range(self.num_planets):
            if self.futures[i][-1][1] == 1:
                self.mask[i] = 0
        return obs

    def _get_opp_obs(self):
        # Return a flattened observation: for each planet [x, y, owner, num_ships, growth_rate].
        negative_surplus, src = heapq.heappop(self.oppq)
        surplus = negative_surplus * -1
        src_planet = self.pw._planets[src]

        self._process_fleets()
        obs = np.array(self.distances[src]) / 5
        growth = np.zeros(5)
        growth[max(0, src_planet._growth_rate - 1)] = 1
        features = np.array([self.o_needs[src], self.o_surpluses[src], src_planet._num_ships]) / 100
        filler = np.zeros(26)
        atk_pred = np.zeros(23)
        for i in range(self.num_planets):
            if src != i:
                atk_pred[i] = self._predict_attack_success(surplus, self.distances[src][i], self.pw._planets[i], self.futures[i], enemy=1)
        obs = np.concatenate((obs, growth, features, filler, atk_pred))


        #filler = np.zeros(490)
        #return np.concatenate((obs, filler)), src, surplus

        for i in range(self.num_planets):
            planet = self.pw._planets[i]
            growth = np.zeros(5)
            growth[max(0, planet._growth_rate - 1)] = 1
            owner = np.zeros(3)
            owner[planet._owner] = 1
            features = np.array([self.o_needs[i], self.o_surpluses[i], planet._num_ships]) / 100
            filler = np.zeros(9)
            obs = np.concatenate((obs, growth, owner, features, filler))
        if self.num_planets < 23:
            filler = np.zeros((23 - self.num_planets) * 20)
            obs = np.concatenate((obs, filler))
        filler = np.zeros(30)
        obs = np.concatenate((obs, filler))
        

        self.mask = np.ones(24)
        self.mask[src] = 0
        for i in range(self.num_planets):
            if self.futures[i][-1][1] == 2:
                self.mask[i] = 0

        return obs, src, surplus


        """obs = []
        for planet in self.pw._planets:
            obs.extend([planet.X(), planet.Y(), float(planet.Owner()), float(planet.NumShips()), float(planet.GrowthRate())])
        return np.array(obs, dtype=np.float32)"""

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

    def get_action_mask(self, owner=1):
        return self.mask
        return np.ones(24)
        action_mask = np.array([], dtype=int)
        for i in range(23):
            if i < self.num_planets and self.pw._planets[i].Owner() == owner and self.pw._planets[i].NumShips() > 0:
                action_mask = np.concatenate((action_mask, np.ones(self.num_planets - 1, dtype=int)))
                action_mask = np.concatenate((action_mask, np.zeros(23 - self.num_planets, dtype=int)))
            else:
                action_mask = np.concatenate((action_mask, np.zeros(22, dtype=int)))
        action_mask = np.concatenate((action_mask, np.ones(1, dtype=int)))
        return action_mask

    def action_masks(self, owner=1):
        return self.mask
        return np.ones(24)
        action_mask = np.array([], dtype=int)
        for i in range(23):
            if i < self.num_planets and self.pw._planets[i].Owner() == owner and self.pw._planets[i].NumShips() > 0:
                action_mask = np.concatenate((action_mask, np.ones(self.num_planets - 1, dtype=int)))
                action_mask = np.concatenate((action_mask, np.zeros(23 - self.num_planets, dtype=int)))
            else:
                action_mask = np.concatenate((action_mask, np.zeros(22, dtype=int)))
        action_mask = np.concatenate((action_mask, np.ones(1, dtype=int)))
        return action_mask

if __name__ == "__main__":
    pass
