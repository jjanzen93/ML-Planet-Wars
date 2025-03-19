import gym
from gym import spaces
import numpy as np
import random
from PlanetWars import PlanetWars
from PlanetWars import Fleet
import heapq
import copy
import config

class PlanetWarsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_turns=1000, opponent_model=None, visualize=False):
        super(PlanetWarsEnv, self).__init__()
        #self.max_turns = max_turns
        self.max_turns = 150
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
        with open(f"amaps/map1.txt", "r") as f:
            map_data = f.read()
        self.pw = PlanetWars(map_data)
        self.num_planets = len(self.pw._planets)
        # observation: flatten each planet’s features: [x, y, owner, num_ships, growth_rate]
        self.observation_dim = 360 #self.num_planets * 5           now 80 for current planet, 20 for all 23 other planets, and 30 for extra game data 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        # action: (source_planet, target_planet, fraction_index [0..10])
        self.action_space = spaces.Discrete(21) #chunked % of troops SHOULD BE IMPROVED
        self.current_turn = 0
        self.last_planets_owned = 0
        self.last_planets_owned_opponent = 0
        self.planets_lost = 0
        self.planets_opponent_lost = 0
        self.max_agent_planets = 0
        self.max_enemy_planets = 0
        self.last_score = self._compute_score()

        self.opponents_state = copy.deepcopy(self.pw)
        self.passed = False
        self.distances = []
        for i in range(5):
            l = []
            for j in range(5):
                if i >= self.num_planets or j >= self.num_planets or i == j:
                    l.append(0)
                else:
                    l.append(self.pw.Distance(i, j))
            self.distances.append(l)

    def reset(self):
        self._load_map()
        return self._get_obs(self.pw, player = 1)
    
    def step(self, action):
        # Process the agent’s action (player 1)
        #print("STEPPING")
        if action == 20:
            self.passed = True
        else:
            src = action // 4
            dst = action % 4
            if dst >= src:
                dst += 1
            self._process_action(src, dst, player=1)
        
        done = False
        while (self.passed) and (not done):
            # q empty, no more moves to make.
            # make opponents moves
            """try:
                self.opponent_model.do_turn(self.pw)
                self._simulate_turn()
                self.current_turn += 1
                done = self._check_done()"""
            if True:
                self.passed = False
                while not self.passed:
                    opp_obs = self._get_obs(self.opponents_state, player = 2)
                    opp_action, _ = self.opponent_model.predict(opp_obs, deterministic=False, action_masks = self.mask)
                    if opp_action == 20:
                        self.passed = True
                    else:
                        opp_src = opp_action // 4
                        opp_dst = opp_action % 4
                        if opp_dst >= opp_src:
                            opp_dst += 1
                        self._process_action(4 - opp_src, 4 - opp_dst, player=2)
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
        
        

        reward = 0
        """if dst != 23 and fleet_size > 0:
            reward += 20 * (self.pw._planets[dst]._growth_rate / (self.pw._planets[dst].NumShips() + self.distances[src][dst]))
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
                reward -= 10"""

        if done:
                # Determine if agent and enemy are still alive.
                agent_alive = any(p.Owner() == 1 for p in self.pw._planets) or any(f._owner == 1 for f in self.pw._fleets)
                enemy_alive = any(p.Owner() == 2 for p in self.pw._planets) or any(f._owner == 2 for f in self.pw._fleets)
                # Initialize win flag as False.
                win = False
                if not agent_alive:
                    reward += -200
                    config.recent_wr.pop(0)
                    config.recent_wr.append(0)
                    print("Loss")
                elif not enemy_alive:
                    reward += 200
                    win = True
                    config.recent_wr.pop(0)
                    config.recent_wr.append(1)
                    print("Win")
                elif self.current_turn >= self.max_turns:
                    # Timeout: determine win by total ships.
                    agent_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 1) + \
                                        sum(f._num_ships for f in self.pw._fleets if f.Owner() == 1)
                    enemy_total_ships = sum(p.NumShips() for p in self.pw._planets if p.Owner() == 2) + \
                                        sum(f._num_ships for f in self.pw._fleets if f.Owner() == 2)
                    if agent_total_ships > enemy_total_ships:
                        reward += 10  # Smaller bonus for timeout win.
                        win = True
                        config.recent_wr.pop(0)
                        config.recent_wr.append(1)
                        print("Win")
                    elif agent_total_ships < enemy_total_ships:
                        reward += -200
                        config.recent_wr.pop(0)
                        config.recent_wr.append(0)
                        print("Loss")
                self.episode_results.append(win)
                #obs = self._get_obs()
                #mask = self.mask
                #info = {"action_mask": mask}
                return np.zeros(360), reward, done, {}
        obs = self._get_obs(self.pw, player = 1)
        mask = self.mask
        info = {"action_mask": mask}
        current_score = (obs[8] + obs[9])
        reward = current_score - self.last_score
        self.last_score = current_score
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

    def _process_action(self, src, dst, player=1):
        # Action is a tuple: (source, target, fraction_index)
        #src, dst, frac = action
        # For player 1 use owner==1, for opponent use owner==2.
        planet = self.pw._planets[src]

        # Remove ships from the source planet and create a new fleet.
        planet.RemoveShips(1)
        trip_length = self.pw.Distance(src, dst)
        new_fleet = Fleet(player, 1, src, dst, trip_length, trip_length)
        self.pw._fleets.append(new_fleet)
        if player == 2:
            o_planet = self.opponents_state._planets[src]
            o_planet.RemoveShips(1)
            self.opponents_state._fleets.append(new_fleet)
        return 1

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

        self.passed = True
        self.opponents_state = copy.deepcopy(self.pw)
        self.my_planets = self.pw.MyPlanets()
        for planet in self.my_planets:
            if planet._num_ships > 0:
                self.passed = False

    def _process_fleets(self, state, player=1):
        self.sorted_fleets = [[],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[]] # 23 empty lists
        self.inc_allies = [0, 0, 0, 0, 0]
        self.inc_enemies = [0, 0, 0, 0, 0]
        for fleet in state._fleets:
            heapq.heappush(self.sorted_fleets[fleet._destination_planet], (fleet._turns_remaining, (fleet._owner, fleet._num_ships, fleet._source_planet)))
            if fleet._owner == 1:
                self.inc_allies[fleet._destination_planet] += fleet._num_ships
            else:
                self.inc_enemies[fleet._destination_planet] += fleet._num_ships
        self.futures = [[],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[],[],[], [],[],[]] # 23 empty lists

        state._fleets = []
        for i in range(self.num_planets):
            planet = self.pw._planets[i]
            original_owner = planet._owner
            current_owner = original_owner
            growth = planet._growth_rate
            current_ships = planet._num_ships
            time = 0

            
            net_ships = 0
            while self.sorted_fleets[i]:
                delay, fleet = heapq.heappop(self.sorted_fleets[i])
                if time != delay:
                    if net_ships != 0:
                        trip_length = self.pw.Distance(source, i)
                        if net_ships > 0:
                            new_fleet = Fleet(1, net_ships, source, i, trip_length, time)
                        else:
                            new_fleet = Fleet(2, net_ships * -1, source, i, trip_length, time)
                        state._fleets.append(new_fleet)
                        net_ships = 0

                    self.futures[i].append([time, current_owner, current_ships])
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

                if fleet[0] == 1:
                    net_ships += fleet[1]
                    source = fleet[2]
                else:
                    net_ships -= fleet[1]
                    source = fleet[2]

            if net_ships != 0:
                trip_length = self.pw.Distance(source, i)
                if net_ships > 0:
                    new_fleet = Fleet(1, net_ships, source, i, trip_length, time)
                else:
                    new_fleet = Fleet(2, net_ships * -1, source, i, trip_length, time)
                state._fleets.append(new_fleet)

            self.futures[i].append([time, current_owner, current_ships])
        return self.futures


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

    def _get_obs(self, state, player=1):
        # Return a flattened observation: for each planet [x, y, owner, num_ships, growth_rate].
        planets = state._planets
        futures = self._process_fleets(state)

        if player == 1:
            is_player = True
            enemy = 2
        else:
            is_player = False
            enemy = 1

        obs = np.zeros(10)
        p1 = 0
        o1 = 0
        pg = 0
        og = 0
        for planet in state._planets:
            if planet._owner == player:
                p1 += planet._num_ships
                pg += planet._growth_rate
            if planet._owner == enemy:
                o1 += planet._num_ships
                og += planet._growth_rate
        p2 = 0
        o2 = 0
        for fleet in state._fleets:
            if fleet._owner == player:
                p2 += fleet._num_ships
            if fleet._owner == enemy:
                o2 += fleet._num_ships
        obs[0] = pg / 10
        obs[1] = og / 10
        obs[2] = (p1 + p2) / 300
        obs[3] = (o1 + o2) / 300
        obs[4] = p1 / 300
        obs[5] = o1 / 300
        obs[6] = p2 / 300
        obs[7] = o2 / 300
        obs[8] = pg - og
        obs[9] = (p1 + p2 - o1 - o2) / 10
        # first ten pieces of observation:
        # global state data, not specific to a planet
        # global player growth rate, global enemy growth rate, player total ships, enemy total ships,
        # player planet ships, enemy planet ships, player fleet ships, enemy fleet ships,
        # global growth rate advantage, global ship advantage

        # rest of data:
        # 70 pieces of data for all 5 planets
        # 0-2: one-hot owner
        # 3-5: one-hot final owner
        # 6-10: one-hot growth rate
        # 11-31: neutral ships (from t=0 to t=20)
        # 32-52: player ships (from t=0 to t=20)
        # 53: min ships
        # 54: max ships
        # 55: heuristic weakness score
        # 56: heuristic strength score
        # 57-58: one-hot [planet owner type] will be lost to enemy
        # 59-60: one-hot [planet owner type] will be obtained by player
        # 61: bool incoming player fleets
        # 62: bool incoming enemy fleets
        # 63: incoming player fleet count
        # 64: incoming enemy fleet count
        # 65: incoming fleet count advantage
        # 66-69: empty for now (should not impact performance)
        all_data = np.array([])
        for i in range(5):
            planet = state._planets[i]
            pfutures = futures[i]
            growth = planet._growth_rate
            data = np.zeros(70)

            # one-hot owner
            if planet._owner == 0:
                data[0] = 1
            elif planet._owner == player:
                data[1] = 1
            else:
                data[2] = 1
            # one-hot future owner
            if futures[i][-1][1] == 0:
                data[3] = 1
            elif futures[i][-1][1] == player:
                data[4] = 1
            else:
                data[5] = 1
            # one-hot growth rate
            if planet._growth_rate <= 1:
                data[6] = 1
            elif planet._growth_rate == 2:
                data[7] = 1
            elif planet._growth_rate == 3:
                data[8] = 1
            elif planet._growth_rate == 4:
                data[9] = 1
            else:
                data[10] = 1

            # ships from t=0 to t=20
            delay, current_owner, current_ships = pfutures[0]
            if len(pfutures) > 1:
                delay = pfutures[1][0]
            else:
                delay = 99
            future_ind = 0
            min_ships = current_ships
            max_ships = -999
            weakness_score = 999
            strength_score = -999
            original_owner = current_owner
            for t in range(21):
                # update info to next future
                if t == delay:
                    future_ind += 1
                    delay, current_owner, current_ships = pfutures[future_ind]
                    if len(pfutures) > future_ind + 1:
                        delay = pfutures[future_ind + 1][0]
                    else:
                        delay = 99

                # update data, calc growth
                if current_owner == 0:
                    data[t + 11] = current_ships / 100
                    min_ships = min(current_ships * -1, min_ships)
                    max_ships = max(current_ships * -1, max_ships)
                    if t >= 10:
                        weakness_score =  min(current_ships * -1, weakness_score)
                        strength_score =  max(current_ships * -1, strength_score)
                else:
                    if current_owner == player:
                        data[t + 32] = current_ships / 100
                        min_ships = min(current_ships, min_ships)
                        max_ships = max(current_ships, max_ships)
                        if t >= 10:
                            weakness_score =  min(current_ships, weakness_score)
                            strength_score =  max(current_ships, strength_score)
                        current_ships += growth

                    else:
                        data[t + 32] = current_ships / -100
                        min_ships = min(current_ships * -1, min_ships)
                        max_ships = max(current_ships * -1, max_ships)
                        if t >= 10:
                            weakness_score =  min(current_ships * -1, weakness_score)
                            strength_score =  max(current_ships * -1, strength_score)
                        current_ships += growth

            data[53] = min_ships / 100
            data[54] = max_ships / 100
            data[55] = 100 / (weakness_score - 0.5)
            data[56] = 100 / (strength_score - 0.5)
            if original_owner == 0:
                if pfutures[-1][1] == enemy:
                    data[57] = 1
                elif pfutures[-1][1] == player:
                    data[58] = 1
            elif original_owner == player:
                if pfutures[-1][1] == enemy:
                    data[59] = 1
            else:
                if pfutures[-1][1] == player:
                    data[60] = 1
            if player == 1:
                if self.inc_allies[i] > 0:
                    data[61] = 1
                    data[63] = self.inc_allies[i] / 20
                if self.inc_enemies[i] > 0:
                    data[62] = 1
                    data[64] = self.inc_enemies[i] / 20
                data[65] = (self.inc_allies[i] - self.inc_enemies[i]) / 10
            else:
                if self.inc_enemies[i] > 0:
                    data[61] = 1
                    data[63] = self.inc_enemies[i] / 20
                if self.inc_allies[i] > 0:
                    data[62] = 1
                    data[64] = self.inc_allies[i] / 20

                data[65] = (self.inc_enemies[i] - self.inc_allies[i]) / 10
                
            #print(data)
            if player == 1:
                all_data = np.concatenate((all_data, data))
            else:
                all_data = np.concatenate((data, all_data))
        obs = np.concatenate((obs, all_data))
        #print(obs)

        mask = np.ones(20)
        for i in range(5):
            planet = planets[i]
            if planet._owner != player or planet._num_ships == 0:
                mask[i*4] = 0
                mask[i*4 + 1] = 0
                mask[i*4 + 2] = 0
                mask[i*4 + 3] = 0
        if player == 2:
            mask = mask[::-1]
        mask = np.append(mask, 1)
        self.mask = mask

        return obs


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


    def action_masks(self, owner=1):
        return self.mask

if __name__ == "__main__":
    pass
