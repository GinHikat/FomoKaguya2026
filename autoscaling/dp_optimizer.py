import numpy as np
from tqdm import tqdm
import itertools

class DPOptimizer:
    def __init__(self, config, cost_model, load_data):
        self.config = config
        self.cost_model = cost_model
        # Load data is list or array of requests count per interval
        self.load_data = transform_data_gap(load_data)
        self.T = len(self.load_data)
        
        self.min_servers = config["simulation"]["min_servers"]
        self.max_servers = config["simulation"]["max_servers"]
        self.cooldown = config["simulation"]["cooldown_minutes"]
        self.boot_time = config["cost_parameters"]["startup_time"]
        self.capacity = config["cost_parameters"]["capacity_per_server"]
        
        # --- State Space Definition ---
        # State: (ActiveServers, Pending_1, ..., Pending_K) where K = boot_time
        # P_i: Servers becoming active in i minutes
        # Queue length required: boot_time - 1
        # Example: boot_time=3. Queue=[P1, P2]. P1 active next step. P2 active in 2 steps.
        # Action determines what enters P_last.
        
        self.queue_len = max(0, self.boot_time - 1)
        
        print(f"Generating DP State Space (Queue Len={self.queue_len})...")
        
        # Generate valid states: Tuple (active, p1, p2...)
        # Constraint: active + sum(queue) <= max_servers
        # Optimization: We iterate total N from min to max, and partition N into slots.
        
        valid_states = []
        
        # Iterate total number of servers provisioned
        for n in range(self.min_servers, self.max_servers + 1):
            # Partition n into (Active, P1...PK)
            # number of slots = 1 + queue_len
            num_slots = 1 + self.queue_len
            
            # Use itertools to generate partitions
            # Approach: partitions of N into K bins.
            # Stars and Bars equivalent or just simple product since N is small (10).
            # Max servers=10, Slots=3. 
            # We can just iterate all tuples of length `num_slots` where sum is in range [min, max]
            # Actually range(0, max+1) for each slot, check sum.
            pass
            
        # Brute force generator for small state space
        # Max servers is small (typically < 20 for this demo)
        ranges = [range(self.max_servers + 1) for _ in range(1 + self.queue_len)]
        for p in itertools.product(*ranges):
            if self.min_servers <= sum(p) <= self.max_servers:
                valid_states.append(p)
                
        self.states = sorted(list(set(valid_states)))
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.S = len(self.states)
        print(f"DP State Space Size: {self.S}")
        
        self.C = self.cooldown + 1 
        self.P = 3 
        
        self.action_space = [-1, 0, 1]
        
        # Precompute Transitions
        self.transitions = np.full((self.S, len(self.action_space)), -1, dtype=int)
        
        print("Precomputing Transitions...")
        for i, s in enumerate(self.states):
            current_A = s[0]
            queue = list(s[1:])
            total = sum(s)
            
            for a_idx, action in enumerate(self.action_space):
                # 1. Bounds Check
                if total + action < self.min_servers or total + action > self.max_servers:
                    continue 

                # 2. Determine Next State (s')
                # Logic:
                # - Action modifies the "pipeline" input or removes from pipeline
                # - Time evolution shifts pipeline
                
                # Step A: Apply Action (Scaling)
                # This gives us the "Target Pipeline" before time shift
                
                temp_A = current_A
                temp_queue = queue.copy() # [p1, p2]
                
                newly_launched = 0 # Will enter at end of queue
                
                if action == 1:
                    newly_launched = 1
                elif action == -1:
                    # Remove 1 server. Prioritize removing from latest additions.
                    # Order of removal: newly_launched (implicit) -> queue end -> ... -> queue start -> active
                    
                    # Since we chose action=-1, we simply don't have "newly_launched" (it's 0)
                    # We start removing from existing queue
                    rem = 1
                    
                    # Check queue from end
                    for q_i in range(len(temp_queue)-1, -1, -1):
                        if rem == 0: break
                        if temp_queue[q_i] > 0:
                            temp_queue[q_i] -= 1
                            rem = 0
                            
                    # If still need remove, remove from Active
                    if rem > 0:
                        if temp_A > 0:
                            temp_A -= 1
                            
                # Step B: Time Evolution (Shift)
                # Next Active = temp_A + temp_queue[0]
                # Next Queue[i] = temp_queue[i+1]
                # Next Queue[last] = newly_launched
                
                next_A = temp_A
                if len(temp_queue) > 0:
                    next_A += temp_queue[0]
                    
                next_queue_list = []
                if len(temp_queue) > 1:
                    next_queue_list = temp_queue[1:]
                
                if self.boot_time > 1:
                    next_queue_list.append(newly_launched)
                else:
                    # If boot time is 1, newly launched lands explicitly in Active next step
                    next_A += newly_launched
                    
                next_state_tuple = tuple([next_A] + next_queue_list)
                
                if next_state_tuple in self.state_to_idx:
                    self.transitions[i, a_idx] = self.state_to_idx[next_state_tuple]
                else:
                    # Should correspond to a valid state if logic is correct
                    pass

    def prev_action_to_idx(self, p):
        return p + 1
        
    def idx_to_prev_action(self, idx):
        return idx - 1

    def optimize(self):
        V = np.full((self.T + 1, self.S, self.C, self.P), np.inf)
        Policy = np.full((self.T, self.S, self.C, self.P), 0, dtype=int)
        
        V[self.T, :, :, :] = 0
        
        for t in tqdm(range(self.T - 1, -1, -1), desc="DP Optimization"):
            load = self.load_data[t]
            
            for s_idx in range(self.S):
                state_tuple = self.states[s_idx]
                
                active_s = state_tuple[0]
                total_provisioned = sum(state_tuple)
                
                # Cost Calculation
                # Violation depends on ACTIVE servers
                drop = max(0, load - active_s * self.capacity)
                viol_cost = drop * self.config["cost_parameters"]["violation_cost"]
                
                # Server cost depends on PROVISIONED servers
                serv_cost = total_provisioned * self.config["cost_parameters"]["server_cost"]
                
                for c in range(self.C):
                    for p_idx in range(self.P):
                        prev_action = self.idx_to_prev_action(p_idx)
                        
                        best_cost = np.inf
                        best_a_idx = 1
                        
                        for a_idx, action in enumerate(self.action_space):
                            # Transition Check
                            s_next_idx = self.transitions[s_idx, a_idx]
                            if s_next_idx == -1: 
                                continue
                                
                            # Constraints
                            if c > 0 and action != 0: continue
                            if prev_action == 1 and action == -1: continue
                            
                            scal_cost = abs(action) * self.config["cost_parameters"]["scale_cost"]
                            immediate_cost = serv_cost + scal_cost + viol_cost
                            
                            # Next Cooldown
                            if action != 0:
                                c_next = self.config["simulation"]["cooldown_minutes"]
                            else:
                                c_next = max(0, c - 1)
                            p_next_idx = self.prev_action_to_idx(action)
                            
                            if t < self.T: # Boundary check not needed given loop range, but good practice
                                future_cost = V[t+1, s_next_idx, c_next, p_next_idx]
                                total_cost = immediate_cost + future_cost
                                
                                if total_cost < best_cost:
                                    best_cost = total_cost
                                    best_a_idx = a_idx
                                
                        V[t, s_idx, c, p_idx] = best_cost
                        Policy[t, s_idx, c, p_idx] = best_a_idx
                        
        self.V = V
        self.Policy = Policy
        
    def reconstruct_path(self):
        # Start: Min servers active, empty queue
        start_queue = [0] * self.queue_len
        start_tuple = tuple([self.min_servers] + start_queue)
        
        if start_tuple not in self.state_to_idx:
             # Fallback
             start_tuple = self.states[0]
             
        s_idx = self.state_to_idx[start_tuple]
        c_current = 0
        p_current_idx = self.prev_action_to_idx(0)
        
        timeline = []
        
        for t in range(self.T):
            if s_idx == -1: break
            
            a_idx = self.Policy[t, s_idx, c_current, p_current_idx]
            action = self.action_space[a_idx]
            
            s_tuple = self.states[s_idx]
            
            timeline.append({
                "time_step": t,
                "servers": sum(s_tuple), 
                "servers_active": s_tuple[0],
                "action": action,
                "load": self.load_data[t]
            })
            
            s_idx = self.transitions[s_idx, a_idx]
            
            if action != 0:
                c_current = self.config["simulation"]["cooldown_minutes"]
            else:
                c_current = max(0, c_current - 1)
            p_current_idx = self.prev_action_to_idx(action)
            
        return timeline

def transform_data_gap(data):
    return np.nan_to_num(data, nan=0.0)
