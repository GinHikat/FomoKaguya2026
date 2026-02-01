import numpy as np
from tqdm import tqdm
import math

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
        
        # State dimensions
        self.S = self.max_servers - self.min_servers + 1 # Server states indices (0 to S-1, mapping to min..max)
        self.C = self.cooldown + 1 # Cooldown states 0..cooldown (e.g. 0 to 5)
        self.P = 3 # Previous action: 0: -1(in), 1: 0(no-op), 2: +1(out)
        
        # Mapping helpers
        self.action_space = [-1, 0, 1]
        
    def server_to_idx(self, s):
        return s - self.min_servers
        
    def idx_to_server(self, idx):
        return idx + self.min_servers
        
    def prev_action_to_idx(self, p):
        # -1 -> 0, 0 -> 1, 1 -> 2
        return p + 1
        
    def idx_to_prev_action(self, idx):
        return idx - 1

    def optimize(self):
        """
        Run the DP algorithm (Backward Induction).
        V[t][s_idx][c][p_idx] = min total cost from t to T
        """
        # Initialize V table with infinity
        # Dimensions: T+1, S, C, P
        # Initialize V table with infinity
        # Dimensions: T+1, S, C, P
        
        V = np.full((self.T + 1, self.S, self.C, self.P), np.inf)
        # Policy table to store best action index
        # actions lie in 0..2 (indices)
        Policy = np.full((self.T, self.S, self.C, self.P), 0, dtype=int)
        
        # Terminal condition: 0 cost at T
        V[self.T, :, :, :] = 0
        
        # Backward Induction
        # Note on Lookahead for Boot Time:
        # We calculate violation cost against load[t + boot_time] to enforce proactive scaling.
        
        for t in tqdm(range(self.T - 1, -1, -1), desc="DP Optimization"):
            # Lookahead load for violation calculation
            t_lookahead = min(t + self.boot_time, self.T - 1)
            load_target = self.load_data[t_lookahead] 
            
            for s_idx in range(self.S):
                s = self.idx_to_server(s_idx)
                
                for c in range(self.C):
                    for p_idx in range(self.P):
                        prev_action = self.idx_to_prev_action(p_idx)
                        
                        best_cost = np.inf
                        best_a_idx = 1 # default 0 NO_OP
                        
                        # Try all actions
                        for a_idx, action in enumerate(self.action_space):
                            # Feasibility Checks
                            
                            # 1. Bounds
                            s_next = s + action
                            if s_next < self.min_servers or s_next > self.max_servers:
                                continue
                            
                            # 2. Cooldown
                            # If cooldown active (c > 0), only action 0 is allowed
                            if c > 0 and action != 0:
                                continue
                                
                            # 3. Hysteresis
                            # If prev_action was +1, do NOT allow -1
                            if prev_action == 1 and action == -1:
                                continue
                            
                            # Calculate Immediate Cost
                            # Server Cost + Scale Cost
                            # Using s_t for violations:
                            active_s = s
                            
                            # Cost Data
                            # We construct a synthetic cost for DP objective
                            drop = max(0, load_target - active_s * self.capacity)
                            viol_cost = drop * self.config["cost_parameters"]["violation_cost"]
                            serv_cost = active_s * self.config["cost_parameters"]["server_cost"]
                            scal_cost = abs(action) * self.config["cost_parameters"]["scale_cost"]
                            
                            immediate_cost = serv_cost + scal_cost + viol_cost
                            
                            # Next State
                            s_next_idx = self.server_to_idx(s_next)
                            
                            # Cooldown update
                            # If action != 0, reset cooldown to max
                            # If action == 0, decrement c (if c > 0) or c=0
                            if action != 0:
                                c_next = self.config["simulation"]["cooldown_minutes"]
                            else:
                                c_next = max(0, c - 1)
                                
                            # Previous Action update
                            p_next_idx = self.prev_action_to_idx(action)
                            
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
        """
        Forward pass to build the optimal trajectory.
        """
        # Initial State
        # Assume start with min servers? Or given?
        # Let's assume start at min servers, no cooldown, no prev action (0)
        s_current = self.min_servers
        c_current = 0
        p_current_idx = self.prev_action_to_idx(0)
        
        timeline = []
        
        for t in range(self.T):
            s_idx = self.server_to_idx(s_current)
            
            # Look up best action
            # Note: Policy stores indices into action_space
            a_idx = self.Policy[t, s_idx, c_current, p_current_idx]
            action = self.action_space[a_idx]
            
            # Record
            timeline.append({
                "time_step": t,
                "servers": s_current,
                "action": action,
                "load": self.load_data[t]
            })
            
            # Update state
            s_current += action
            if action != 0:
                c_current = self.config["simulation"]["cooldown_minutes"]
            else:
                c_current = max(0, c_current - 1)
            p_current_idx = self.prev_action_to_idx(action)
            
        return timeline

def transform_data_gap(data):
    # Ensure data is clean (fill nans or handle 0s if they are gaps)
    return np.nan_to_num(data, nan=0.0)
