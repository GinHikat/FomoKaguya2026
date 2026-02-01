from tqdm import tqdm

class Simulator:
    def __init__(self, config, cost_model, policies):
        self.config = config
        self.cost_model = cost_model
        self.policies = policies # Dictionary of policy_name -> PolicyObject
        self.boot_time = config["cost_parameters"]["startup_time"]
        
    def run_scenario(self, policy_name, load_data, predictions=None):
        """
        Run simulation for a specific policy.
        """
        policy = self.policies[policy_name]
        results = []
        
        # Initial State
        current_servers = self.config["simulation"]["min_servers"] # Provisioned
        # State:
        #   ready_servers (int)
        #   booting_servers (list of remaining_time)
        
        ready_servers = current_servers
        booting_servers = [] # List of remaining minutes
        
        cooldown_counter = 0
        last_action_type = 0 # for hysteresis
        
        for t, load in enumerate(tqdm(load_data, desc=f"Simulating {policy_name}")):
            # 1. Update Booting Servers
            # Decrement timers
            new_booting = []
            for timer in booting_servers:
                if timer > 1:
                    new_booting.append(timer - 1)
                else:
                    ready_servers += 1
            booting_servers = new_booting
            
            total_provisioned = ready_servers + len(booting_servers)
            
            # 2. Get Decision
            # Policy applies to PROVISIONED servers.
            
            pred = predictions[t] if predictions is not None else None
            
            decision = policy.decide(
                current_step=t,
                current_servers=total_provisioned,
                current_load=load,
                predicted_load=pred,
                cooldown_counter=cooldown_counter,
                last_action_type=last_action_type
            )
            
            action = decision["action"]
            
            # 3. Apply Action
            # Scale Out
            if action > 0:
                # Add servers to booting
                for _ in range(action):
                    booting_servers.append(self.boot_time)
                cooldown_counter = self.config["simulation"]["cooldown_minutes"]
                last_action_type = 1
                
            # Scale In
            elif action < 0:
                # Remove servers
                # Logic: Remove booting first to save future capacity cost immediately.
                remove_count = abs(action)
                
                # Check feasibility (min bounds handled by policy)
                
                removed = 0
                while remove_count > 0 and removed < abs(action):
                    if len(booting_servers) > 0:
                        booting_servers.pop()
                    elif ready_servers > self.config["simulation"]["min_servers"]:
                        ready_servers -= 1
                    else:
                        break
                    remove_count -= 1
                    removed += 1
                    
                cooldown_counter = self.config["simulation"]["cooldown_minutes"]
                last_action_type = -1
                
            else:
                # No Op
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                # Resets cooldown if action was taken.
                pass # last_action_type remains
                
            # 4. Calculate Step Metrics
            # Cost includes ALL provisioned servers (Ready + Booting)
            # Violation uses ONLY Ready servers
            total_provisioned_now = ready_servers + len(booting_servers)
            
            step_cost = self.cost_model.calculate_step_cost(
                num_servers=total_provisioned_now,
                previous_servers=total_provisioned, # used for scale cost (delta)
                load=load,
                active_servers=ready_servers
            )
            
            results.append({
                "time_step": t,
                "load": load,
                "servers_provisioned": total_provisioned_now,
                "servers_active": ready_servers,
                "action": action,
                "costs": step_cost,
                "policy": policy_name
            })
            
        return results
