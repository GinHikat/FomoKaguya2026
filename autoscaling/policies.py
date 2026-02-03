import numpy as np

class ScalingPolicy:
    def __init__(self, config):
        self.config = config
        self.min_servers = config["simulation"]["min_servers"]
        self.max_servers = config["simulation"]["max_servers"]
        self.cooldown = config["simulation"]["cooldown_minutes"]
        self.capacity = config["cost_parameters"]["capacity_per_server"]
        
    def decide(self, current_step, current_servers, current_load, predicted_load=None, cooldown_counter=0, **kwargs):
        """
        Decide the number of servers for the next step.
        
        Args:
            current_step (int): Current simulation step index/time.
            current_servers (int): Current number of servers.
            current_load (float): Current load (requests).
            predicted_load (float, optional): Forecasted load for next step(s).
            cooldown_counter (int): Steps remaining in cooldown.
            
        Returns:
            dict: {
                "action": int (-1, 0, 1 or more),
                "new_servers": int,
                "reason": str
            }
        """
        raise NotImplementedError

class ReactivePolicy(ScalingPolicy):
    def __init__(self, config):
        super().__init__(config)
        self.thresholds = config["thresholds"]["reactive"]

    def decide(self, current_step, current_servers, current_load, predicted_load=None, cooldown_counter=0, **kwargs):
        if cooldown_counter > 0:
             return {"action": 0, "new_servers": current_servers, "reason": "Cooldown"}

        # Calculate utilization
        # Avoid div/0
        effective_capacity = current_servers * self.capacity
        if effective_capacity == 0:
             util = 1.0 if current_load > 0 else 0.0
        else:
             util = current_load / effective_capacity

        action = 0
        reason = "Stable"
        
        if util > self.thresholds["high"]:
            if current_servers < self.max_servers:
                action = 1
                reason = "High Load"
        elif util < self.thresholds["low"]:
            if current_servers > self.min_servers:
                action = -1
                reason = "Low Load"
                
        new_servers = np.clip(current_servers + action, self.min_servers, self.max_servers)
        return {"action": new_servers - current_servers, "new_servers": new_servers, "reason": reason}

class PredictivePolicy(ScalingPolicy):
    def __init__(self, config):
        super().__init__(config)
        self.thresholds = config["thresholds"]["predictive"]
        # Predictive policy uses 'predicted_load' to verify thresholds.

    def decide(self, current_step, current_servers, current_load, predicted_load=None, cooldown_counter=0, **kwargs):
        if cooldown_counter > 0:
             return {"action": 0, "new_servers": current_servers, "reason": "Cooldown"}

        if predicted_load is None:
            # Fallback to reactive if no prediction
            predicted_load = current_load
            
        # Handle dict input (use mean)
        forecast = predicted_load
        if isinstance(predicted_load, dict):
            forecast = predicted_load.get("mean", current_load)
            
        # Utilization based on PREDICTED load
        effective_capacity = current_servers * self.capacity
        if effective_capacity == 0:
             pred_util = 1.0 if forecast > 0 else 0.0
        else:
             pred_util = forecast / effective_capacity

        action = 0
        reason = "Stable"

        if pred_util > self.thresholds["high"]:
             # Calculate needed servers to bring utilization down to target
             if current_servers < self.max_servers:
                action = 1
                reason = "High Predicted Load"
        elif pred_util < self.thresholds["low"]:
            if current_servers > self.min_servers:
                action = -1
                reason = "Low Predicted Load"

        new_servers = np.clip(current_servers + action, self.min_servers, self.max_servers)
        return {"action": new_servers - current_servers, "new_servers": new_servers, "reason": reason}

class HybridPolicy(ScalingPolicy):
    def __init__(self, config):
        super().__init__(config)
        self.thresholds = config["thresholds"]["hybrid"]

    def decide(self, current_step, current_servers, current_load, predicted_load=None, cooldown_counter=0, **kwargs):
        if cooldown_counter > 0:
             return {"action": 0, "new_servers": current_servers, "reason": "Cooldown"}

        current_util = current_load / (current_servers * self.capacity) if current_servers > 0 else (1.0 if current_load > 0 else 0.0)
        
        # 0. Anomaly Emergency Override
        is_anomaly = kwargs.get("is_anomaly", False)
        if is_anomaly:
             # Force max scale out or +2
             # If anomaly, we want to be safe. Let's try to reach max capacity faster.
             # Scale +2 if possible, or up to max.
             target = min(current_servers + 2, self.max_servers)
             if target > current_servers:
                 return {"action": target - current_servers, "new_servers": target, "reason": "Anomaly Detected! 🚨"}
                 
        # 1. Emergency Scale Out (Reactive)
        if current_util > self.thresholds["emergency"]:
            # Scale out aggressively (+2 if possible)
            add = 2 if (current_servers + 2) <= self.max_servers else (1 if (current_servers + 1) <= self.max_servers else 0)
            if add > 0:
                return {"action": add, "new_servers": current_servers + add, "reason": "Emergency Reactive"}

        # 2. Predictive Check
        # Handle predicted_load (float or dict with keys 'mean', 'upper', 'lower')
        forecast = predicted_load
        forecast_upper = predicted_load
        forecast_lower = predicted_load
        
        if isinstance(predicted_load, dict):
            forecast = predicted_load.get("mean", current_load)
            forecast_upper = predicted_load.get("upper", forecast)
            forecast_lower = predicted_load.get("lower", forecast)
        elif predicted_load is None:
            forecast = current_load
            forecast_upper = current_load
            forecast_lower = current_load
            
        # Use UPPER bound for scale-out check (Risk Averse: assume load might be higher)
        pred_util = forecast_upper / (current_servers * self.capacity) if current_servers > 0 else (1.0 if forecast_upper > 0 else 0.0)
        
        # Use LOWER bound for scale-in check (Risk Averse: ensure load is definitely low before scaling in)
        pred_util_low = forecast_lower / (current_servers * self.capacity) if current_servers > 0 else 0.0
        
        action = 0
        reason = "Stable"
        
        # Scale Out Decision (using Upper Bound)
        if pred_util > self.thresholds["out"]:
             if current_servers < self.max_servers:
                 action = 1
                 reason = "High Predicted Load (Upper Bound)"
                 
        # Scale In Decision (using Lower Bound)
        elif pred_util_low < self.thresholds["in"]:
             # Hysteresis Check: Do not scale in if we recently scaled out.
             # "LAST{Last Action was SCALE_OUT?} --> Yes --> NO_ACTION"
             # "LAST{Last Action was SCALE_OUT?} --> No --> SCALE_IN"
             last_action_type = kwargs.get("last_action_type", 0)
             
             if last_action_type == 1:
                 action = 0
                 reason = "Hysteresis Block"
             elif current_servers > self.min_servers:
                 action = -1
                 reason = "Low Predicted Load"
                 
        new_servers = np.clip(current_servers + action, self.min_servers, self.max_servers)
        return {"action": new_servers - current_servers, "new_servers": new_servers, "reason": reason}

class OptimalFixedPolicy(ScalingPolicy):
    """
    Fixed policy implementation.
    """
    def __init__(self, config, fixed_servers=None):
        super().__init__(config)
        self.fixed_servers = fixed_servers if fixed_servers else config["simulation"]["max_servers"] 

    def decide(self, current_step, current_servers, current_load, predicted_load=None, cooldown_counter=0, **kwargs):
        # Always try to be at fixed_servers
        diff = self.fixed_servers - current_servers
        return {"action": diff, "new_servers": self.fixed_servers, "reason": "Fixed Target"}

