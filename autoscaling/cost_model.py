import yaml
import numpy as np

class CostModel:
    def __init__(self, config_path="config.yaml"):
        # Load config if path provided, else expect config dict
        if isinstance(config_path, str):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path

        self.params = self.config["cost_parameters"]
        self.server_cost = self.params["server_cost"]
        self.scale_cost = self.params["scale_cost"]
        self.violation_cost = self.params["violation_cost"]
        self.capacity = self.params["capacity_per_server"]

    def calculate_step_cost(self, num_servers, previous_servers, load, active_servers=None):
        """
        Calculate cost for a single time step.
        
        Args:
            num_servers (int): Current number of provisioned servers.
            previous_servers (int): Number of servers in previous step.
            load (float): Current traffic load (requests).
            active_servers (int, optional): Effective active servers (ready to serve). 
                                            If None, assumes all provisioned are active (no boot time logic).
        
        Returns:
            dict: Cost components and total.
        """
        if active_servers is None:
            active_servers = num_servers

        # 1. Server Cost
        # Cost is based on provisioned servers (you pay even if booting)
        # Assuming cost is per server per minute
        s_cost = num_servers * self.server_cost

        # 2. Scale Cost
        # Incurred when number of servers changes
        rescaling_ops = abs(num_servers - previous_servers)
        sc_cost = rescaling_ops * self.scale_cost

        # 3. Violation Cost (SLA)
        # Based on detailed logic: dropped requests = max(0, load - capacity)
        total_capacity = active_servers * self.capacity
        dropped_requests = max(0, load - total_capacity)
        v_cost = dropped_requests * self.violation_cost
        
        total_cost = s_cost + sc_cost + v_cost

        return {
            "total": total_cost,
            "server_cost": s_cost,
            "scale_cost": sc_cost,
            "violation_cost": v_cost,
            "dropped_requests": dropped_requests,
            "utilization": load / total_capacity if total_capacity > 0 else (1.0 if load > 0 else 0.0)
        }

    def calculate_total_cost(self, timeline_results):
        """
        Aggregate costs over a stimulation timeline.
        
        Args:
            timeline_results (list of dict): List of step results.
            
        Returns:
            dict: Aggregated costs.
        """
        total = 0
        server = 0
        scale = 0
        violation = 0
        dropped = 0
        
        for step in timeline_results:
            costs = step.get("costs", {})
            total += costs.get("total", 0)
            server += costs.get("server_cost", 0)
            scale += costs.get("scale_cost", 0)
            violation += costs.get("violation_cost", 0)
            dropped += costs.get("dropped_requests", 0)
            
        return {
            "total_cost": total,
            "server_cost": server,
            "scale_cost": scale,
            "violation_cost": violation,
            "total_dropped_requests": dropped
        }
