from typing import List

def reorder_robot_states(states: List[float], 
                          origin_order: List[str], 
                          target_order: List[str]) -> List[float]:
    """
    Reorder robot states based on origin and target leg orders.
    
    Args:
        states (List[float]): Input states to be reordered
        origin_order (List[str]): Original leg order
        target_order (List[str]): Desired leg order
    
    Returns:
        List[float]: Reordered states
    """
    # Validate input lengths (4 for feet states and 12 for joint states)
    if len(states) not in {4, 12}:
        raise ValueError(f"Expected 4 or 12 states, got {len(states)}")
    
    if len(origin_order) != 4 or len(target_order) != 4:
        raise ValueError("Both origin and target orders must be lists of 4 legs")
    
    num_legs = 4
    # Determine number of entries per leg
    entries_per_leg = len(states) // num_legs
    
    # Create reordering indices
    reorder_indices = []
    for target_leg in target_order:
        # Find the index of the target leg in the origin order
        origin_leg_index = origin_order.index(target_leg)
        
        # Calculate the start and end indices for this leg
        start = origin_leg_index * entries_per_leg
        end = start + entries_per_leg
        
        # Add indices for this leg
        reorder_indices.extend(range(start, end))
    
    # Reorder states
    return [states[idx] for idx in reorder_indices]