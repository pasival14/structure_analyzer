import numpy as np
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Node:
    """Represents a node (joint) in the structure."""
    name: str
    x: float
    y: float
    is_support: bool = False
    support_type: str = None  # 'fixed', 'pinned', 'roller'

@dataclass
class Member:
    """Represents a member in the structure."""
    name: str
    start_node: Node
    end_node: Node
    EI: float  # Flexural rigidity (E*I)
    length: float = None
    
    def __post_init__(self):
        if self.length is None:
            dx = self.end_node.x - self.start_node.x
            dy = self.end_node.y - self.start_node.y
            self.length = np.sqrt(dx**2 + dy**2)

@dataclass
class Load:
    """Represents a load on a member."""
    member: Member
    load_type: str  # 'point', 'uniform', 'moment'
    magnitude: float
    location: float = None  # For point loads and moments, distance from start node
    
class StructureAnalyzer:

    """
        Initialize the analyzer with a specific sway case.
        
        Sway Cases:
        1: Basic sway with horizontal loading
        2: Sway with distributed loads at top
        3: Sway with axial load P at top
        4: No sway, axial load at top
        5: Sway with axial load and distributed load at top
        """
    
    def __init__(self, sway_case=1):
        self.nodes = {}  # Changed from list to dict
        self.members = {}  # Changed from list to dict
        self.loads = []
        self.unknowns = {}
        self.equations = []
        self.results = {}
        self.sway_case = sway_case
        self.horizontal_loads = [] 
        self.support_reactions = defaultdict(float)
    
    def add_node(self, name: str, x: float, y: float, is_support: bool = False, support_type: str = None):
        """Add a node to the structure."""
        node = Node(name, x, y, is_support, support_type)
        self.nodes[name] = node  # Store by name instead of appending to list
        return node
    
    def add_member(self, name: str, start_node: Node, end_node: Node, EI: float):
        """Add a member to the structure."""
        member = Member(name, start_node, end_node, EI)
        self.members[name] = member  # Store by name instead of appending to list
        return member
    
    def add_load(self, member: Member, load_type: str, magnitude: float, 
             location: float = None, is_lateral: bool = False):
        """Add a load to a member."""
        if is_lateral:  # Track lateral loads separately
            self.horizontal_loads.append((magnitude, location))
        else:
            # Existing load handling
            if load_type == 'point' and location is None:
                location = member.length / 2
            load = Load(member, load_type, magnitude, location)
            self.loads.append(load)
        return load
    
    def calculate_fixed_end_moments(self, member: Member) -> Tuple[float, float]:
        """Calculate the fixed end moments for a member due to applied loads."""
        MFAB = 0  # Fixed end moment at start node
        MFBA = 0  # Fixed end moment at end node
        
        # Find all loads on this member
        member_loads = [load for load in self.loads if load.member.name == member.name]
        
        for load in member_loads:
            L = member.length
            
            if load.load_type == 'uniform':
                w = load.magnitude
                # Fixed end moments for uniformly distributed load
                MFAB -= (w * L**2) / 12
                MFBA += (w * L**2) / 12

            elif load.load_type == 'point':
                P = load.magnitude
                
                # Fixed end moments for point load
                MFAB -= (P * L) / 8
                MFBA += (P * L) / 8
                
            elif load.load_type == 'point_x':
                P = load.magnitude
                a = load.location
                b = L - a
                
                # Fixed end moments for point load at a distance
                MFAB -= (P * a * b**2) / L**2
                MFBA += (P * a**2 * b) / L**2
                
            elif load.load_type == 'double_point':
                P = load.magnitude
                
                # Fixed end moments for point load
                MFAB -= (2 * P * L) / 9
                MFBA += (2 * P * L) / 9

            elif load.load_type == 'half_uniform':
                w = load.magnitude
                # Fixed end moments for uniformly distributed load
                MFAB -= (11 * w * L**2) / 192
                MFBA += (5 * w * L**2) / 192

            elif load.load_type == 'moment':
                M = load.magnitude
                a = load.location
                b = L - a
                
                # Fixed end moments for applied moment
                MFAB -= (M * b) / L
                MFBA -= (M * a) / L
            
            else:
                MFAB = 0
                MFBA = 0

        return MFAB, MFBA
    
    def setup_equations(self):
        """Set up the equilibrium equations for the structure."""
        self.unknowns = {}
        self.equations = []

        # ==== Adjustment 1: Correct Free End Detection ====
        # Free ends are nodes with only ONE connected member and not supports
        free_ends = []
        for node in self.nodes.values():
            connected = self.get_connected_members(node)
            if len(connected) == 1 and not node.is_support:
                free_ends.append(node.name)

        # ==== Adjustment 2: Smarter Sway Detection ====
        # Check if horizontal translation is possible (true sway frames only)
        vertical_members = [m for m in self.members.values() 
                          if m.start_node.y != m.end_node.y]
        horizontal_restrained = any(
            n.support_type in ['fixed', 'pinned', 'roller'] 
            for n in self.nodes.values()
        )
        has_sidesway = (len(vertical_members) > 0 and not horizontal_restrained)

        # Add unknown rotations for non-fixed, non-free nodes
        for node in self.nodes.values():
            if node.support_type != 'fixed' and node.name not in free_ends:
                self.unknowns[f"theta_{node.name}"] = 0

        # Add sway unknown only if truly needed
        if has_sidesway:
            self.unknowns["delta"] = 0
            
        # Create equilibrium equations only for non-free nodes
        for node in self.nodes.values():
            # Skip free ends entirely
            if node.name in free_ends:
                continue

            # Process nodes that are not free ends(CANTILEVER)
            if not node.is_support or node.support_type != 'fixed':
                equation = {'constant': 0}
                connected_members = self.get_connected_members(node)

                for member in connected_members:
                    L = member.length
                    EI = member.EI
                    k = EI / L  # Stiffness coefficient

                    # Determine if node is start or end of the member
                    is_start = (member.start_node == node)
                    other_node = member.end_node if is_start else member.start_node

                    # Skip contributions from free-end nodes
                    if other_node.name in free_ends:
                        continue

                    # Fixed end moments
                    MFAB, MFBA = self.calculate_fixed_end_moments(member)

                    # Add fixed end moment to equation constant
                    if is_start:
                        equation['constant'] += MFAB
                    else:
                        equation['constant'] += MFBA

                    # Slope-deflection terms (only include non-free nodes)
                    current_theta = f"theta_{node.name}"
                    other_theta = f"theta_{other_node.name}"
                    
                    equation[current_theta] = equation.get(current_theta, 0) + 4 * k
                    if other_theta in self.unknowns:
                        equation[other_theta] = equation.get(other_theta, 0) + 2 * k

                    # Sway terms (if applicable)
                    if "delta" in self.unknowns and member.start_node.y != member.end_node.y:
                        sway_term = -6 * k / L
                        equation["delta"] = equation.get("delta", 0) + sway_term

                if equation != {'constant': 0}:
                    self.equations.append(equation)

        # Add global sway equilibrium equation if needed
        if has_sidesway:
            self.unknowns["delta"] = 0
            sway_equation = {'constant': 0}
            
            # Process all vertical members (columns)
            columns = [m for m in self.members.values() 
                    if m.start_node.y != m.end_node.y]
            
            for col in columns:
                L = col.length
                EI = col.EI
                k = EI / L

                # Get moments at both ends (will be calculated later)
                M_ab = f"M_{col.name}_start"
                M_ba = f"M_{col.name}_end"

                # Contribution from column moments (Σ(M_ab + M_ba)/L)
                sway_equation[M_ab] = sway_equation.get(M_ab, 0) + 1/L
                sway_equation[M_ba] = sway_equation.get(M_ba, 0) + 1/L

            # Add external horizontal loads (ΣP = 0)
            for load in self.horizontal_loads:
                P, h = load  # Magnitude and height from base
                sway_equation['constant'] -= P * h / L  # Adjust based on load position

            self.equations.append(sway_equation)
    
    def get_connected_members(self, node: Node) -> List[Member]:
        """Get all members connected to a node."""
        return [m for m in self.members.values() if m.start_node == node or m.end_node == node]
    
    def solve_equations(self):
        """Solve the system of equations to find unknown rotations and displacements."""
        n = len(self.unknowns)
        num_eq = len(self.equations)
        
        if num_eq != n:
            print(f"Error: {num_eq} equations but {n} unknowns. System is unbalanced.")
            return False
        
        A = np.zeros((n, n))
        b = np.zeros(n)
        unknown_keys = list(self.unknowns.keys())
        
        for i, eq in enumerate(self.equations):
            if i >= n:
                print(f"Error: Equation index {i} exceeds unknowns count {n}.")
                return False
            b[i] = -eq.get("constant", 0)
            for j, key in enumerate(unknown_keys):
                A[i, j] = eq.get(key, 0)
        
        try:
            x = np.linalg.solve(A, b)
            for i, key in enumerate(unknown_keys):
                self.unknowns[key] = x[i]
            return True
        except np.linalg.LinAlgError:
            print("Error: Singular matrix. Check for unstable structure or redundant constraints.")
            return False
    
    def calculate_member_end_moments(self):
        """Calculate the final end moments for all members."""
        for member in self.members.values():
            # Get fixed end moments
            MFAB, MFBA = self.calculate_fixed_end_moments(member)
            
            # Get rotations
            theta_A = self.unknowns.get(f"theta_{member.start_node.name}", 0)
            theta_B = self.unknowns.get(f"theta_{member.end_node.name}", 0)
            
            # Get sidesway if applicable
            delta = self.unknowns.get("delta", 0)
            
            # Calculate end moments using slope deflection equations
            k = member.EI / member.length
            
            # Check if we need to include sidesway term
            sway_term = 0
            if member.start_node.y != member.end_node.y:
                delta = self.unknowns.get("delta", 0)
                sway_term = -3 * delta / member.length
            
            # Calculate end moments
            MAB = MFAB + 2 * k * (2 * theta_A + theta_B + sway_term)
            MBA = MFBA + 2 * k * (theta_A + 2 * theta_B + sway_term)
            
            # Store results
            self.results[f"M_{member.name}_start"] = MAB
            self.results[f"M_{member.name}_end"] = MBA
    
    # def calculate_shear_forces(self):
    #     """Calculate shear forces for all members."""
    #     for member in self.members.values():
    #         # Get end moments
    #         MAB = self.results[f"M_{member.name}_start"]
    #         MBA = self.results[f"M_{member.name}_end"]
            
    #         # Initialize shear forces at both ends
    #         VAB = 0
    #         VBA = 0
            
    #         # Find all loads on this member
    #         member_loads = [load for load in self.loads if load.member == member]
            
    #         # Calculate reactions due to loads
    #         for load in member_loads:
    #             L = member.length
                
    #             if load.load_type == 'uniform':
    #                 w = load.magnitude
    #                 VAB += w * L / 2
    #                 VBA += w * L / 2
                    
    #             elif load.load_type == 'point':
    #                 P = load.magnitude
    #                 a = load.location
    #                 b = L - a
    #                 VAB += P * b / L
    #                 VBA += P * a / L
                    
    #             elif load.load_type == 'point_x':
    #                 P = load.magnitude
    #                 a = load.location
    #                 b = L - a
    #                 VAB += P * b / L
    #                 VBA += P * a / L
                    
    #             elif load.load_type == 'double_point':
    #                 P = load.magnitude
    #                 # Assuming two point loads at L/3 and 2L/3
    #                 a1 = L / 3
    #                 a2 = 2 * L / 3
    #                 VAB += P * (L - a1) / L + P * (L - a2) / L
    #                 VBA += P * a1 / L + P * a2 / L
                    
    #             elif load.load_type == 'half_uniform':
    #                 w = load.magnitude
    #                 # Uniform load over first half of the member
    #                 VAB += (3 * w * L) / 8
    #                 VBA += (w * L) / 8
                    
    #             elif load.load_type == 'moment':
    #                 # Moment loads do not contribute to shear
    #                 pass
            
    #         # Add contribution from end moments
    #         VAB += (MAB + MBA) / member.length
    #         VBA -= (MAB + MBA) / member.length
            
    #         # Store results
    #         self.results[f"V_{member.name}_start"] = VAB
    #         self.results[f"V_{member.name}_end"] = VBA
    
    def analyze(self):
        """Run the complete analysis process."""
        self.setup_equations()
        if self.solve_equations():
            self.calculate_member_end_moments()
            self.calculate_reactions()
            self.calculate_global_shear()  # New global shear calculation
            self.calculate_global_bending_moment()
            return True
        return False
    
    def calculate_global_shear(self):
        """Calculate global shear force diagram using reactions and loads."""
        # Create global x-axis
        total_length = max(n.x for n in self.nodes.values())
        x = np.linspace(0, total_length, 200)
        shear = np.zeros_like(x)

        # Apply support reactions first
        for reaction_key, reaction_value in self.support_reactions.items():
            if isinstance(reaction_key, str) and reaction_key.startswith('R_'):
                node_name = reaction_key.split('_')[1]  # Extract node name from R_NodeName
                node = self.nodes.get(node_name)
                if node:
                    shear[x >= node.x] += float(reaction_value)  # Convert to float
            elif hasattr(reaction_key, 'name') and reaction_key.name.startswith('R_'):
                # Handle Symbol objects
                node_name = reaction_key.name.split('_')[1]
                node = self.nodes.get(node_name)
                if node:
                    shear[x >= node.x] += float(reaction_value)

        # Apply loads (with proper signs)
        for load in self.loads:
            member = load.member
            start_x = member.start_node.x
            
            if load.load_type == 'point':
                load_pos = start_x + load.location
                # Point load causes sudden change in shear
                shear[x >= load_pos] -= load.magnitude
                
            elif load.load_type == 'uniform':
                end_x = member.end_node.x
                # For UDL, shear changes linearly across the member
                in_range = (x >= start_x) & (x <= end_x)
                shear[in_range] -= load.magnitude * (x[in_range] - start_x)
                shear[x > end_x] -= load.magnitude * (end_x - start_x)

        self.results["global_shear"] = shear
        self.results["global_shear_x"] = x

    def calculate_global_shear(self):
        """Calculate global shear force diagram using reactions and loads."""
        # Create global x-axis
        total_length = max(n.x for n in self.nodes.values())
        x = np.linspace(0, total_length, 200)
        shear = np.zeros_like(x)

        # Apply support reactions first
        for reaction_key, reaction_value in self.support_reactions.items():
            if isinstance(reaction_key, str) and reaction_key.startswith('R_'):
                node_name = reaction_key.split('_')[1]  # Extract node name from R_NodeName
                node = self.nodes.get(node_name)
                if node:
                    shear[x >= node.x] += float(reaction_value)  # Convert to float
            elif hasattr(reaction_key, 'name') and reaction_key.name.startswith('R_'):
                # Handle Symbol objects
                node_name = reaction_key.name.split('_')[1]
                node = self.nodes.get(node_name)
                if node:
                    shear[x >= node.x] += float(reaction_value)

        # Apply loads (with proper signs)
        for load in self.loads:
            member = load.member
            start_x = member.start_node.x
            end_x = member.end_node.x
            L = member.length
            
            if load.load_type == 'point':
                load_pos = start_x + load.location
                # Point load causes sudden change in shear
                shear[x >= load_pos] -= load.magnitude
                
            elif load.load_type == 'uniform':
                # For UDL, shear changes linearly across the member
                in_range = (x >= start_x) & (x <= end_x)
                shear[in_range] -= load.magnitude * (x[in_range] - start_x)
                shear[x > end_x] -= load.magnitude * (end_x - start_x)
                
            elif load.load_type == 'double_point':
                # For double point load at L/3 and 2L/3
                first_load_pos = start_x + L/3
                second_load_pos = start_x + 2*L/3
                
                # Apply first point load at L/3
                shear[x >= first_load_pos] -= load.magnitude
                
                # Apply second point load at 2L/3
                shear[x >= second_load_pos] -= load.magnitude

        self.results["global_shear"] = shear
        self.results["global_shear_x"] = x

    def calculate_global_bending_moment(self):
        """Calculate global bending moment diagram from shear force diagram."""
        x = self.results["global_shear_x"]
        shear = self.results["global_shear"]
        
        # Initialize moment array with zero at left end
        moment = np.zeros_like(x)
        
        # Integrate shear to get moment
        dx = x[1] - x[0]  # Step size
        
        # Integrate from left to right
        for i in range(1, len(x)):
            # Trapezoidal rule for integration
            moment[i] = moment[i-1] + (shear[i-1] + shear[i])/2 * dx
        
        # Adjust based on known end moments
        # Find all end nodes
        support_nodes = [n for n in self.nodes.values() if n.is_support]
        
        # Apply correction to ensure moment is correct at support locations
        for node in support_nodes:
            # Find connected members
            connected = self.get_connected_members(node)
            for member in connected:
                # Check if this is start or end node of member
                if member.start_node == node:
                    known_moment = self.results[f"M_{member.name}_start"]
                    idx = np.abs(x - node.x).argmin()  # Find closest x index
                    # Calculate offset needed
                    offset = known_moment - moment[idx]
                    # Apply offset to all moments after this point
                    moment[idx:] += offset
                    break
        
        self.results["global_bending_moment"] = moment
        self.results["global_bending_x"] = x

    def plot_results(self):
        """Plot the structure with bending moment and shear force diagrams."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Structure
        plt.subplot(3, 1, 1)
        plt.title('Structure')
        
        # Plot nodes
        for node in self.nodes.values():
            plt.plot(node.x, node.y, 'o', markersize=10)
            plt.text(node.x, node.y + 0.2, node.name)
        
        # Plot members
        for member in self.members.values():
            plt.plot([member.start_node.x, member.end_node.x], 
                    [member.start_node.y, member.end_node.y], 'k-', linewidth=2)
            
            # Plot loads
            member_loads = [load for load in self.loads if load.member == member]
            for load in member_loads:
                if load.load_type == 'uniform':
                    # Draw uniform load arrows
                    x_start, y_start = member.start_node.x, member.start_node.y
                    x_end, y_end = member.end_node.x, member.end_node.y
                    arrow_spacing = member.length / 10
                    num_arrows = int(member.length / arrow_spacing)
                    for i in range(num_arrows + 1):
                        t = i / num_arrows
                        x = x_start + t * (x_end - x_start)
                        y = y_start + t * (y_end - y_start)
                        arrow_length = 0.3
                        if abs(y_end - y_start) < 0.001:  # Horizontal member
                            plt.arrow(x, y + arrow_length, 0, -arrow_length, head_width=0.1, head_length=0.05, 
                                    fc='r', ec='r', length_includes_head=True)
                        else:  # Vertical member
                            plt.arrow(x + arrow_length, y, -arrow_length, 0, head_width=0.1, head_length=0.05, 
                                    fc='r', ec='r', length_includes_head=True)
                            
                elif load.load_type == 'point':
                    # Draw point load arrow
                    load_pos_x = member.start_node.x + load.location
                    load_pos_y = member.start_node.y
                    arrow_length = 0.5
                    plt.arrow(load_pos_x, load_pos_y + arrow_length, 0, -arrow_length, 
                            head_width=0.1, head_length=0.1, fc='r', ec='r')
                    plt.text(load_pos_x, load_pos_y + arrow_length, f"{load.magnitude} kN")
                    
                elif load.load_type == 'double_point':
                    # Draw two point load arrows at L/3 and 2L/3
                    L = member.length
                    x_start, y_start = member.start_node.x, member.start_node.y
                    x_end, y_end = member.end_node.x, member.end_node.y
                    
                    # First point at L/3
                    first_pos_x = x_start + L/3
                    first_pos_y = y_start + (y_end - y_start) * (1/3)
                    arrow_length = 0.5
                    plt.arrow(first_pos_x, first_pos_y + arrow_length, 0, -arrow_length, 
                            head_width=0.1, head_length=0.1, fc='r', ec='r')
                    plt.text(first_pos_x, first_pos_y + arrow_length, f"{load.magnitude} kN")
                    
                    # Second point at 2L/3
                    second_pos_x = x_start + 2*L/3
                    second_pos_y = y_start + (y_end - y_start) * (2/3)
                    plt.arrow(second_pos_x, second_pos_y + arrow_length, 0, -arrow_length, 
                            head_width=0.1, head_length=0.1, fc='r', ec='r')
                    plt.text(second_pos_x, second_pos_y + arrow_length, f"{load.magnitude} kN")
        
        plt.grid(True)
        plt.axis('equal')
            
        # Plot 2: Shear Force Diagram (inverted convention)
        plt.subplot(3, 1, 2)
        plt.title('Shear Force Diagram')
        plt.plot(self.results["global_shear_x"], -self.results["global_shear"], 'b-')
        plt.fill_between(self.results["global_shear_x"], -self.results["global_shear"], alpha=0.3)
        plt.grid(True)
        
        # Plot 3: Bending Moment Diagram (inverted convention)
        plt.subplot(3, 1, 3)
        plt.title('Bending Moment Diagram')
        plt.plot(self.results["global_bending_x"], -self.results["global_bending_moment"], 'r-')
        plt.fill_between(self.results["global_bending_x"], -self.results["global_bending_moment"], alpha=0.3)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def print_member_details(self):
        """Print detailed bending moment and shear force values for each member."""
        print("\n==== Member Details ====")
        for member in self.members.values():
            print(f"\nMember {member.name}:")
            
            # Bending Moments
            print("\nBending Moments:")
            x_bm = self.results[f"BM_x_{member.name}"]
            bm = self.results[f"BM_{member.name}"]
            for x, m in zip(x_bm, bm):
                print(f"x = {x:.2f} m | BM = {m:.2f} kNm")
            
            # Shear Forces
            print("\nShear Forces:")
            x_sf = self.results[f"SF_x_{member.name}"]
            sf = self.results[f"SF_{member.name}"]
            for x, s in zip(x_sf, sf):
                print(f"x = {x:.2f} m | SF = {s:.2f} kN")

    def print_results(self):
        """Print analysis results."""
        print("\n==== Analysis Results ====")
        
        # Print unknown rotations and displacements
        print("\nUnknown Rotations and Displacements:")
        for key, value in self.unknowns.items():
            print(f"{key} = {value:.6f}")
        
        # Print member end moments
        print("\nMember End Moments:")
        for member in self.members.values():
            MAB = self.results[f"M_{member.name}_start"]
            MBA = self.results[f"M_{member.name}_end"]
            print(f"Member {member.name}:")
            print(f"  M_{member.start_node.name}{member.end_node.name} = {MAB:.2f} kNm")
            print(f"  M_{member.end_node.name}{member.start_node.name} = {MBA:.2f} kNm")
        
        # Print global shear forces
        print("\nGlobal Shear Forces:")
        print(f"Max Shear = {np.max(self.results['global_shear']):.2f} kN")
        print(f"Min Shear = {np.min(self.results['global_shear']):.2f} kN")
        
        # Print global bending moments
        print("\nGlobal Bending Moments:")
        print(f"Max Moment = {np.max(self.results['global_bending_moment']):.2f} kNm")
        print(f"Min Moment = {np.min(self.results['global_bending_moment']):.2f} kNm")
    
    def calculate_reactions(self):
        """Calculate support reactions using equilibrium equations."""
        self.support_reactions = defaultdict(float)
        span_data = self._prepare_span_data()
        
        # Process each span independently
        for span_label, data in span_data.items():
            reactions = self._solve_span_equations(data)
            for reaction, value in reactions.items():
                self.support_reactions[reaction] += value

    def _prepare_span_data(self):
        """Organize member data into span-based format."""
        span_data = {}
        for member in self.members.values():
            span_label = member.name
            span_data[span_label] = {
                'length': member.length,
                'loads': [load for load in self.loads if load.member == member],
                'moments': (
                    self.results[f"M_{member.name}_start"],
                    self.results[f"M_{member.name}_end"]
                ),
                'start_node': member.start_node.name,  # ADD THESE
                'end_node': member.end_node.name       # TWO LINES
            }
        return span_data

    def _solve_span_equations(self, data):
        """Solve equilibrium equations for a single span."""
        L = data['length']
        R_start = symbols(f"R_{data['start_node']}")
        R_end = symbols(f"R_{data['end_node']}")
        
        # Sum of vertical forces
        total_load = sum(self._convert_load(load) for load in data['loads'])
        eq1 = Eq(R_start + R_end, total_load)

        # Sum of moments
        moment_sum = data['moments'][0] + data['moments'][1]
        for load in data['loads']:
            moment_sum += self._calculate_load_moment(load, L)
        eq2 = Eq(R_start * L - moment_sum, 0)

        return solve((eq1, eq2), (R_start, R_end))

    def _convert_load(self, load):
        """Convert UDL to equivalent point load."""
        if load.load_type == 'uniform':
            return load.magnitude * load.member.length
        return load.magnitude

    def _calculate_load_moment(self, load, span_length):
        """Calculate moment contribution from a load."""
        if load.load_type == 'uniform':
            return load.magnitude * span_length**2 / 2
        elif load.load_type == 'double_point':
            return load.magnitude * span_length  # Each point is load.magnitude, total moment is magnitude * span_length
        else:
            return load.magnitude * load.location

    def print_results(self):
        """Print analysis results including reactions."""
        # ... existing print code ...
        
        print("\n=== Support Reactions ===")
        for reaction, value in self.support_reactions.items():
            print(f"{reaction} = {float(value):.2f} kN")


# Example usage - Simple continuous beam
def example_continuous_beam():
    analyzer = StructureAnalyzer()
    
    # Define nodes
    node_A = analyzer.add_node("A", 0, 0, is_support=True, support_type="fixed")
    node_B = analyzer.add_node("B", 4, 0, is_support=True, support_type="roller")
    node_C = analyzer.add_node("C", 10, 0, is_support=True, support_type="roller")
    node_D = analyzer.add_node("D", 14, 0, is_support=True, support_type="fixed")
    
    # Define members
    EI = 1000  # Assumed constant EI value
    member_AB = analyzer.add_member("AB", node_A, node_B, EI)
    member_BC = analyzer.add_member("BC", node_B, node_C, 2*EI)  # 2I for the middle section
    member_CD = analyzer.add_member("CD", node_C, node_D, EI)
    
    # Add loads
    analyzer.add_load(member_AB, "uniform", 24)  # 20 kN/m
    analyzer.add_load(member_BC, "double_point", 80)  # 80 kN at 2m from B
    analyzer.add_load(member_CD, "uniform", 24)  # 15 kN/m
    
    # Run analysis
    if analyzer.analyze():
        analyzer.print_results()
        analyzer.plot_results()
    else:
        print("Analysis failed.")

# Example usage - Portal frame
def example_portal_frame():
    analyzer = StructureAnalyzer()
    
    # Define nodes
    node_A = analyzer.add_node("A", 0, 0, is_support=True, support_type="fixed")
    node_B = analyzer.add_node("B", 0, 5, is_support=False)
    node_C = analyzer.add_node("C", 4, 5, is_support=False)
    node_D = analyzer.add_node("D", 4, 0, is_support=True, support_type="fixed")
    
    # Define members
    EI = 1000  # Assumed constant EI value
    member_AB = analyzer.add_member("AB", node_A, node_B, EI)
    member_BC = analyzer.add_member("BC", node_B, node_C, EI)
    member_CD = analyzer.add_member("CD", node_C, node_D, EI)
    
    # Add loads
    analyzer.add_load(member_BC, "point", -50, 2)  # 50 kN at mid-span
    
    # Run analysis
    if analyzer.analyze():
        analyzer.print_results()
        analyzer.plot_results()
    else:
        print("Analysis failed.")

# Run examples
if __name__ == "__main__":
    print("Example 1: Continuous Beam Analysis")
    example_continuous_beam()
    
    print("\nExample 2: Portal Frame Analysis")
    example_portal_frame()