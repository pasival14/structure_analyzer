def calculate_reactions_and_shear_force(number_of_spans,
                                        number_of_supports,
                                        span_lengths: list,
                                        load_types: list,
                                        load_values: list,
                                        cantilever_spans: list,
                                        span_labels: list,
                                        load_distances_from_left_support,
                                        load_distances_from_left_of_beam,
                                        moment_list: list,
                                        ):
    """
    Calculate reactions_symbols at supports and shear forces for a beam with multiple loads per span.

    Args:
        number_of_spans (int): Number of spans.
        number_of_supports (int): Number of supports
        span_lengths (list): Lengths of each span.
        load_types (list): Types of loads ("udl", "point)
        load_values (list): Magnitude of each load
        span_labels (list): Span labels ("AB", "BC" etc...)
        load_distances_from_left_support (list): List of distances of each load from the leftmost support.
        moments (list): list of moments of each span

    Returns:
        dict: Reactions at supports.
        list: Shear force values at key positions.
    """

    counter = 0
    support_reaction_list = []
    support_reactions_dict = defaultdict(float)
    for span in span_lengths:
        reactions_symbols = []
        span_length = span_lengths[counter]
        span_label = span_labels[counter]
        load_type = load_types[counter]
        span_load_values = load_values[counter]
        load_value = []
        cantilever_span = cantilever_spans[counter]
        distance_from_left_support = load_distances_from_left_support[counter]
        moments = moment_list[counter]

        # Convert udl to point load
        n = 0
        for load in span_load_values:
            if load_type[n] == "udl":
                load = span_load_values[n] * span_length
                load_value.append(load)
                distance_from_left_support[n] = span_length/2
            else:
                load_value.append(load)
            n += 1

        globals()[f"R_{span_label[0]}"] = f"R_{span_label[0]}"
        globals()[f"R_{span_label[1]}"] = f"R_{span_label[1]}"
        reactions_symbols.extend([symbols(f"R_{span_label[0]}"), symbols(f"R_{span_label[1]}")]) if not cantilever_span else reactions_symbols.extend([symbols(f"R_{span_label[0]}")])

        if not cantilever_span:
            eq1 = Eq(reactions_symbols[0] + reactions_symbols[1], sum(load_value))

            # Load moments
            n, load_moment = 0, 0
            for load in load_value:
                load_moment += (load * (span_length - distance_from_left_support[n]))
                n += 1

            eq2 = Eq((span_length * reactions_symbols[0]) + moments[1] + moments[0] - load_moment, 0)
        elif cantilever_span:
            eq1 = Eq(reactions_symbols[0], sum(load_value))

        # Solve
        solutions = solve((eq1, eq2), reactions_symbols)
        support_reaction_list.append(solutions)

        counter += 1

    # Convert sympy values to strigns and floats
    support_reaction_list = [{str(k): float(v) for k, v in reaction.items()} for reaction in support_reaction_list]

    for d in support_reaction_list:
        for key, value in d.items():
            support_reactions_dict[str(key)] += value

    # Convert reactions to a list
    support_reactions_list = [float(value) for key, value in support_reactions_dict.items()]

    # Convert to original dict
    support_reactions_dict = dict(support_reactions_dict)

    # Define spans
    total_length = sum(span_lengths)  # Total beam length
    supports = np.cumsum([0] + span_lengths)  # Support locations

    # Given reactions at supports (example values, replace with actual reactions)
    reactions = support_reactions_list  # kN at each support

    # Define loads
    loads = [
        {"type": "udl", "start": 0, "end": 4, "mag": 20},  # UDL on Span 1 (3 kN/m from 0m to 4m)
        {"type": "point", "position": 6, "mag": 80},  # Point Load on Span 2 (8 kN at x=6m)
        {"type": "point", "position": 8, "mag": 80},  # Point Load on Span 2 (5 kN at x=8m)
        {"type": "udl", "start": 10, "end": 14, "mag": 15},  # UDL on Span 3 (2 kN/m from 9m to 12m)
    ]

    # Extracting loads from beam data
    loads = []

    for span_index, span_type in enumerate(load_types):
        for load_index, load in enumerate(span_type):
            if load == "udl":
                start = load_distances_from_left_of_beam[span_index][0]
                end = start + span_lengths[span_index]
                mag = load_values[span_index][load_index]
                loads.append({"type": "udl", "start": start, "end": end, "mag": mag})
            elif load == "point":
                position = load_distances_from_left_of_beam[span_index][load_index]
                mag = load_values[span_index][load_index]
                loads.append({"type": "point", "position": position, "mag": mag})

    # Define x values for plotting
    x = np.linspace(0, total_length, 200)
    shear_force = np.zeros_like(x)

    # Apply reactions at supports
    for i, support in enumerate(supports[:-1]):  # Ignore last cumulative sum
        shear_force[x >= support] += reactions[i]

    # Apply loads
    for load in loads:
        if load["type"] == "point":
            shear_force[x >= load["position"]] -= load["mag"]
        elif load["type"] == "udl":
            in_range = (x >= load["start"]) & (x <= load["end"])
            shear_force[in_range] -= load["mag"] * (x[in_range] - load["start"])
            shear_force[x > load["end"]] -= load["mag"] * (load["end"] - load["start"])

    # PLOT SHEAR FORCE DIAGRAM
    plot_shear_force(x, shear_force)


    # BENDING MOMENT DIAGRAM
    counter = 0
    cummulative_span_lengths_sum = [sum(span_lengths[:i]) for i in range(len(span_lengths))]
    m = np.array([])
    dist = np.array([])
    for span in span_lengths:
        span_length = span_lengths[counter]
        left_moment = moment_list[counter][0]
        left_reaction = support_reaction_list[counter][list(support_reaction_list[counter].keys())[0]]
        load_type = load_types[counter]
        load_value = load_values[counter]
        distance_from_left_support = load_distances_from_left_support[counter]
        distance = np.arange(0, span_length+0.01, 0.01)

        n = np.array([bending_moment(load_type, load_value, left_reaction, left_moment, x, distance_from_left_support) for x in distance])

        distance = distance + cummulative_span_lengths_sum[counter]

        dist = np.concatenate((dist, distance))
        m = np.concatenate((m, n))

        counter += 1

    # PLOT BENDING MOMENT DIAGRAM
    plot_bending_moment(dist, m)



    return support_reactions_dict


def bending_moment(load_type, load_value, left_reaction, left_moment, x, distance_from_left_support):
    for load in load_type:
        if load == "udl":
            moment = left_moment + (left_reaction * x) - ((load_value[0] * x * x/2))
        elif load == "point":
            moment = left_moment + (left_reaction * x)
            for P, a in zip(load_value, distance_from_left_support):
                if x >= a:  # Apply effect of point loads only after their position
                    moment -= P * (x - a)
        return moment


def plot_shear_force(x, shear_force):
    # Plot Shear Force Diagram
    plt.figure(figsize=(10, 5))
    plt.plot(x, shear_force, label="Shear Force", color="b", linewidth=2)
    plt.fill_between(x, shear_force, alpha=0.3, color="blue")
    plt.axhline(0, color="black", linewidth=0.8)

    # Labels and formatting
    plt.title("Shear Force Diagram")
    plt.xlabel("Beam Length (m)")
    plt.ylabel("Shear Force (kN)")
    plt.grid()
    plt.show()


def plot_bending_moment(dist, moment):
    plt.figure(figsize=(10, 5))
    plt.plot(dist, moment, label="Bending Moment", color="r", linewidth=2)
    plt.fill_between(dist, moment, color="red", alpha=0.5, hatch="||")
    plt.axhline(0, color="black", linewidth=0.8)

    # Labels and formatting
    plt.title("Bending Moment Diagram")
    plt.xlabel("Beam Length (m)")
    plt.ylabel("Bending Moment (kNm)")
    plt.legend()
    plt.grid()
    plt.show()