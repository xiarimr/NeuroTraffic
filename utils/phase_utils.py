def get_phase_encoding(phase_map, phase_value):
    if isinstance(phase_value, (list, tuple)):
        if not phase_value:
            raise KeyError("Empty phase value")
        phase_value = phase_value[0]

    candidates = [phase_value]

    if isinstance(phase_value, float) and phase_value.is_integer():
        int_value = int(phase_value)
        candidates.extend([int_value, str(int_value)])
    elif isinstance(phase_value, int):
        candidates.append(str(phase_value))
    elif isinstance(phase_value, str):
        stripped = phase_value.strip()
        if stripped != phase_value:
            candidates.append(stripped)
        if stripped.isdigit():
            candidates.append(int(stripped))

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        marker = (type(candidate), candidate)
        if marker in seen:
            continue
        seen.add(marker)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if candidate in phase_map:
            return phase_map[candidate]

    raise KeyError(
        "Unable to resolve phase key {0!r}. Available keys: {1}".format(
            phase_value, list(phase_map.keys())
        )
    )
