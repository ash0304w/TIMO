def repair_subset(subset, beta, prompt_num):
    cleaned = []
    seen = set()
    for i in subset:
        j = int(max(0, min(prompt_num - 1, i)))
        if j not in seen:
            cleaned.append(j)
            seen.add(j)
    i = 0
    while len(cleaned) < beta and i < prompt_num:
        if i not in seen:
            cleaned.append(i)
        i += 1
    return cleaned[:beta], float(len(cleaned) != beta)
