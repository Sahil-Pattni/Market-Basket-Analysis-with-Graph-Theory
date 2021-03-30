x = 0
for b in range(3):
    for c in range(3):
        if b == 2:
            break
        x += 1

print(x)

below_threshold = 0
ab = 0
below_threshold_count = 0


for _ in range(10):
    # Old
    if any([set(x).issubset(ab) for x in below_threshold]):
        below_threshold_count += 1
        continue

    # New
    for x in below_threshold:
        if set(x).issubset(ab):
            below_threshold_count += 1
            is_above_threshold = False
            break
