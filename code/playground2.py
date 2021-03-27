import time
import random
data = set()

limit = 10000000
size_at_time = 0
largest_time = 0

for _ in range(limit):
    num = random.randint(0, limit)
    start = time.time()
    if num != data:
        data.add(num)
    duration = time.time() - start

    if duration > largest_time:
        largest_time = duration
        size_at_time = len(data)

print(f'Longest time when set had {size_at_time:,} elements, taking {largest_time} time.')



    
