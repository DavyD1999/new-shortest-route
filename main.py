# timer
import time

i = 0
start_time = time.time()
for i in range(10**7):
    i += 1

print(time.time()-start_time)
