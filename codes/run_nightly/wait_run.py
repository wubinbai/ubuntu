import os
import time
for i in range(1000):
    t = time.time()
    print(t)
    hour = time.localtime().tm_hour
    #print(hour)
    print(hour)
    if hour == 2:
        os.system('./script.sh')
    time.sleep(60*30)
