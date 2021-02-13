import os
import time

MAX_TRIALS = 5
MAX_ITER = int(1e99)
NAP_HOUR = 0.2 # 0.2 hours
NAP_SEC = int(NAP_HOUR * 60 * 60)
DO_AT = 6 # 6 A.M.
DO_AT2 = 10 # Recheck at 11 AM.

########## init dict
d = dict()
for i in range(1,366):
    d[i] = 0

########## init dict2
d2 = dict()
for i in range(1,366):
    d2[i] = 0

for i in range(MAX_ITER):
    
    t = time.localtime()
    hour = t.tm_hour
    tm_yday = t.tm_yday
    
    if hour == DO_AT: # if it is the time(correct hour) to do it
        if d[tm_yday] == 0: # if have not done for less than the number of trials
            for num_trials in range(MAX_TRIALS):
                os.system('date')
                start = time.time()
                os.system('./do_it_feb.sh')
                end = time.time()
                print('Done!', round(end - start,3), 'secs')
                d[tm_yday] += 1
    
    if hour == DO_AT2: # if it is the time(correct hour) to do it
        if d2[tm_yday] == 0: # if have not done for less than the number of trials
            for num_trials in range(MAX_TRIALS):
                os.system('date')
                start = time.time()
                os.system('./do_it.sh')
                end = time.time()
                print('Done!!', round(end - start,3), 'secs')
                d2[tm_yday] += 1
    
    
    time.sleep(NAP_SEC)
