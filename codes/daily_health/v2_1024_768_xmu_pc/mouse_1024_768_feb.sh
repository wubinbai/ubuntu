#!/bin/bash
# for debug
SLEEP_TIME=10


### fuwudatin
xdotool mousemove 551 198 click 1
sleep $SLEEP_TIME

### jiaozhigong
xdotool mousemove 193 640 click 1
sleep $SLEEP_TIME

### daily
xdotool mousemove 979 358 click 1
#xdotool mousemove 985 461 click 1
sleep $((SLEEP_TIME*2))

############# solvin jingling problem
### move to the jingling for a moment
xdotool mousemove 897 377
sleep 0.1
#### move to the center for the moment
xdotool mousemove 528 316
sleep 0.1

### submit
xdotool mousemove 792 308 click 1
sleep $SLEEP_TIME

### confirm
xdotool mousemove 513 576 click 1
sleep $SLEEP_TIME

