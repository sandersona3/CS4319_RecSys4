'''[Optional]Used to Time stamp'''

import datetime
import time
start = time.time()
clock = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
#print(clock)
