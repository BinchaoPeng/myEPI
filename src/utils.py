import time, math
import numpy as np

def time_since(start):
    s = time.time() - start
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)