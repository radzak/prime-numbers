from collections import defaultdict
from functools import partial
from functools import wraps
import time
import timeit


timings = defaultdict(lambda: defaultdict(list))


def time_once(f):
    # @wraps(f)
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.7f sec' % (f.__name__, args, kw, te-ts))
        timings[f.__name__]['x'].append(kw['n'])  # x axis
        timings[f.__name__]['y'].append(te - ts)  # y axis
        return result
    return timed


def timer(repeats=5):
    def real_timer(f):
        def timed(*args, **kwargs):
            exec_time = min(timeit.Timer(partial(f, *args, **kwargs)).repeat(repeat=repeats, number=1))
            print('%s%r took: %2.7f sec' % (f.__name__, args, exec_time))
            timings[f.__name__]['x'].append(kwargs['n'])  # x axis
            timings[f.__name__]['y'].append(exec_time)  # y axis
            pass
        return timed
    return real_timer
