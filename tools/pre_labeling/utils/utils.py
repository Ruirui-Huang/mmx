import json
import numpy as np
import multiprocessing 

def multi_processing_pipeline(single_func, task_list, n_process=None, callback=None, **kw):
    pool = multiprocessing.Pool(processes=n_process)
    process_pool = []
    for i in range(len(task_list)):
        process_pool.append(
            pool.apply_async(single_func, args=(task_list[i], ), kwds=kw, callback=callback)
        )
    pool.close()
    pool.join()
    print('success!')
    return process_pool

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)