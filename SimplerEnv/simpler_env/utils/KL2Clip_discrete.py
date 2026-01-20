from scipy.optimize import fsolve


from dotmap import DotMap

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from tensorflow.python.framework import ops

# import tensorflow as tf
from math import log, exp
import os
import fcntl


class KL2Clip_fsolve(object):

    def f_setting1(self, pa, delta):
        def f(m):
            m = float(m)
            if (1 + m * pa) == 0:
                return None
            part_1 = (m ** (1 - pa))
            part_2 = ((1 / pa + m) ** pa)
            part_3 = ((1 - pa) / m + pa ** 2 / (1 + m * pa))
            part_1_2 = part_1 * part_2
            # return [part_1_2, part_3]
            if isinstance(part_1_2, complex):
                if abs(part_1_2.imag) > 1e-10:
                    return None
                part_1_2 = part_1_2.real
            part_1_2_3 = part_1_2 * part_3
            if part_1_2_3 > 0:
                return log(part_1_2_3) - delta
            return None

        return f

    def f_setting2(self, pa, delta):
        def f(r):
            if r < 0 \
                    or ((1 - pa) / (1 - pa * r)) < 0:
                return None
            return (1 - pa) * log((1 - pa) / (1 - pa * r)) - pa * log(r) - delta

        return f

    def opt_entity1(self, pa, delta, type='max', sol_ini=None):
        pa, delta = (float(item) for item in (pa, delta))
        f = self.f_setting1(pa, delta)
        if type == 'min':
            if sol_ini is not None:
                m_ini = sol_ini
            else:
                m_ini = 1
        else:
            if sol_ini is not None:
                m_ini = sol_ini
            else:
                m_ini = -1
                while f(m_ini) is None:
                    m_ini -= 1
        m = fsolve(f, m_ini, full_output=0)

        # lam  = exp(log(abs((m ** p0) * ((1 / p1 + m) ** p1))) - delta)
        # if m < 0:
        #     lam = -lam
        if isinstance(m, np.ndarray) and m.ndim >= 1:
            m = m[0]
        m = float(m)
        lam = m ** (1 - pa) * (1 / pa + m) ** pa / exp(delta)
        if isinstance(lam, complex):
            lam = lam.real
        qi_sum = lam * (1 - pa) / m
        qa = lam * pa / (1 / pa + m)
        ratio = qa / pa
        # print(f'type:{type},ratio:{ratio},qa:{qa},kl_constraint:{f(m)},m:{m},lam:{lam}')
        return ratio, m


    def opt_entity2(self, pa, delta, type='max', sol_ini=None):
        pa, delta = (float(item) for item in (pa, delta))
        f = self.f_setting2(pa, delta)
        if type == 'min':
            if sol_ini is not None:
                r_ini = sol_ini
            else:
                r_ini = 1. / exp(0.011 * (1. / pa))
        else:
            if sol_ini is not None:
                r_ini = sol_ini
            else:
                r_ini = exp(0.011 * (1. / pa))
                while f(r_ini) is None:
                    r_ini -= (r_ini - 1) * 0.9
        r = fsolve(f, r_ini, full_output=0)

        if isinstance(r, np.ndarray) and r.ndim >= 1:
            r = r[0]
        return r

    def __call__(self, pas, delta, initialwithpresol=False):
         # This was used as an argument before. But I make it fix now. If you set initialwithpresol=True, it will use the previous solution as initial solution. It will result in some issues. For e.g., when run result for 0.0001, 0.0002, ...., the ratio_max in fact would change greatly.
        ratio_pre = None
        sol_pre = None
        ratio_maxs = []
        for pa in pas:

            if ratio_pre is not None and ratio_pre > 2:
                sol_pre = None # initial solution will result in issue if its value is very large.

            if pa <= 0.95:
                ratio, sol_pre = self.opt_entity1(pa, delta, 'max', sol_pre if initialwithpresol else None)
                ratio_maxs.append(ratio)
                ratio_pre = ratio
            else:
                ratio_pre = ratio = self.opt_entity2(pa, delta, 'max', ratio_pre if initialwithpresol else None)
                ratio_maxs.append(ratio)

            # print(f'pa:  {pa}   ratio_max:  {ratio}')

        ratio_pre = None
        sol_pre = None
        ratio_mins = []
        for pa in pas:
            # print(f'pa:  {pa}   ratio_min:  {ratio}')
            if pa <= 0.95:
                ratio, sol_pre = self.opt_entity1(pa, delta, 'min', sol_pre if initialwithpresol else None)
                ratio_mins.append(ratio)
                ratio_pre = ratio
            else:
                ratio_pre = ratio = self.opt_entity2(pa, delta, 'min', ratio_pre if initialwithpresol else None)
                ratio_mins.append(ratio)

        ratio_maxs = np.array(ratio_maxs)
        ratio_mins = np.array(ratio_mins)
        return DotMap(max=ratio_maxs, min=ratio_mins)


path_root = os.path.dirname(os.path.abspath(__file__))
path_root_tabular = f'{path_root}/KL2Clip_discrete_tabular'
os.makedirs(path_root_tabular, exist_ok=True)

from . import tools

class KL2Clip(object):
    def __init__(self):
        self.tabular = {}


    def get_tabular(self, delta, decimal_places):
        file_name = f'delta={delta:.16f},decimal_places={decimal_places}'
        file_path = f'{path_root_tabular}/{file_name}'
        if file_name in self.tabular:
            pass
        elif os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            self.tabular[file_name] = tools.load_vars(file_path)
        else:
            with tools.FileLocker(f'{path_root_tabular}/.{file_name},FileLocker' ):
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    self.tabular[file_name] = tools.load_vars(file_path)
                else:
                    self.tabular[file_name] = self.create_tabular(delta, decimal_places=decimal_places)
                    tools.save_vars(file_path, self.tabular[file_name])
        # self.tabular[file_name] = self.create_tabular(delta, decimal_places)
        return self.tabular[file_name]

    def create_tabular(self, delta, decimal_places):
        time0 = time.time()
        step = self.get_step( decimal_places  )
        print(f'Generating KL2Clip tabular, delta={delta}, decimal_places={decimal_places}')
        fsolver = KL2Clip_fsolve()
        p_grid = np.arange(0, 1.0+step,step )

        start= 1
        end =1
        ratio_dicts = fsolver(pas=p_grid[start:-end], delta=delta, initialwithpresol=True)
        ratio_min, ratio_max = ratio_dicts.min, ratio_dicts.max

        ratio_min = np.concatenate( ( [ ratio_min[0] ]*start, ratio_min, [1.] *(end+1)  )  )
        ratio_max = np.concatenate(([ ratio_max[0] ]*start, ratio_max, [1.]*(end+1) ))
        tabular = np.zeros(  (p_grid.shape[0], 5 ), dtype=np.float64 )
        for i in range(p_grid.shape[0]):
            assert p_grid[i] == step * i
            tabular[ i ] = [ ratio_min[i], ratio_max[i],
                             ratio_min[i+1], ratio_max[i+1] ,
                             p_grid[i]
                             ]
        time0 = time.time() - time0
        print(f'Finished generated KL2Clip tabular, delta={delta}, decimal_places={decimal_places}, with time {time0}s')
        return tabular

    def get_step(self, decimal_places):
        return 10 ** (-decimal_places)


    def __call__(self, probabilities, delta, decimal_places=4):
        '''
        :param probabilities: Array of probability values, shape (N,).
        :param delta: Delta parameter。
        :param decimal_places: Decimal precision for discretizing the unit interval.
                               For example, decimal_places=3 generates values at increments of 0.001 (10^{-3}).
        :return: DotMap(min=clippingrange_lower, max=clippingrange_upper), where `min` and `max` denote
                 the lower and upper clipping bounds, respectively.
        '''
        assert probabilities.ndim == 1 and np.all((probabilities >= 0) & (probabilities <= 1))

        step = self.get_step(decimal_places)

        tabular = self.get_tabular(delta=delta, decimal_places=decimal_places)

        ps_int =  (probabilities * (10 ** decimal_places)).astype(int)

        result = tabular[ ps_int ]

        ratio_min_y1 = result[:, 0]
        ratio_min_y2 = result[:, 2]

        ratio_max_y1 = result[:, 1]
        ratio_max_y2 = result[:, 3]

        x1 = result[:, 4]
        x2 = result[:, 4] + step

        ratio_min = tools.linear_interp(
            x=probabilities,
            x1=x1,
            y1=ratio_min_y1,
            x2=x2,
            y2=ratio_min_y2,
        )

        ratio_max = tools.linear_interp(
            x=probabilities,
            x1=x1,
            y1=ratio_max_y1,
            x2=x2,
            y2=ratio_max_y2,
        )

        assert not np.any( np.isnan(ratio_min) )
        assert not np.any(np.isnan(ratio_max))

        return DotMap(min=ratio_min, max=ratio_max)




def tes_kl2clip_discrete():
    delta = 0.07
    ps = np.arange(0, 1., 0.0001)
    kl2clip = KL2Clip( )
    result = kl2clip(ps, delta, decimal_places=4)


    plt.scatter(ps, result.max, c='blue', s=5)
    plt.scatter(ps, result.min, c='blue', s=5)

    # plt.plot(ps, result.max, 'blue')

    # plt.plot(ps, result.min, 'red')
    plt.show()

    perm = np.random.permutation(len(ps))
    ps = ps[perm]

    result = kl2clip(ps, delta, decimal_places=2)
    plt.scatter(ps, result.max, c='red', s=5)
    plt.scatter(ps, result.min, c='red', s=5)

    plt.show()
    # plt.pause(1e10)

if __name__ == '__main__':
    tes_kl2clip_discrete()


