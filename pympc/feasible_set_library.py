import time
import numpy as np
from geometry.polytope import Polytope
from pympc.geometry.convex_hull import PolytopeProjectionInnerApproximation

class FeasibleSetLibrary:
    """
    library[switching_sequence]
    - program
    - feasible_set
    """

    def __init__(self, controller):
        self.controller = controller
        self.library = dict()
        self.read_gurobi_status = {
            '2': 'OPTIMAL',
            '3': 'INFEASIBLE',
            '4': 'INFEASIBLE OR UNBOUNDED',
            '9': 'TIME LIMIT',
            '11': 'INTERRUPTED',
            }
        return

    def sample_policy(self, n_samples, check_sample_function=None):
        n_rejected = 0
        n_included = 0
        n_new_ss = 0
        n_unfeasible = 0
        for i in range(n_samples):
            try:
                print('Sample ' + str(i) + ': ')
                x = self.random_sample(check_sample_function)
                if not self.sampling_rejection(x):
                    print('solving MIQP... '),
                    tic = time.time()
                    ss = self.controller.feedforward(x)[2]
                    print('time spent: ' + str(time.time()-tic) + ' s, model status: ' + self.read_gurobi_status[str(self.controller._model.status)] + '.')
                    if not any(np.isnan(ss)):
                        if self.library.has_key(ss):
                            n_included += 1
                            print('included.')
                            print('including sample in inner approximation... '),
                            tic = time.time()
                            self.library[ss]['feasible_set'].include_point(x)
                            print('sample included in ' + str(time.time()-tic) + ' s.')
                        else:
                            n_new_ss += 1
                            print('new switching sequence ' + str(ss) + '.')
                            self.library[ss] = dict()
                            print('condensing QP... '),
                            tic = time.time()
                            prog = self.controller.condense_program(ss)
                            print('QP condensed in ' + str(time.time()-tic) + ' s.')
                            self.library[ss]['program'] = prog
                            lhs = np.hstack((-prog.C_x, prog.C_u))
                            rhs = prog.C
                            residual_dimensions = range(prog.C_x.shape[1])
                            print('constructing inner simplex... '),
                            tic = time.time()
                            feasible_set = PolytopeProjectionInnerApproximation(lhs, rhs, residual_dimensions)
                            print('inner simplex constructed in ' + str(time.time()-tic) + ' s.')
                            print('including sample in inner approximation... '),
                            tic = time.time()
                            feasible_set.include_point(x)
                            print('sample included in ' + str(time.time()-tic) + ' s.')
                            self.library[ss]['feasible_set'] = feasible_set
                    else:
                        n_unfeasible += 1
                        print('unfeasible.')
                else:
                    n_rejected += 1
                    print('rejected.')
            except ValueError:
                print 'Something went wrong with this sample...'
                pass
        print('\nTotal number of samples: ' + str(n_samples) + ', switching sequences found: ' + str(n_new_ss) + ', included samples: ' + str(n_included) + ', rejected samples: ' + str(n_rejected) + ', unfeasible samples: ' + str(n_unfeasible) + '.')
        return

    def random_sample(self, check_sample_function=None):
        if check_sample_function is None:
            x = np.random.rand(self.controller.sys.n_x, 1)
            x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
        else:
            is_inside = False
            while not is_inside:
                x = np.random.rand(self.controller.sys.n_x,1)
                x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
                is_inside = check_sample_function(x)
        return x

    def sampling_rejection(self, x):
        for ss_value in self.library.values():
            if ss_value['feasible_set'].applies_to(x):
                return True
        return False

    def get_feasible_switching_sequences(self, x):
        return [ss for ss, ss_values in self.library.items() if ss_values['feasible_set'].applies_to(x)]

    def feedforward(self, x, given_ss=None, max_qp= None):
        V_list = []
        V_star = np.nan
        u_star = [np.full((self.controller.sys.n_u, 1), np.nan) for i in range(self.controller.N)]
        ss_star = [np.nan]*self.controller.N
        fss = self.get_feasible_switching_sequences(x)
        # print 'number of feasible QPs:', len(fss)
        if given_ss is not None:
            fss.insert(0, given_ss)
        if not fss:
            return u_star, V_star, ss_star, V_list
        else:
            if max_qp is not None and max_qp < len(fss):
                fss = fss[:max_qp]
                print 'number of QPs limited to', max_qp
            for ss in fss:
                u, V = self.library[ss]['program'].solve(x)
                V_list.append(V)
                if V < V_star or (np.isnan(V_star) and not np.isnan(V)):
                    V_star = V
                    u_star = [u[i*self.controller.sys.n_u:(i+1)*self.controller.sys.n_u,:] for i in range(self.controller.N)]
                    ss_star = ss
        return u_star, V_star, ss_star, V_list

    def feedback(self, x, given_ss=None, max_qp= None):
        u_star, V_star, ss_star = self.feedforward(x, given_ss, max_qp)[0:-1]
        return u_star[0], ss_star

    def add_shifted_switching_sequences(self, terminal_domain):
        for ss in self.library.keys():
            for shifted_ss in self.shift_switching_sequence(ss, terminal_domain):
                if not self.library.has_key(shifted_ss):
                    self.library[shifted_ss] = dict()
                    self.library[shifted_ss]['program'] = self.controller.condense_program(shifted_ss)
                    self.library[shifted_ss]['feasible_set'] = EmptyFeasibleSet()

    @staticmethod
    def shift_switching_sequence(ss, terminal_domain):
        return [ss[i:] + (terminal_domain,)*i for i in range(1,len(ss))]

    def plot_partition(self):
        for ss_value in self.library.values():
            color = np.random.rand(3,1)
            fs = ss_value['feasible_set']
            if not fs.empty:
                p = Polytope(fs.hull.A, fs.hull.b)
                p.assemble()#redundant=False, vertices=fs.hull.points)
                p.plot(facecolor=color, alpha=.5)
        return

    def save(self, name):
        self.controller._model = None
        self.controller._u_np = None
        self.controller._x_np = None
        self.controller._z = None
        self.controller._d = None
        np.save(name, self)
        self.controller._MIP_model()
        return

def load_library(name):
    library = np.load(name + '.npy').item()
    library.controller._MIP_model()
    return library

class EmptyFeasibleSet:

    def __init__(self):
        self.empty = True
        return

    def applies_to(self, x):
        return False