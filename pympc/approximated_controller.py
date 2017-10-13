import time
import numpy as np
from geometry.polytope import Polytope
from pympc.control import upload_HybridModelPredictiveController
from pympc.geometry.inner_approximation_polytope_projection import InnerApproximationOfPolytopeProjection, upload_InnerApproximationOfPolytopeProjection
from optimization.gurobi import read_status
from optimization.parametric_programs import upload_ParametricQP
import h5py
import ast

class FeasibleSetLibrary:
    
    def __init__(self, controller, mode_sequences=None):
        self.controller = controller
        if mode_sequences is None:
            self.mode_sequences = dict()
        else:
            self.mode_sequences = mode_sequences
        return

    def sample_policy(self, n_samples, check_sample_function=None):

        # initialize count
        n_rejected = 0
        n_included = 0
        n_new_sequences = 0
        n_unfeasible = 0

        # build library
        for i in range(n_samples):
            print('Sample ' + str(i) + ': ')
            x = self._random_sample(check_sample_function)

            # reject sample if already covered
            if self._sampling_rejection(x):
                n_rejected += 1
                print('rejected.')

            # solve for the optimal mode sequence
            else:
                print('solving MIQP... '),
                tic = time.time()
                mode_sequence = self.controller.feedforward(x)[2]
                solver_status = read_status(str(self.controller._model.status))
                print('time spent: ' + str(time.time()-tic) + ' s, model status: ' + solver_status + '.')

                # discard sample if unfeasible
                if not solver_status in ['OPTIMAL', 'TIME LIMIT', 'SUBOPTIMAL']:
                    n_unfeasible += 1

                # add sample to the library
                else:

                    # include sample if mode sequence is in library
                    if self.mode_sequences.has_key(mode_sequence):
                        n_included += 1
                        print('included.\nincluding sample in inner approximation... '),
                        tic = time.time()
                        self.mode_sequences[mode_sequence]['feasible_set'].include_point(x)
                        print('sample included in ' + str(time.time()-tic) + ' s.')

                    # add mode sequence if it is not in the library
                    else:
                        n_new_sequences += 1
                        print('new mode sequence ' + str(mode_sequence) + '.')
                        self.mode_sequences[mode_sequence] = dict()

                        # condense qp
                        print('condensing QP... '),
                        tic = time.time()
                        prog = self.controller.condense_program(mode_sequence)
                        print('QP condensed in ' + str(time.time()-tic) + ' s.')
                        self.mode_sequences[mode_sequence]['program'] = prog

                        # generate inner approximation of feasible set
                        lhs = np.hstack((-prog.C_x, prog.C_u))
                        rhs = prog.C
                        residual_dimensions = range(prog.C_x.shape[1])
                        print('building inner approximation... '),
                        tic = time.time()
                        feasible_set = InnerApproximationOfPolytopeProjection(lhs, rhs, residual_dimensions)
                        feasible_set.include_point(x)
                        print('approximation built ' + str(time.time()-tic) + ' s.')
                        self.mode_sequences[mode_sequence]['feasible_set'] = feasible_set
                        
        # report results of the sampling
        print('\nTotal number of samples: ' + str(n_samples) + ', switching sequences found: ' + str(n_new_sequences) + ', included samples: ' + str(n_included) + ', rejected samples: ' + str(n_rejected) + ', unfeasible samples: ' + str(n_unfeasible) + '.')

        return

    def _random_sample(self, check_sample_function=None):
        x = np.random.rand(self.controller.sys.n_x, 1)
        x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
        if check_sample_function is not None:
            while not check_sample_function(x):
                x = np.random.rand(self.controller.sys.n_x, 1)
                x = np.multiply(x, (self.controller.sys.x_max - self.controller.sys.x_min)) + self.controller.sys.x_min
        return x

    def _sampling_rejection(self, x):
        for mode_sequence in self.mode_sequences.values():
            if mode_sequence['feasible_set'].applies_to(x):
                return True
        return False

    def get_feasible_mode_sequences(self, x):
        feasible_mode_sequences = []
        for mode_sequence_key, mode_sequence_value in self.mode_sequences.items():
            if mode_sequence_value['feasible_set'].applies_to(x):
                feasible_mode_sequences.append(mode_sequence_key)
        return feasible_mode_sequences

    def store_halfspaces(self):
        for mode_sequence in self.mode_sequences.values():
            mode_sequence['feasible_set'].store_halfspaces()
        return

    def plot_partition(self, **kwargs):
        for mode_sequence in self.mode_sequences.values():
            p = Polytope(
                mode_sequence['feasible_set'].A_ld,
                mode_sequence['feasible_set'].b_ld)
            p.assemble()
            if not p.empty:
                color = np.random.rand(3,1).flatten()
                p.plot(facecolor=color, alpha=.5, **kwargs)
        return

    def save(self, group_name, super_group=None):

        # open the file
        if super_group is None:
            group = h5py.File(group_name + '.hdf5', 'w')
        else:
            group = super_group.create_group(group_name)

        # write controller
        group = self.controller.save('controller', group)

        # write mode sequences
        mode_sequences = group.create_group('mode_sequences')
        for mode_sequence_key, mode_sequence_value in self.mode_sequences.items():
            mode_sequence = mode_sequences.create_group(str(mode_sequence_key))
            mode_sequence = mode_sequence_value['program'].save('program', mode_sequence)
            mode_sequence = mode_sequence_value['feasible_set'].save('feasible_set', mode_sequence)

        # close the file and return
        if super_group is None:
            group.close()
            return
        else:
            return super_group

def upload_FeasibleSetLibrary(group_name, super_group=None):
    """
    Reads the file group_name.hdf5 and generates a FeasibleSetLibrary from the data therein.
    If a super_group is provided, reads the sub group named group_name which belongs to the super_group.
    """

    # open the file
    if super_group is None:
        library = h5py.File(group_name + '.hdf5', 'r')
    else:
        library = super_group[group_name]

    # read controller
    controller = upload_HybridModelPredictiveController('controller', library)

    # read mode sequences
    mode_sequences = dict()
    for mode_sequence in library['mode_sequences']:
        mode_sequence_tuple = ast.literal_eval(mode_sequence)
        mode_sequences[mode_sequence_tuple] = dict()
        mode_sequences[mode_sequence_tuple]['program'] = upload_ParametricQP('program', library['mode_sequences'][str(mode_sequence)])
        mode_sequences[mode_sequence_tuple]['feasible_set'] = upload_InnerApproximationOfPolytopeProjection('feasible_set', library['mode_sequences'][str(mode_sequence)])

    # close the file and return
    if super_group is None:
        library.close()
    return FeasibleSetLibrary(controller, mode_sequences)



class ApproximatedHybridModelPredictiveController:

    def __init__(self, library, terminal_mode):
        self.library = library
        self.shifted_qps = self._programs_shifted_mode_sequences(terminal_mode)
        return

    def _programs_shifted_mode_sequences(self, terminal_mode):
        shifted_qps = dict()
        mode_sequences = self.library.mode_sequences
        for mode_sequence in self.library.mode_sequences.keys():
            for shifted_mode_sequence in shift_mode_sequence(mode_sequence, terminal_mode):
                if not shifted_qps.has_key(shifted_mode_sequence):
                    shifted_qps[shifted_mode_sequence] = self.library.controller.condense_program(shifted_mode_sequence)
        return shifted_qps

    def feedforward(self, x, first_mode_sequence=None, max_programs=None):

        # initilize output
        n_u = self.library.controller.sys.n_u
        N = self.library.controller.N
        if first_mode_sequence is not None:
            input_sequence, cost = self.shifted_qps[first_mode_sequence].solve(x)
            input_sequence = [input_sequence[i*n_u:(i+1)*n_u,:] for i in range(N)]
            mode_sequence = first_mode_sequence
        else:
            cost = np.nan
            mode_sequence = [np.nan] * N
            input_sequence = [np.full((n_u, 1), np.nan)] * N

        # get and cut feasible qps
        feasible_mode_sequences = self.library.get_feasible_mode_sequences(x)
        if max_programs is not None:
            feasible_mode_sequences = feasible_mode_sequences[:max_programs-1]

        # solve qps
        for feasible_mode_sequence in feasible_mode_sequences:
            new_input_sequence, new_cost = self.library.mode_sequences[feasible_mode_sequence]['program'].solve(x)
            if new_cost < cost or np.isnan(cost):
                cost = new_cost
                input_sequence = [new_input_sequence[i*n_u:(i+1)*n_u,:] for i in range(N)]
                mode_sequence = feasible_mode_sequence

        return input_sequence, cost, mode_sequence

    def feedback(self, x, first_mode_sequence=None, max_programs=None):
        input_sequence, cost, mode_sequence = self.feedforward(x, first_mode_sequence, max_programs)
        return input_sequence[0], mode_sequence

def shift_mode_sequence(mode_sequence, terminal_mode):
    return [mode_sequence[i:] + (terminal_mode,)*i for i in range(1,len(mode_sequence))]

def load_library(name):
    library = np.load(name + '.npy').item()
    library.controller._MIP_model()
    return library