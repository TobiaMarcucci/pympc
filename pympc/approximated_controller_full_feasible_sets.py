import time
import numpy as np
from geometry.polytope import Polytope
from pympc.geometry.convex_hull import orthogonal_projection_CHM
from pympc.control import upload_HybridModelPredictiveController
from pympc.geometry.inner_approximation_polytope_projection import InnerApproximationOfPolytopeProjection, upload_InnerApproximationOfPolytopeProjection
from optimization.gurobi import read_status
from optimization.parametric_programs import upload_ParametricQP
import h5py
import ast
from pympc.mode_sequence_tree_qp import ModeSequenceTree, Solution

class PolicySampler:
    
    def __init__(self, controller, qp_library=None):
        self.controller = controller
        if qp_library is None:
            self.qp_library = dict()
        else:
            self.qp_library = qp_library
        return

    def sample_policy(self, n_samples, check_sample_function=None):

        # initialize count
        n_rejected = 0
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
                if any(np.isnan(mode_sequence)):
                    n_unfeasible += 1
                    print('problem unfeasible.')

                # add sample to the library
                else:

                    # condense qp
                    prog = self.controller.condense_program(mode_sequence)
                    self.qp_library[mode_sequence] = prog
                    n_new_sequences += 1
                    print('QP condensed.')
                        
        # report results of the sampling
        print('\nTotal number of samples: ' + str(n_samples) + ', switching sequences found: ' + str(n_new_sequences) + ', rejected samples: ' + str(n_rejected) + ', unfeasible samples: ' + str(n_unfeasible) + '.')

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
        for qp in self.qp_library.values():
            if qp.is_feasible(x):
                return True
        return False

    def get_feasible_mode_sequences(self, x):
        feasible_mode_sequences = []
        for mode_sequence, qp in self.qp_library.items():
            if qp.is_feasible(x):
                feasible_mode_sequences.append(mode_sequence)
        return feasible_mode_sequences

    def plot_partition(self, **kwargs):
        for qp in self.qp_library.values():
            A = np.hstack((-qp.C_x, qp.C_u))
            b = qp.C
            residual_dimensions = range(qp.C_x.shape[1])
            A_projection, b_projection, vertices_projection = orthogonal_projection_CHM(A, b, residual_dimensions)
            projection = Polytope(A_projection, b_projection).assemble(vertices=vertices_projection)
            if not projection.empty:
                color = np.random.rand(3,1).flatten()
                projection.plot(facecolor=color, alpha=.5, **kwargs)
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
        qp_library = group.create_group('qp_library')
        for mode_sequence, qp in self.qp_library.items():
            mode_sequence = qp_library.create_group(str(mode_sequence))
            mode_sequence = qp.save('program', mode_sequence)

        # close the file and return
        if super_group is None:
            group.close()
            return
        else:
            return super_group

def upload_PolicySampler(group_name, super_group=None):
    """
    Reads the file group_name.hdf5 and generates a FeasibleSetLibrary from the data therein.
    If a super_group is provided, reads the sub group named group_name which belongs to the super_group.
    """

    # open the file
    if super_group is None:
        sampler = h5py.File(group_name + '.hdf5', 'r')
    else:
        sampler = super_group[group_name]

    # read controller
    controller = upload_HybridModelPredictiveController('controller', sampler)

    # read mode sequences
    qp_library = dict()
    for mode_sequence in sampler['qp_library']:
        mode_sequence_tuple = ast.literal_eval(mode_sequence)
        qp_library[mode_sequence_tuple] = upload_ParametricQP('program', sampler['qp_library'][str(mode_sequence)])

    # close the file and return
    if super_group is None:
        sampler.close()
    return PolicySampler(controller, qp_library)



# class ApproximatedHybridModelPredictiveController:

#     def __init__(self, sampler, terminal_mode):
#         self.sampler = sampler
#         self.shifted_qps = self._programs_shifted_mode_sequences(terminal_mode)
#         return

#     def _programs_shifted_mode_sequences(self, terminal_mode):
#         shifted_qps = dict()
#         for mode_sequence in self.sampler.qp_library.keys():
#             for shifted_mode_sequence in shift_mode_sequence(mode_sequence, terminal_mode):
#                 if not shifted_qps.has_key(shifted_mode_sequence):
#                     shifted_qps[shifted_mode_sequence] = self.sampler.controller.condense_program(shifted_mode_sequence)
#         return shifted_qps

#     def feedforward(self, x, previous_mode_sequence=None, max_programs=None):

#         # initilize output
#         n_u = self.sampler.controller.sys.n_u
#         N = self.sampler.controller.N
#         if previous_mode_sequence is not None:
#             input_sequence, cost = self.shifted_qps[previous_mode_sequence].solve(x)
#             input_sequence = [input_sequence[i*n_u:(i+1)*n_u,:] for i in range(N)]
#             mode_sequence = previous_mode_sequence
#         else:
#             cost = np.nan
#             mode_sequence = [np.nan] * N
#             input_sequence = [np.full((n_u, 1), np.nan)] * N

#         # get and cut feasible qps
#         feasible_mode_sequences = self.sampler.get_feasible_mode_sequences(x)
#         if max_programs is not None:
#             feasible_mode_sequences = feasible_mode_sequences[:max_programs-1]

#         # solve qps
#         for mode_sequence in feasible_mode_sequences:
#             new_input_sequence, new_cost = self.sampler.qp_library[mode_sequence].solve(x)
#             if new_cost < cost or np.isnan(cost):
#                 cost = new_cost
#                 input_sequence = [new_input_sequence[i*n_u:(i+1)*n_u,:] for i in range(N)]
#                 mode_sequence = mode_sequence

#         return input_sequence, cost, mode_sequence

#     def feedback(self, x, first_mode_sequence=None, max_programs=None):
#         input_sequence, cost, mode_sequence = self.feedforward(x, first_mode_sequence, max_programs)
#         return input_sequence[0], mode_sequence



class ApproximatedHybridModelPredictiveController:

    def __init__(self, sampler, terminal_mode):
        self.sampler = sampler
        self.tree = ModeSequenceTree()
        self.tree.expand(sampler.controller, sampler.qp_library)
        self.shifted_qps = self._programs_shifted_mode_sequences(terminal_mode)
        return

    def _programs_shifted_mode_sequences(self, terminal_mode):
        shifted_qps = dict()
        for mode_sequence in self.sampler.qp_library.keys():
            for shifted_mode_sequence in shift_mode_sequence(mode_sequence, terminal_mode):
                if not shifted_qps.has_key(shifted_mode_sequence):
                    shifted_qps[shifted_mode_sequence] = self.sampler.controller.condense_program(shifted_mode_sequence)
        return shifted_qps

    def feedforward(self, x, previous_mode_sequence=None):
        n_u = self.sampler.controller.sys.n_u
        N = self.sampler.controller.N
        if previous_mode_sequence is not None:
            argmin, cost = self.shifted_qps[previous_mode_sequence].solve(x)
            candidate_solution = Solution(
                min = cost,
                argmin = argmin,
                mode_sequence = previous_mode_sequence
                )
        else:
            candidate_solution = None
        solution = self.tree.solve(x, candidate_solution)
        if solution is None:
            cost = np.nan
            input_sequence = np.full((n_u * N, 1), np.nan)
            mode_sequence = (np.nan,) * N
        else:
            input_sequence = [solution.argmin[i*n_u:(i+1)*n_u,:] for i in range(N)]
            cost = solution.min
            mode_sequence = solution.mode_sequence
        return input_sequence, cost, mode_sequence

    def feedback(self, x, previous_mode_sequence=None):
        input_sequence, _, mode_sequence = self.feedforward(x, previous_mode_sequence)
        return input_sequence[0], mode_sequence

def shift_mode_sequence(mode_sequence, terminal_mode):
    shifted_sequences = [mode_sequence[i:] + (terminal_mode,)*i for i in range(1,len(mode_sequence))]
    shifted_sequences = list(set(shifted_sequences))
    return shifted_sequences