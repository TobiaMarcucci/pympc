import gurobipy as grb
import numpy as np

class BasisStandardForm:
    def __init__(self, ineq=None, x_pos=None,x_neg=None):
        self.ineq = ineq
        self.x_pos = x_pos
        self.x_neg = x_neg

def recursive_feasibility_check(A_2, b_2, A_1=None, b_1=None, basis_1=None, tol=1.e-6):
    """
    Solves the feasibility problem:
    exists (x_1, x_2) such that
    A_1 x_1 <= b_1             (1)
    A_2 [x_1' x_2']' <= b_2    (2)
    with basis_1 optimal active set of the subproblem (1).
    """

    # problem dimensions
    n_x = A_2.shape[1]
    n_x_1 = 0
    if A_1 is not None:
        n_x_1 = A_1.shape[1]
    
    # initialize model
    model = grb.Model()
    x_pos = model.addVars(n_x, name='x_pos')
    x_neg = model.addVars(n_x, name='x_neg')
    s = model.addVar(name='s')

    # first block of constraints
    if A_1 is not None:
        for i in range(A_1.shape[0]):
            expr = grb.LinExpr()
            for j in range(A_1.shape[1]):
                if np.abs(A_1[i,j]) > tol:
                    expr.add(A_1[i,j]*(x_pos[j] - x_neg[j]))
            if np.abs(b_1[i]) > tol:
                expr.add(-b_1[i])
            model.addConstr(expr <= 0., name='ineq_1_'+str(i))
        
    # second block of constraints
    for i in range(A_2.shape[0]):
        expr = grb.LinExpr()
        for j in range(A_2.shape[1]):
            if np.abs(A_2[i,j]) > tol:
                expr.add(A_2[i,j]*(x_pos[j] - x_neg[j]))
        if np.abs(b_2[i]) > tol:
            expr.add(-b_2[i])
        expr.add(-s)
        model.addConstr(expr <= 0., name='ineq_2_'+str(i))
        
    # cost function
    model.setObjective(grb.LinExpr(s))
    
    # warm start variables
    model.update()
    for i in range(n_x):
        if i < n_x_1:
            x_pos[i].setAttr('VBasis', basis_1.x_pos[i])
            x_neg[i].setAttr('VBasis', basis_1.x_neg[i])
        else:
            x_pos[i].setAttr('VBasis', -1)
            x_neg[i].setAttr('VBasis', -1)
    greatest_b = np.argmax(- b_2)
    if b_2[greatest_b,0] > 0:
        s.setAttr('VBasis', 0)
    else:
        s.setAttr('VBasis', -1)
        
    # warm start constraints
    if basis_1 is not None:
        for i in range(A_1.shape[0]):
            constr = model.getConstrByName('ineq_1_'+str(i))
            constr.setAttr('CBasis', basis_1.ineq[i])
    for i in range(A_2.shape[0]):
        constr = model.getConstrByName('ineq_2_'+str(i))
        constr.setAttr('CBasis', 0)
    if b_2[greatest_b,0] > 0:
        constr = model.getConstrByName('ineq_2_'+str(greatest_b))
        constr.setAttr('CBasis', -1)
    
    # run the optimization
    model.setParam('OutputFlag', 0)
    #model.setParam('Method', -1)
    model.optimize()
    
    # retrieve solution
    is_feasible = False
    basis = BasisStandardForm()
    if s.getAttr('x') < tol:
        is_feasible = True
        basis.ineq = []
        if A_1 is not None:
            for i in range(A_1.shape[0]):
                constr = model.getConstrByName('ineq_1_'+str(i))
                basis.ineq.append(constr.getAttr('CBasis'))
        for i in range(A_2.shape[0]):
            constr = model.getConstrByName('ineq_2_'+str(i))
            basis.ineq.append(constr.getAttr('CBasis'))
        basis.x_pos = []
        basis.x_neg = []
        for i in range(n_x):
            basis.x_pos.append(x_pos[i].getAttr('VBasis'))
            basis.x_neg.append(x_neg[i].getAttr('VBasis'))
    
    return is_feasible, basis, model.Runtime