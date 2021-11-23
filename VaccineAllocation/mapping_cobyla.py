import numpy as np
import gurobipy as gb
from gurobipy import GRB
    
class quadratic_program:
    '''
    Linear quadratic program that maps the vaccine time windows of cobyla to vaccine allocations.
    Decision variables:
         X: the number of doses allocated to age-risk group (a,r) in week t.
         N_u: the number of people from age-risk group (a,r) who are not vaccinated yet in time t.
    '''
    def __init__(self, instance, vaccines, is_qp = True):
        self.is_qp = is_qp
        self.population = instance.N
        self.T = len(vaccines.vaccine_time)
        self.A, self.L = instance.A, instance.L
        self.type = np.amax(vaccines.vaccine_type)
        self.V = [[v*np.sum(instance.N) for v in prop] for prop in vaccines.vaccine_proportion]

        
    def solve_QP(self, time_windows):
        A, L, T, population, V = self.A, self.L, self.T, self.population, self.V
        len_type = [len(v) for v in V] # number of different type of vaccine supply in time t.
        
        model = gb.Model("quadratic_model")
        X = []*T
        for t in range(T):
            X.append(model.addVars(len_type[t], A, L, lb = 0.0, ub = GRB.INFINITY, name = "X"+str(t+1)))
             
        N_u = model.addVars(T, A, L, lb = 0.0, ub = GRB.INFINITY, name = "N") 
        # Objective function:
        #if self.is_qp == 'True':
        model.setObjective(sum((N_u[t,a,r]/population[a][r])*(N_u[t,a,r]/population[a][r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)
        #else:
            #model.setObjective(sum((N_u[t,a,r]/population[a][r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)
              
        #Constraints:
        model.addConstrs((sum(X[t][v,a,r] for a in range(A) for r in range(L)) <= V[t][v] for t in range(T) for v in range(len_type[t])), "c1")

        model.addConstrs((N_u[t,a,r] >= population[a][r] - sum(X[t_prime][v,a,r]  for t_prime in set(range(time_windows[a][r],T)).intersection(range(0,t+1)) for v in range(len_type[t_prime])) 
                              for a in range(A) for r in range(L) for t in range(T)), "c2")
        
        model.addConstrs((sum(X[t][v,a,r] for t in range(T) for v in range(len_type[t])) <= population[a][r] 
                          for a in range(A) for r in range(L)), "c3")
        #No need to but just in case:
        # for a in range(A):
        #     for r in range(L):
        #         print(a,r)
        #         if time_windows[a][r] != 0:
        #             t_end = min(time_windows[a][r],T)
        #             model.addConstrs((X[t,a,r] == 0 for t in range(0,t_end)), "c_null")
        
        model.setParam(GRB.Param.OutputFlag,0) 
        model.optimize()
        model.write("quadratic_model.lp")
        #print('obj %f' % model.objVal)
        vaccine_proportions = []
        vaccine_allocations = []

        for t in range(T):
            proportion_temp = []
            allocation_temp = []  
            for v in range(len_type[t]):
                p_temp = np.zeros((A,L))
                a_temp = np.zeros((A,L))          
                for a in range(A):
                    for r in range(L):
                        p_temp[a][r] = X[t][v,a,r].x/V[t][v]
                        a_temp[a][r] = X[t][v,a,r].x
                proportion_temp.append(p_temp)
                allocation_temp.append(a_temp)
            vaccine_proportions.append(proportion_temp)
            vaccine_allocations.append(allocation_temp)
            
        return vaccine_proportions, vaccine_allocations
    
    def solve_QP_ub_lb(self, time_windows_lb, time_windows_ub):
        A, L, T, population, V = self.A, self.L, self.T, self.population, self.V
        len_type = [len(v) for v in V] # number of different type of vaccine supply in time t.
        
        epsilon = 1/sum(sum(V, []))
        model = gb.Model("quadratic_model")
        X = []*T
        for t in range(T):
            X.append(model.addVars(len_type[t], A, L, lb = 0.0, ub = GRB.INFINITY, name = "X"+str(t+1)))
             
        N_u = model.addVars(T, A, L, lb = 0.0, ub = GRB.INFINITY, name = "N") 
        # Objective function:
        if self.is_qp == 1:
            #5a
            model.setObjective(sum((N_u[t,a,r]/population[a][r])*(N_u[t,a,r]/population[a][r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)
        if self.is_qp == 2:
            #5b
            model.setObjective(sum((1/population[a][r])*(N_u[t,a,r]*N_u[t,a,r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)
        if self.is_qp == 3:
            #5c
            model.setObjective(sum((1/population[a][r])*(N_u[T-1,a,r]*N_u[T-1,a,r]) for a in range(A) for r in range(L)) + 
                               epsilon * sum((1/population[a][r]) * sum(X[t][v,a,r]*X[t][v,a,r] for v in range(len_type[t])) for t in range(T) for a in range(A) for r in range(L)), GRB.MINIMIZE)
        if self.is_qp == 4:    
            #linear for comparison purpose
            model.setObjective(sum((N_u[t,a,r]/population[a][r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)

        #Constraints:
        model.addConstrs((sum(X[t][v,a,r] for a in range(A) for r in range(L)) <= V[t][v] for t in range(T) for v in range(len_type[t])), "c1")

        model.addConstrs((N_u[t,a,r] >= population[a][r] - sum(X[t_prime][v,a,r] for t_prime in set(range(time_windows_lb[a][r], time_windows_ub[a][r] + 1)).intersection(range(0, t+1)) for v in range(len_type[t_prime])) 
                              for a in range(A) for r in range(L) for t in range(T)), "c2")
        
        model.addConstrs((sum(X[t][v,a,r] for t in range(T) for v in range(len_type[t])) <= population[a][r] 
                          for a in range(A) for r in range(L)), "c3")
        #No need to but just in case:
        # for a in range(A):
        #     for r in range(L):
        #         print(a,r)
        #         if time_windows[a][r] != 0:
        #             t_end = min(time_windows[a][r],T)
        #             model.addConstrs((X[t,a,r] == 0 for t in range(0,t_end)), "c_null")
        
        model.setParam(GRB.Param.OutputFlag,0) 
        model.optimize()
        model.write("quadratic_model.lp")
        print('obj %f' % model.objVal)
        vaccine_proportions = []
        vaccine_allocations = []

        for t in range(T):
            proportion_temp = []
            allocation_temp = []  
            for v in range(len_type[t]):
                p_temp = np.zeros((A,L))
                a_temp = np.zeros((A,L))          
                for a in range(A):
                    for r in range(L):
                        p_temp[a][r] = X[t][v,a,r].x/V[t][v]
                        a_temp[a][r] = X[t][v,a,r].x
                proportion_temp.append(p_temp)
                allocation_temp.append(a_temp)
            vaccine_proportions.append(proportion_temp)
            vaccine_allocations.append(allocation_temp)
            
        return vaccine_proportions, vaccine_allocations
    
    def solve_QP_equality(self, time_windows_lb, time_windows_ub):
        A, L, T, population, V = self.A, self.L, self.T, self.population, self.V
        len_type = [len(v) for v in V] # number of different type of vaccine supply in time t.
        
        epsilon = 1/np.sum(V)
        model = gb.Model("quadratic_model")
        X = []*T
        for t in range(T):
            X.append(model.addVars(len_type[t], A, L, lb = 0.0, ub = GRB.INFINITY, name = "X"+str(t+1)))
             
        N_u = model.addVars(T, A, L, lb = 0.0, ub = GRB.INFINITY, name = "N") 
        # Objective function:
        if self.is_qp == 1:
            #5a
            model.setObjective(sum((N_u[t,a,r]/population[a][r])*(N_u[t,a,r]/population[a][r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)
        if self.is_qp == 2:
            #5b
            model.setObjective(sum((1/population[a][r])*(N_u[t,a,r]*N_u[t,a,r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)
        if self.is_qp == 3:
            #5c
            model.setObjective(sum((1/population[a][r])*(N_u[T-1,a,r]*N_u[T-1,a,r]) for a in range(A) for r in range(L)) + 
                               epsilon * sum((1/population[a][r]) * sum(X[t][v,a,r]*X[t][v,a,r] for v in range(len_type[t])) for t in range(T) for a in range(A) for r in range(L)), GRB.MINIMIZE)
        if self.is_qp == 4:    
            #linear for comparison purpose
            model.setObjective(sum((N_u[t,a,r]/population[a][r]) for a in range(A) for r in range(L) for t in range(T)), GRB.MINIMIZE)
    
        #Constraints:
        model.addConstrs((sum(X[t][v,a,r] for a in range(A) for r in range(L)) <= V[t][v] for t in range(T) for v in range(len_type[t])), "c1_le")
        model.addConstrs((sum(X[t][v,a,r] for a in range(A) for r in range(L)) >= V[t][v] for t in range(T) for v in range(len_type[t])), "c1_ge")
        
        model.addConstrs((N_u[t,a,r] >= population[a][r] - sum(X[t_prime][v,a,r] for t_prime in set(range(time_windows_lb[a][r],time_windows_ub[a][r] + 1)).intersection(range(0,t+1)) for v in range(len_type[t_prime])) 
                              for a in range(A) for r in range(L) for t in range(T)), "c2")
        
        model.addConstrs((sum(X[t][v,a,r] for t in range(T) for v in range(len_type[t])) <= population[a][r] 
                          for a in range(A) for r in range(L)), "c3")
        #No need to but just in case:
        for a in range(A):
            for r in range(L):
                #print(a,r)
                if time_windows_lb[a][r] > 0:
                    model.addConstrs((X[t][v,a,r] == 0 for t in range(0,time_windows_lb[a][r]) for v in range(len_type[t])), "c_null_1")
        for a in range(A):
            for r in range(L):
                #print(a,r)
                if time_windows_ub[a][r] != T:
                    model.addConstrs((X[t][v,a,r] == 0 for t in range(time_windows_ub[a][r]+1,T) for v in range(len_type[t])), "c_null_2")
        
        model.setParam(GRB.Param.OutputFlag,0) 
        model.optimize()
        model.write("quadratic_model.lp")
        vaccine_proportions = []
        vaccine_allocations = []
        
        if model.status == GRB.OPTIMAL:
            print('obj %f' % model.objVal)
 
            for t in range(T):
                proportion_temp = []
                allocation_temp = []  
                for v in range(len_type[t]):
                    p_temp = np.zeros((A,L))
                    a_temp = np.zeros((A,L))          
                    for a in range(A):
                        for r in range(L):
                            p_temp[a][r] = X[t][v,a,r].x/V[t][v]
                            a_temp[a][r] = X[t][v,a,r].x
                    proportion_temp.append(p_temp)
                    allocation_temp.append(a_temp)
                vaccine_proportions.append(proportion_temp)
                vaccine_allocations.append(allocation_temp)
            
        return vaccine_proportions, vaccine_allocations