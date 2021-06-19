import numpy as np
import tqdm
from sklearn.decomposition._nmf import _initialize_nmf as init_nmf # for initing the nmf decomposition

class NMF:
    def __init__(self,n_components,tol_rat = 1e-5, max_iter = 5000):
        self.n_components = n_components
        self.tol_rat = tol_rat
        self.max_iter = max_iter
        self.B = None
        self.G = None

    def fit_transform(self,X, W, Delta, beta = 0.325, alpha = 0.15):
        # in this case, X is Feature * Number
        self.B, self.G = init_nmf(X.transpose(), self.n_components, init = 'random') # B is N * d, G is D * n B (feature * new_feature) G (new_feature, number)
        self.G = self.G.transpose() # B is base (F, D), and G is the coefficient matrix, (N, D)
        # S = B * G'
        for _ in range(self.max_iter):
            self.B = self.B * (X @ self.G + 2 * beta*self.B) / (self.B @ self.G.transpose() @ self.G + 2*beta*self.B @ self.B.transpose() @ self.B)
            self.G = self.G * (X.transpose() @ self.B + alpha * W @ self.G) / (self.G @ self.B.transpose() @self.B + alpha * Delta @ self.G)
            X_est = self.B @ self.G.transpose()
            # judge the reconstruction error
            error = np.linalg.norm((X - X_est))
            if error < self.tol_rat:
                print('The error is below the tol_rat')
                break
        print(f'The final return error is {error}')
        return self.B, self.G

    # def transform(self, Y):
        

    #     pass