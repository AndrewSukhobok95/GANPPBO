import numpy as np
from scipy.stats import norm, beta
from matplotlib import pyplot as plt
from GPyOpt.methods import BayesianOptimization #Use as global optimizer
    
class Prior:
    
    def __init__(self,family,valrange,lambda_indices,fixed_hyperparams=None,is_location_param=True):
        
        self.family=family #The distribution famility of the prior
        self.valrange = valrange
        if family=='normal':
            l = abs(valrange[1]-valrange[0])
            self.domain=(valrange[0]-1.5*l,valrange[1]+1.5*l)
        if family=='beta':
            epsilon = 1e-5
            self.domain=(0+epsilon,1-epsilon)
        
        self.lambda_indices = lambda_indices #The indices in the lambda vecter which refer to the hyperparameters of the prior
        self.fixed_hyperparams = fixed_hyperparams #Other hyperparameters that are fixed to some value
        if len(lambda_indices) == 1:
            self.is_location_param=is_location_param #If there is only one hyperparameter is it location or scale parameter?
        else:
            self.is_location_param=None
        
        
    ''' Function gives the probability density of the prior '''
    def pdf(self,theta,lambda_):
        if self.family=='normal':
            if len(self.lambda_indices)==1:
                if self.is_location_param:
                    return norm.pdf(theta, lambda_[self.lambda_indices[0]], self.fixed_hyperparams[0])
                else:
                    return norm.pdf(theta, self.fixed_hyperparams[0], lambda_[self.lambda_indices[0]])
            else:
                return norm.pdf(theta, lambda_[self.lambda_indices[0]], lambda_[self.lambda_indices[1]])
        if self.family=='beta':
            if len(self.lambda_indices)==1:
                if self.is_location_param:
                    return beta.pdf(theta, lambda_[self.lambda_indices[0]], self.fixed_hyperparams[0])
                else:
                    return beta.pdf(theta, self.fixed_hyperparams[0], lambda_[self.lambda_indices[0]])
            else:
                return beta.pdf(theta, lambda_[self.lambda_indices[0]], lambda_[self.lambda_indices[1]])      
        else:
            print('The given distribution family is not supported!')
            raise NotImplementedError
        
        
def g_theta(theta,Prior,lambda_sample):
    mc_sample = np.empty(len(lambda_sample))
    for i,lambda_ in enumerate(lambda_sample):
        mc_sample[i] = Prior.pdf(theta,lambda_)
    return np.mean(mc_sample)

def KL_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def minimize_KL(q,g,prior):
    thetas = np.arange(prior.domain[0], prior.domain[1], 0.01)
    if prior.family=='normal':
        bounds = [{'name': 'location', 'type': 'continuous', 'domain': (prior.domain[0], prior.domain[1])}]
    if prior.family=='beta':
        bounds = [{'name': 'location', 'type': 'continuous', 'domain': (0.01, 10)}]
    BO = BayesianOptimization(lambda hyperparam: KL_divergence(q(thetas, hyperparam, prior.fixed_hyperparams[0]),g),
                              domain=bounds,optimize_restarts = 1,normalize_Y=True)
    BO.run_optimization(max_iter = 100)
    opt_hyperparam = BO.x_opt[0]
    return opt_hyperparam
    

def sample_h(GP_model,sample_size,ignore_noise_factor=1,n_input_points=1000):
    sample = np.empty((sample_size,GP_model.D))
    print('Progress:')
    for i in range(0,sample_size):
        if (i+1) % 10 == 0:
            print(str(i+1) + '/' + str(sample_size))
        input_points = np.random.random((n_input_points, GP_model.D)) #Unif([0.0, 1.0))
        unscaled_input_points = GP_model.FP.unscale(input_points)
        mu,Sigma= GP_model.mu_Sigma_pred(input_points)
        mu = ignore_noise_factor*mu #i.e. scale up posterior_mean to decrease relative randomness in GP
        f_values = list(np.random.multivariate_normal(mu,Sigma)) #predict/sample GP
        argmax = unscaled_input_points[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)


#start = time.time()
#lambda_sample = sample_h(GP_model,100)
#print(time.time()-start)


def plot(q,g,thetas,fig_id,legends=False,title=False):
    plt.figure(fig_id)
    plt.plot(thetas, q, c='blue')
    plt.plot(thetas, g, c='red')
    if title:
        plt.title('KL($\~g||g$) = %1.3f' % KL_divergence(q, g), fontsize=20)
    if legends:
        plt.legend(['KL-optimal parametric prior $\~g$','Non-parametric prior $g = \int g(\u00B7|\u03BB)h(\u03BB)d\u03BB$'], fontsize=15)


#thetas = np.arange(prior0.domain[0], prior0.domain[1], 0.01)
#g = [g_theta(theta,prior0,lambda_sample) for theta in thetas]
#q = beta.pdf
#opt_hyperparam = minimize_KL(q,g,prior0)
#plot(q(thetas, opt_hyperparam, prior0.fixed_hyperparams[0]),g)
    
    
    
    
    
    
    