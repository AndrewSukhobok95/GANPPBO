import numpy as np
import scipy
import scipy.stats
import time

GP_model = GP_model[0][1]

#Plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

''' Plotting '''
slice_1_dim = 1
slice_2_dim = 2


''' f_MAP '''
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
ax.scatter3D(GP_model.X[:,slice_1_dim-1], GP_model.X[:,slice_2_dim-1], GP_model.f_MAP, c=GP_model.f_MAP, cmap='hsv');
plt.show()

''' Posterior Mode location'''
x_star_, mu_star_  = GP_model.mu_star()
x_star_ = GP_model.FP.unscale(x_star_)
print(x_star_)







''' Generate Argmin-distribution '''

#############################################
#Set ignore_noise_factor=1 by default. If dim(X) is high then consider something > 1.
def generate_argmin_posterior(GP_model,sample_size,ignore_noise_factor=1):
    n_input_points = 1000
    sample = np.empty((sample_size,GP_model.D))
    for i in range(0,sample_size):
        print(i)
        input_points = np.random.random((n_input_points, GP_model.D)) #Unif([0.0, 1.0))
        unscaled_input_points = GP_model.FP.unscale(input_points)
        mu,Sigma= GP_model.mu_Sigma_pred(input_points)
        mu = ignore_noise_factor*mu #i.e. scale up posterior_mean to decrease relative randomness in GP
        #print(mu)
        #print(np.diag(Sigma))
        f_values = list(np.random.multivariate_normal(mu,Sigma)) #predict/sample GP
        argmax = unscaled_input_points[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)
start = time.time()
histogram = generate_argmin_posterior(GP_model,100)
print(time.time()-start)
#####################################




############## Plotting ##################################
mean_ = [np.mean(histogram[:,slice_1_dim-1]),np.mean(histogram[:,slice_2_dim-1])]
mode_ = [scipy.stats.mode(histogram[:,slice_1_dim-1])[0],scipy.stats.mode(histogram[:,slice_2_dim-1])[0]]
plt.plot(histogram[:,slice_1_dim-1],histogram[:,slice_2_dim-1],'ro',alpha=0.1)
plt.axis([GP_model.original_bounds[0][0],GP_model.original_bounds[0][1],GP_model.original_bounds[0][0],GP_model.original_bounds[0][1]])
plt.scatter(mean_[0],mean_[1], s=50,alpha=1,marker="x",color="blue")
plt.scatter(mode_[0],mode_[1], s=50,alpha=1,marker="x",color="purple")
plt.legend(["posterior draw","posterior mean","posterior mode"])
plt.xlabel("Variable $x_1$")
plt.ylabel("Variable $x_2$")
plt.show()
################################################################

############################ CONTOUR PLOTS ###########################
def plot_density(histogram,var1,var2,GP_model=GP_model):
    x = histogram[:, var1-1]
    y = histogram[:, var2-1]
    xmin = GP_model.original_bounds[var1-1][0]
    xmax = GP_model.original_bounds[var1-1][1]
    ymin = GP_model.original_bounds[var2-1][0]
    ymax = GP_model.original_bounds[var2-1][1]
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel("Variable " + str(var1))
    ax.set_ylabel("Variable " + str(var2))
    plt.title('2D Gaussian Kernel density estimation')
    #emprirical mean of the estimated distribution
    print('Mean: ' + str(np.mean(kernel.resample(100000),axis=1)))
plot_density(histogram,var1=1,var2=3)
#################################################################################






from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


# use grid search cross-validation to optimize the bandwidth
params = {'bandwidth': np.logspace(-1, 2, 500)} #This depends on dim(X)!
grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
grid.fit(histogram)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

def kde_sample(kde,n,GP_model=GP_model):
    sample = kde.sample(n)
    outside_ind = [0]
    while not len(outside_ind)==0:                                                                                                                                                                                                                                                                                                                                                     
        outside_ind = []
        for d in range(GP_model.D):
            outside_ind.extend(list(np.where((sample[:,d] < GP_model.original_bounds[d][0]) | (sample[:,d] > GP_model.original_bounds[d][1]))[0]))
        outside_ind = list(set(outside_ind))
        sample[outside_ind] = kde.sample(len(outside_ind))
    return(sample)
def plot_sample(sample,var1,var2,GP_model=GP_model):
    plt.plot(sample[:,var1-1],sample[:,var2-1],'ro',alpha=0.05)
    #plt.axis([GP_model.original_bounds[var1-1][0],GP_model.original_bounds[var1-1][1],GP_model.original_bounds[var2-1][0],GP_model.original_bounds[var2-1][1]])    
lambda_sample = kde_sample(kde,1000)
plot_sample(lambda_sample,var1=1,var2=2)
















