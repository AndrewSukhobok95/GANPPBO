import sys
import os
import numpy as np

PROJECTDIR = os.getcwd()
PPBO_DIR = os.path.join(PROJECTDIR, "../PPBO")
GAN_DIR = os.path.join(PROJECTDIR, "../base_modules/ganspace")

sys.path.insert(0, PPBO_DIR)
sys.path.insert(0, GAN_DIR)

from PPBO.gp_model import GPModel
from PPBO.ppbo_settings import PPBO_settings
from PPBO.acquisition import next_query

from wrap_utils.ppbo_utils.feedback_storage import FeedbackStore

acquisition_strategy = 'EI'  # PCD
NUMBER_OF_QUERIES = 12
ADAPTIVE_INITIALIZATION = True  # At initilization: immediatly update the coordinate according to the user feedback
OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION = False
OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION = False
OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER = 1000

PPBO_settings = PPBO_settings(
    D=6,
    bounds=((-0.5, 0.5), (-0.5, 0.5), (4, 7), (-180, 180), (-180, 180), (-180, 180)),
    xi_acquisition_function=acquisition_strategy,
    verbose=False)

initial_queries_xi = np.array([list(np.eye(6)[i]) for i in range(6)])  # Initial xi:s correspond to unit vectors
initial_queries_x = np.array([[-0.5, -0.5, 5.0, -84.4, 142.8, 2.7],
                              [0.25, -0.25, 5.0, -84.4, 142.8, 2.7],
                              [-0.125, -0.125, 5.0, -84.4, 142.8, 2.7],
                              [-0.3147064250413807, -0.1379205809600735, 5.0, -84.4, 142.8, 2.7],
                              [-0.1420906798614234, -0.3133597318361268, 5.0, -84.4, 142.8, 2.7],
                              [0.3564088304603891, 0.3885800560423534, 5.0, -84.4, 142.8, 2.7]])
print("Number of initial queries is: " + str(len(initial_queries_xi)))

fs_ses = FeedbackStore()

GP_model = GPModel(PPBO_settings)

results_mu_star = []
results_x_star = []

for i in range(len(initial_queries_xi)):
    print("===== ITERATION", i, "=====")

    xi = initial_queries_xi[i].copy()
    x = initial_queries_x[i].copy()
    x[xi != 0] = 0

    current_bounds = PPBO_settings.original_bounds[np.argmax(xi)]
    alpha = np.random.uniform(current_bounds[0], current_bounds[1], 1)[0]
    fs_ses.save_feedback(x, xi, alpha)
    feedback = fs_ses.get_np_feedback()

    GP_model.update_feedback_processing_object(feedback)
    GP_model.update_data()
    GP_model.update_model()

    results_mu_star.append(GP_model.mu_star_)
    results_x_star.append(GP_model.x_star_)

for i in range(NUMBER_OF_QUERIES):
    print("Starting query " + str(i + 1) + "/" + str(NUMBER_OF_QUERIES) + " ...")

    ''' Compute next query '''
    xi_next, x_next = next_query(PPBO_settings, GP_model, unscale=True)

    ''' Present this to the user '''
    current_bounds = PPBO_settings.original_bounds[np.argmax(xi_next)]
    alpha = np.random.uniform(current_bounds[0], current_bounds[1], 1)[0]
    fs_ses.save_feedback(x_next, xi_next, alpha)
    feedback = fs_ses.get_np_feedback()

    ''' Append the user feedback '''
    GP_model.update_feedback_processing_object(feedback)
    GP_model.mu_star_previous_iteration = GP_model.mu_star_

    ''' Update the model '''
    GP_model.update_data()

    if i + 1 == OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER:
        GP_model.update_model(optimize_theta=True)
    else:
        GP_model.update_model(optimize_theta=OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION)

    ''' Save the predictive mean maximum and maximizer'''
    print("x_star of the iteration: " + str(GP_model.x_star_))
    results_mu_star.append(GP_model.mu_star_)
    results_x_star.append(GP_model.x_star_)

print("FINISH")

# should_log = False
# if should_log:
#     orig_stdout = sys.stdout
#     log_file = open('Camphor_Copper/user_session_log_'+str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+'.txt', "w")
#     sys.stdout = log_file
# GUI_ses = GUI.GUI_session(PPBO_settings)
# results_mu_star = []
# results_x_star = []


# GUI_ses = GUI.GUI_session(PPBO_settings)
# for i in range(len(initial_queries_xi)):
#     print("===== ITERATION", i, "=====")
#     ''' Present query to the user '''
#     xi = initial_queries_xi[i].copy()
#     if not i==0 and GUI_ses.user_feedback is not None and ADAPTIVE_INITIALIZATION:
#         initial_queries_x[i:,:] = GUI_ses.user_feedback
#     x = initial_queries_x[i].copy()
#     x[xi!=0] = 0
#     GUI_ses.set_x(x)
#     GUI_ses.set_xi(xi)
#     GUI_ses.run_iteration(allow_feedback=True)
#     ''' Create GP model for first time '''
#     if i==0:
#         GP_model = GPModel(PPBO_settings)
#     ''' Update GP model '''
#     GP_model.update_feedback_processing_object(np.array(GUI_ses.results))
#     GP_model.update_data()
#     GP_model.update_model()
#     ''' Save the predictive mean maximum and maximizer'''
#     print("x_star of the iteration: " + str(GP_model.x_star_))
#     results_mu_star.append(GP_model.mu_star_)
#     results_x_star.append(GP_model.x_star_)

# if OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION:
#     GP_model.update_model(optimize_theta=OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION)
# print("Initialization done!")

# for i in range(NUMBER_OF_QUERIES):
#     print("Starting query " + str(i+1)+"/"+str(NUMBER_OF_QUERIES)+" ...")
#     ''' Compute next query '''
#     xi_next,x_next = next_query(PPBO_settings,GP_model,unscale=True)
#     ''' Present this to the user '''
#     GUI_ses.set_xi(xi_next)
#     GUI_ses.set_x(x_next)
#     GUI_ses.run_iteration(allow_feedback=True)
#     ''' Append the user feedback '''
#     GP_model.update_feedback_processing_object(np.array(GUI_ses.results))
#     GP_model.mu_star_previous_iteration = GP_model.mu_star_
#     ''' Update the model '''
#     GP_model.update_data()
#     if i+1==OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER:
#         GP_model.update_model(optimize_theta=True)
#     else:
#         GP_model.update_model(optimize_theta=OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION)
#     ''' Save the predictive mean maximum and maximizer'''
#     print("x_star of the iteration: " + str(GP_model.x_star_))
#     results_mu_star.append(GP_model.mu_star_)
#     results_x_star.append(GP_model.x_star_)





