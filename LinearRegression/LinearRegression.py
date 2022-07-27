import numpy as np
import matplotlib.pyplot as plt
import copy, math
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

plt.style.use('./deeplearning.mplstyle')


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):
    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

    w = copy.deepcopy(w_in)  # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing


# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0])
y_train = np.array([130.0, 154.0, 206.0, 206.9, 208.0, 140.0, 128.0, 135.0, 161.0, 165.0, 166.0, 170.0, 201.0, 216.0,
                    218.0, 248.0, 251.0, 157.0, 165.0, 180.0, 267.0, 270.0, 280.0, 148.0, 290.0, 300.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# m is the number of training examples (second method)
# m = len(x_train)
# print(f"Number of training examples is: {m}")

# Plot the data points
plt.scatter(x_train, y_train, marker='o', c='r')
# Set the title
plt.title("Thermal Conductivity vs wt% of GNPs")
# Set the y-axis label
plt.ylabel('Thermal Conductivity (W/mK)')
# Set the x-axis label
plt.xlabel('wt% of Graphene in pure Al')
plt.show()

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:4.2f},{b_final:4.2f})")


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


tmp_f_wb = compute_model_output(x_train, w_final, b_final,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title
plt.title("Thermal Conductivity vs wt% of GNPs")
# Set the y-axis label
plt.ylabel('Thermal Conductivity (W/mK)')
# Set the x-axis label
plt.xlabel('wt% of Graphene in pure Al')
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[25, 80, 0.5], b_range=[120, 200, 0.5],
             contours=[10, 50, 100, 200], resolution=0.5)
plt.show()
