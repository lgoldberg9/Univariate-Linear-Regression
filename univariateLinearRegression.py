# -*- coding: utf-8 -*-
"""
some heckin description
"""

from math import sqrt
import functools
import operator
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 10E-6
MAX_STEPS = 500000

def norm(vec):
    """norm, computes the Euclidean norm of a vector

    Args:
        vec, a vector of real numbers

    Returns:
        norm_val, the norm of vec
    """

    return sqrt(functools.reduce(lambda x, y: x + y**2, vec, 0))

def gradient_descent_step(cost_gradient, regression_coefficients, descent_scale):
    """heck
    """
    old_x_reg_coeff = regression_coefficients[0]
    old_y_reg_coeff = regression_coefficients[1]
    new_x_reg_coeff = old_x_reg_coeff - descent_scale * cost_gradient[0](regression_coefficients)
    new_y_reg_coeff = old_y_reg_coeff - descent_scale * cost_gradient[1](regression_coefficients)
    return [new_x_reg_coeff, new_y_reg_coeff]

def gradient_descent(cost_gradient, initial_regression_coefficients, descent_scale, step=0):
    """heckin
    """
    has_converged = False
    has_exceeded_max_steps = False
    previous_regression_coefficients = initial_regression_coefficients
    while not has_converged and not has_exceeded_max_steps:
        previous_norm = norm(previous_regression_coefficients)
        new_regression_coefficients = gradient_descent_step(cost_gradient, \
                                previous_regression_coefficients, descent_scale)
        step = step + 1
        has_converged = (np.abs(previous_norm - norm(new_regression_coefficients)) < EPSILON)
        has_exceeded_max_steps = (step >= MAX_STEPS)
        previous_regression_coefficients = new_regression_coefficients
    return new_regression_coefficients, step

def hypothesis_func(theta, interpolate):
    """hypothesis_func,
    """
    return theta[0] + theta[1] * interpolate

def projective_derivative_term(theta, datum):
    """projective_derivative_term,
    """
    return hypothesis_func(theta, datum[0]) - datum[1]

def first_dimension_derivative_term(theta, datum):
    """first_dimension_derivative_term,
    """
    return (hypothesis_func(theta, datum[0]) - datum[1]) * datum[1]

def linear_regression(x_vec, y_vec, learning_scale=0.01):
    """doge
    """
    data_length = len(x_vec)
    initial_regression_coefficients = [0, 0]
    cost_deriv = \
    lambda deriv, theta: 1./data_length * \
    functools.reduce(operator.add, [deriv(theta, datum) \
                                    for datum in zip(x_vec, y_vec)])
    cost_gradient = [lambda theta: cost_deriv(projective_derivative_term, theta), \
                        lambda theta: cost_deriv(first_dimension_derivative_term, theta)]
    regression_coefficients, steps = gradient_descent(cost_gradient, initial_regression_coefficients, \
                                                learning_scale)
    return lambda x: hypothesis_func(regression_coefficients, x), steps

if __name__ == "__main__":
    # pathology
    # X_VEC = [2, 5, 9, 15, 4]
    # Y_VEC = [8, 4, 16, -2, 1]
    X_VEC = [1, 2, 3, 4, 5, 0]
    Y_VEC = [1, 2, 3, 4, 9, -1]
    X_NORM = norm(X_VEC)
    Y_NORM = norm(Y_VEC)
    NORMED_X_VEC = [x / X_NORM for x in X_VEC]
    NORMED_Y_VEC = [y / Y_NORM for y in Y_VEC]
    HYP, steps = linear_regression(NORMED_X_VEC, NORMED_Y_VEC)
    print "Steps: " + str(steps)
    heckin_range = np.linspace(min(NORMED_X_VEC), max(NORMED_X_VEC))
    plt.plot(heckin_range, [HYP(doggo) for doggo in heckin_range])
    plt.plot(NORMED_X_VEC, NORMED_Y_VEC, 'ro')
    print HYP(5)
    print HYP(6)



