'''
Functions in this file get the breakpoint positions for a piecewise curve
which describes a nonlinear curve. The positions are either optimised, using SLSQP
or are taken as being equidistant along the curve.
'''

import random

from scipy.optimize import minimize
from operator import add

import numpy as np
import matplotlib.pyplot as plt

def _optimise_pieces(x, y, p, plot=None, slope_constraint = False):
    # Find least error piecewise linear curve of `p` pieces to fit the curve
    # described by `x` & `y`
    if p >= 2 and isinstance(p,int) and len(y) == len(x):
        def min_error(listx):
            listy = np.interp(listx, x, y)
            if slope_constraint:
                def get_slope(x, y):
                    if (x[1] - x[0]) != 0:
                        return ((y[1] - y[0])/(x[1] - x[0]))
                    else:
                        return 0
                slope =[]
                for i in range(len(listx)-1):
                    slope.append(get_slope(listx[i:i+2], listy[i:i+2]))
                ascending = sorted(slope) == slope
                descending = sorted(slope, reverse = True) == slope
                if ascending == descending:
                    output = 10
                else:
                    output = 0
            else:
                output = 0
            output += 0 if (sorted(listx) == listx).all() else 10
            listy_long = np.interp(x, listx, listy)
            return np.sqrt(((y - listy_long) ** 2).mean()) + output

        bnds = tuple((x[0], x[0]) for i in range(1)) + tuple((x[0], x[-1])
                     for i in range(p-1)) + tuple((x[-1], x[-1]) for i in range(1))
        # Use sequential least squares programming - need to re-initialise list0
        # with random variables a few times to ensure not
        # getting stuck in a local optimum
        Runs = {}
        x_vals = []
        y_vals = []
        error = []
        for i in range(20):
            list0 = np.hstack((np.array(x[0]), np.sort(np.random.rand(p-1)),
                               np.array(x[-1])))
            res = minimize(min_error, list0, method='SLSQP', bounds=bnds)
            Runs["Run {}".format(i)] = {"x_vals":res.x,
                                        "y_vals":np.interp(res.x, x, y),
                                        "error":res.fun, "initial_vals":list0}
            x_vals.append(res.x)
            y_vals.append(np.interp(res.x,x,y))
            error.append(res.fun)
        Runs["Total"] = {}
        Runs["Total"]["x_vals"] = x_vals[np.where(error==min(error))[0][0]]
        Runs["Total"]["x_var"] = np.var(x_vals,0)

        Runs["Total"]["y_vals"] = y_vals[np.where(error==min(error))[0][0]]
        Runs["Total"]["y_var"] = np.var(y_vals,0)

        Runs["Total"]["error"] = min(error)
        Runs["Total"]["error_var"] = np.var(error,0)

        #print('x values = {} \n y values = {}'.format(x_vals, y_vals))
        if plot:
            plt.plot(x,y)
            plt.plot(list(res.x),np.interp(list(res.x),x,y))
            plt.show()

        return Runs

    elif p < 2:
        print('p must be at least 2')
    elif not isinstance(p,int):
        print('p must be an integer')
    elif not isinstance(p,int):
        print('p must be an integer')
    elif not len(y) == len(x):
        print('length of x and y must be equal')

def _equidistant_pieces(x, y, p, plot=None):
    '''
    Get p breakpoint positions for the given nonlinear curve described by (x, y)
    '''
    if p >= 2 and isinstance(p, int) and len(y) == len(x):
        locx = []
        locy = []

        # get equidistant distance between sections
        step = int(len(x)/(p))
        for i in range(p):
            locx.append(x[step*i])
            locy.append(y[step*i])
        locx.append(x[-1])
        locy.append(y[-1])

        if plot:
            plt.plot(x,y)
            plt.plot(locx,locy)
            plt.show()

        locy_long = np.interp(x,locx,locy)
        return locx, locy, np.sqrt(((y - locy_long) ** 2).mean())
    elif p < 2:
        print('p must be at least 3')
    elif not isinstance(p,int):
        print('p must be an integer')
    elif not len(y) == len(x):
        print('length of x and y must be equal')

def compare_pieces(x, y, p, min_x=0, save_fig=None,
                   save_format='png', plot_results=True,
                   y_type='eff', slope_constraint=False):
    """
    Function to compare single value, straight line, and equidistant/optimised
    piecewise linearisation of curves.

    Curves are assumed to be between 0 and 1 on the x axis

    x = nonlinear x axis list
    y = nonlinear y axis list
    p = number of pieces for piecewise linearisation (can be a list to check
        several piece numbers)
    min_x: float; minimum value of x, used for straight line linearisation.
           Default = 0
    save_fig = list of [directory, filename] where the produced figures can go.
               Default = None
    save_format: str; format in which to save the figures (e.g. 'svg', 'png')
    plot_results = bool; whether to show plotted curves. Default = False
    y_type: str; efficiency 'eff' or consumption 'con' to get the fixed line right
           (either horizontal or diagonal). Default = 'eff'.
    slope_constraint: bool; Force optimised curves to have continuously
                      increasing/decreasing slope. Default = False

    Returns a results nested dictionary, containing lists for breakpoints in each case.
    """
    if not len(y) == len(x):
        return 'length of y must equal length of x'

    if not isinstance(p,list) and not isinstance(p,int):
        return 'p must be list of integers or integer'

    if save_fig:
        if not len(save_fig) == 2:
            return ('save_fig must be two strings in a list, first string = '
                   'directory, second string = filename')
        elif not isinstance(save_fig[0],str) or not isinstance(save_fig[1],str):
            return 'save_fig entries must be strings'
    results = {} # results are saved in dictionary
    for i in p: #need a nested dict for each set of pieces
        results[i] = {}
    #take rated (max. load rate) value across all x

    x_fixed = (x[0], x[-1])
    if y_type == 'eff':
        y_fixed = (y[-1], y[-1])
    elif y_type == 'con':
        y_fixed = (0, y[-1])
    y_fixed_long = np.interp(x,x_fixed,y_fixed) #get y value for each x value

    #get error between fixed & nonlinear
    er_fixed = np.sqrt(((y - y_fixed_long) ** 2).mean())

    results['x_fixed']=x_fixed
    results['y_fixed']=y_fixed
    results['er_fixed']=er_fixed

    #single line from minimum load rate to rated value at max. load rate
    if min_x == 0:
        x_simp = (x[0], x[-1])
        y_simp = (y[0], y[-1])
        y_simp_long = np.interp(x,x_simp,y_simp)
        er_simp = np.sqrt(((y - y_simp_long) ** 2).mean())
    else:
        x_simp = (min_x, x[-1])
        min_x_loc = np.where(np.round(x,2)==min_x)[0][0]
        y_simp = (y[min_x_loc], y[-1])
        y_simp_long = np.hstack((np.zeros(min_x_loc), np.interp(x[min_x_loc:],
                      x_simp,y_simp)))
        er_simp = np.sqrt(((y[min_x_loc:-1] - y_simp_long) ** 2).mean())
    #get error between simple linear & nonlinear


    # Case 1: several pieces in a list
    if isinstance(p,list):
        x_opt = {}
        y_opt = {}
        er_opt = []
        x_eq = {}
        y_eq = {}
        er_eq = []
        fig0 = plt.figure(0)
        plt.plot(x,y,'c-')
        if 1 in p: # p = 1 is the same as simple linear (dealt with above)
            plt.plot(x_simp,y_simp, 'black')
            p = p[1:] # can't handle p = 1 in the other functions, so delete it (assume it is in 0th element of list)
            results['x_simp']=x_simp
            results['y_simp']=y_simp
            results['er_simp']=er_simp

        for i in p:
            Runs = _optimise_pieces(x, y, i, slope_constraint=slope_constraint)
            x_opt[i] = Runs["Total"]["x_vals"]
            y_opt[i] = Runs["Total"]["y_vals"]
            _er_opt = Runs["Total"]["error"]
            x_eq[i], y_eq[i], _er_eq = _equidistant_pieces(x, y, i)
            plt.plot(x_opt[i], y_opt[i], 'g.-')
            plt.plot(x_eq[i], y_eq[i], 'b.-')
            er_opt.append(_er_opt)
            er_eq.append(_er_eq)

            results['x_opt'] = x_opt
            results['y_opt'] = y_opt
            results['er_opt'] = er_opt

            results['x_eq'] = x_eq
            results['y_eq'] = y_eq
            results['er_eq'] = er_eq

            # Include the full dictionary of results from _optimise_pieces
            results[i]["optimisation_runs"]=Runs
        plt.plot(x_fixed,y_fixed,'r-')

        #save the figures if save_fig is given in function initialisation
        if save_fig:
            file_name = save_fig[0] + '\\' + save_fig[1] + '_curves.svg'
            plt.savefig(file_name, format = 'svg')

        # plot the load-rate results if given as True in function initialisation
        if plot_results:
            plt.show()

        fig0.clear()

        # Create bar chart comparing error from using different pieces
        fig1 = plt.figure(1)
        plt.bar(np.array(p)-0.1, er_opt, width=0.1,color='green')
        plt.bar(p, er_eq, width=0.1,color='blue')
        plt.bar(0, er_fixed, width=0.1,color='red')
        plt.bar(0.95, er_simp, width=0.1,color='red')

        #save the figures if save_fig is given in function initialisation
        if save_fig:
            file_name = save_fig[0] + '\\' + save_fig[1] + '_errors.svg'
            plt.savefig(file_name, format = 'svg')

        # plot the error results if given as True in function initialisation
        if plot_results:
            plt.show()

        fig1.clear()
        # construct results dictionary


    else:
        plt.plot(x, y, 'c-')
        if p == 1:
            plt.plot(x_simp, y_simp)
            results['x_simp'] = x_simp
            results['y_simp'] = y_simp
            results['er_simp'] = er_simp
        else:
            Runs = _optimise_pieces(x, y, p, slope_constraint=slope_constraint)
            x_opt = Runs["Total"]["x_vals"]
            y_opt = Runs["Total"]["y_vals"]
            _er_opt = Runs["Total"]["error"]
            x_eq, y_eq, er_eq = _equidistant_pieces(x,y,p)
            plt.plot(x_opt,y_opt,'g.-')
            plt.plot(x_eq, y_eq,'b.-')
            results['x_opt'] = x_opt
            results['y_opt'] = y_opt
            results['er_opt'] = er_opt

            results['x_eq'] = x_eq
            results['y_eq'] = y_eq
            results['er_eq'] = er_eq
            # Include the full dictionary of results from _optimise_pieces
            results["optimisation_runs"]=Runs
        plt.plot(x_fixed,y_fixed,'r-')

        #save the figures if save_fig is given in function initialisation
        if save_fig:
            file_name = save_fig[0] + '\\' + save_fig[1] + '_curves.' + save_format
            print(file_name)
            plt.savefig(file_name, format = save_format)

        # plot the load-rate results if given as True in function initialisation
        if plot_results:
            plt.show()
    return results