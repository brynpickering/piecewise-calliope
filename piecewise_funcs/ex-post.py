'''
This file contains two functions for ex-post analysis of Calliope models with piecewise curves.

`get_error` takes a Calliope solution file (xarray DataArray) and finds
the error in technology consumption when applying nonlinear consumption curves
to the technology production values, at each time-step.

`plot_load_histograms` plots the output histogram of each technology for the
single value efficiency model and each optimised piecewise model. This was used
to produce figure 11 in the paper:
`Pickering, B. and Choudhary, R. (in press). "Applying Piecewise Linear
Characteristic Curves in District Energy Optimisation" 30th INTERNATIONAL
CONFERENCE on Efficiency, Cost, Optimisation, Simulation and Environmental
Impact of Energy Systems. San Diego, USA. 2017.`

`plot_cost_error` loops through `get_error` for each solution file and plots the
results, to show the difference between objective function value and application
of nonlinear consumption curves ex-post. This was used to produce tables 3 and
4 in the paper:
`Pickering, B. and Choudhary, R. (in press). "Applying Piecewise Linear
Characteristic Curves in District Energy Optimisation" 30th INTERNATIONAL
CONFERENCE on Efficiency, Cost, Optimisation, Simulation and Environmental
Impact of Energy Systems. San Diego, USA. 2017.`
'''

import calliope
import os
import csv

from plotly import tools
from plotly.offline import plot

import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import numpy as np
import xarray as xr
import pandas as pd

current_directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def get_colour(colour, opacity=None, gradient=None):
    '''
    translate seaborn colour palette outputs (RGB 0 to 1) to inputs for plotly
    (RGB 0 to 255)
    '''
    if opacity:
        colours = np.multiply(sns.color_palette("cubehelix", 5), 255)
        colour_match = {'blue': 0, 'green': 1, 'brown': 2, 'pink': 3, 'aqua': 4}
        colour_num = colour_match[colour]
        return 'rgba({}, {}, {}, {})'.format(colours[colour_num][0],
                                             colours[colour_num][1],
                                             colours[colour_num][2],
                                             opacity)
    elif gradient:
        colours = np.multiply(sns.cubehelix_palette(gradient), 255)
        return 'rgb({}, {}, {})'.format(colours[colour][0],
                                        colours[colour][1],
                                        colours[colour][2])
    else:
        colours = np.multiply(sns.color_palette("cubehelix", 5), 255)
        colour_match = {'blue': 0, 'green': 1, 'brown': 2, 'pink': 3, 'aqua': 4}
        colour_num = colour_match[colour]
        return 'rgb({}, {}, {})'.format(colours[colour_num][0],
                                        colours[colour_num][1],
                                        colours[colour_num][2])

def get_error(solution):
    '''
    Get information from Calliope solution file and compare it to nonlinear curves,
    found in relevant csv files.
    Outputs two xarray DataArrays, one involving timeseries, one not.
    '''

    techs=['chp','hrar','ec','ahp','boiler']

    # technology carrier production
    prod = xr.DataArray(np.zeros([len(techs),len(solution.x), len(solution.t)]),
                        coords=[solution.y.loc[techs], solution.x, solution.t])
    # technology carrier consumption
    con = xr.DataArray(np.zeros([len(techs),len(solution.x), len(solution.t)]),
                       coords=[solution.y.loc[techs], solution.x, solution.t])
    # technology carrier heat to power ratio (HTP) (only relevant for CHP)
    htp = xr.DataArray(np.zeros([len(solution.x), len(solution.t)]),
                       coords=[solution.x, solution.t])

    # Get, for each technology and location, the production and consumption at
    # each time step, normalised against technology maximum capacity
    for y in techs:
        carrier = solution.metadata.loc[dict(cols_metadata='carrier', y=y)]
        source_carrier = solution.metadata.loc[dict(cols_metadata='source_carrier', y=y)]
        for x in solution.x:
            if solution.e_cap.loc[dict(x=x, y=y)] > 0:
                solution.e.loc[dict(x=x, y=y, c=carrier)][
                    solution.e.loc[dict(x=x, y=y, c=carrier)] < 0] = 0
                solution.e.loc[dict(x=x, y=y, c=source_carrier)][
                    solution.e.loc[dict(x=x, y=y, c=source_carrier)] > 0] = 0
                for t in solution.t:
                    prod.loc[dict(x=x, y=y, t=t)] = \
                        float("{0:.3f}".format(float(solution.e.loc[
                            dict(x=x, y=y, c=carrier, t=t)].to_pandas()) /
                            solution.e_cap.loc[dict(x=x, y=y)].to_pandas()))
                    con.loc[dict(x=x, y=y, t=t)] = \
                        float("{0:.3f}".format(float(solution.e.loc[
                            dict(x=x, y=y, c=source_carrier, t=t)].to_pandas()) /
                            solution.e_cap.loc[dict(x=x, y=y)].to_pandas()))
                    if y == 'chp':
                        htp.loc[dict(x=x, t=t)] = \
                            float("{0:.3f}".format(float(solution.e.loc[
                                dict(x=x, y=y, c='district_heat', t=t)].to_pandas()) /
                                solution.e_cap.loc[dict(x=x, y=y)].to_pandas()))
    # get original nonlinear curves
    data = {}
    techs = ["ahp", "boiler", "chp", "hrar", "ec"]
    data_types = ['power','gas','htp']
    max_load = [500, 750, 352.26, 1000, 2500]
    min_load = [0.10, 0.20, 0.20, 0.30, 0.45]

    # create a dictionary for loading original data
    for i in techs:
        data[i] = {}
        data[i]['max_load'] = max_load[techs.index(i)]
        data[i]['min_load'] = min_load[techs.index(i)]
        for j in data_types:
            data[i][j] = []

    # There is a csv file for consumption of power, gas, and HTP
    for data_type in data_types:
        with open(os.path.join(current_directory,
                               "nonlinear_curves_"+data_type+".csv")) as csvfile:
            reader = csv.DictReader(csvfile)
            for i in reader:
                for k, v in i.items():
                    if v and k in techs:
                        data[k][data_type].append(float(v)/data[k]['max_load'])


    df = pd.DataFrame(data)
    d = xr.DataArray(df.loc[['power', 'gas', 'htp']], dims=['load_type','y'])
    Load = [float("{0:.3f}".format(float(i))) for i in np.linspace(0, 1, 1001)]

    # following technologies don't consume power, so just create empty entries for them
    d.loc[dict(load_type='power', y=['chp', 'boiler', 'hrar'])] = [[],[],[]]

    # sum gas and power loads together, as no technology has a consumption curve for both curves
    con_nl = d.loc[dict(load_type=['power', 'gas'])].sum(dim='load_type')
    # HTP only concerns CHP
    htp_nl = d.loc[dict(load_type='htp', y='chp')]

    errors = ['sum_con', 'sum_htp', 'sum_cost']
    errors_t = ['con', 'htp']
    error = xr.DataArray(np.zeros([len(techs), len(solution.x), len(errors)]),
                         coords=[solution.y.loc[techs], solution.x, errors])
    error = error.rename({'dim_2':'errors'})
    error_t = xr.DataArray(np.zeros([len(techs), len(solution.x), len(solution.t), len(errors_t)]),
                           coords=[solution.y.loc[techs], solution.x, solution.t, errors_t])
    error_t = error_t.rename({'dim_3':'errors'})
    for y in techs:
        for x in solution.x:
            # ignore tiny technologies, which obviously contribute nothing
            if solution.e_cap.loc[dict(y=y, x=x)] < 1e-6:
                continue
            nonlinear_fuel = [] # fuel = gas or power
            nonlinear_heat = []
            # for each value of produced energy (at each timestep),
            # find its corresponding consumption on the nonlinear consumption curve
            for p in prod.loc[dict(y=y, x=x)]:
                if p == 0:
                    nonlinear_fuel.append(0)
                else:
                    nonlinear_fuel.append(con_nl.loc[dict(y=y)].to_dict()['data'][Load.index(p)])
                # if it is CHP do the same for HTP
                if y =='chp':
                    nonlinear_heat.append(htp_nl.to_dict()['data'][Load.index(p)])
            # At every time step, find the error as (consumption_nonlinear - consumption_from_model) * technology capacity. consumption_from_model is a negative value, hence the addition below.
            error_t.loc[dict(y=y, x=x, errors='con')]=list(np.multiply(np.add(nonlinear_fuel, con.loc[dict(y=y, x=x)]), solution.e_cap.loc[dict(y=y, x=x)]))
            # Get the root mean square error across all time steps
            error.loc[dict(y=y, x=x, errors='sum_con')]=np.sqrt(((np.add(nonlinear_fuel, con.loc[dict(y=y, x=x)])) ** 2).mean()) # * results[y][x]['e_cap']
            # Do the same for HTP
            if y == 'chp':
                error_t.loc[dict(y=y, x=x, errors='htp')]=list(np.multiply(np.subtract(nonlinear_heat,htp.loc[dict(x=x)]), solution.e_cap.loc[dict(y=y, x=x)]))
                error.loc[dict(y=y, x=x, errors='sum_htp')]=np.sqrt(((np.add(nonlinear_fuel, htp.loc[dict(x=x)])) ** 2).mean())
            # Now look at cost errors, by multiplying the consumption error at each time step with the cost of consumption
            if y in ['boiler','chp']:
                error.loc[dict(y=y, x=x, errors='sum_cost')]=sum(error_t.loc[dict(y=y, x=x, errors='con')])*.025
            elif y in ['ec', 'ahp']:
                error.loc[dict(y=y, x=x, errors='sum_cost')]=sum(error_t.loc[dict(y=y, x=x, errors='con')])*.095
            else: #hrar
                error.loc[dict(y=y, x=x, errors='sum_cost')]=sum(error_t.loc[dict(y=y, x=x, errors='con')])*.025/.8
    return error, error_t

def plot_load_histograms():
    '''
    plot histograms of technology production, normalised against technology capacity
    '''
    # get list of solution folders. NOTE: Output folder needs to be in 'models' subdirectory, alongside the run files
    solutions = [f.path for f in os.scandir(os.path.join(current_directory, '../models/Output')) if f.is_dir() ]

    # Create plotly figure, with subfigures, assuming you have results for all optimised runs
    fig = tools.make_subplots(rows=5, cols=2,
                              shared_xaxes=True, shared_yaxes=True,
                              vertical_spacing=0.02)
    shownlegend = []

    # Assign colours to technologies
    colours =  {'ec':'blue', 'ahp':'green', 'chp':'brown', 'boiler':'pink', 'hrar':'aqua'}
    for i in solutions:
        try:
            # NOTE: will open 'solution.nc', you may have more recent versions (e.g. 'solution_1.nc')
            # but it will ignore these unless you change it here.
            solution = calliope.read.read_netcdf(os.path.join(i, 'solution.nc'))
        except: #i.e. no solution file found
            continue

        # winter in first column of subfigures, summer in second column
        if 'winter' in i:
            column = 1
        else:
            column = 2

        # row number depends on number of breakpoints
        if 'piecewise_info' in solution.config_model.keys():
            row = solution.config_model.piecewise_info.N
            title = solution.config_model.piecewise_info.linearisation + '_' + \
                str(solution.config_model.piecewise_info.N)
        else: # SVE
            row = 1
            title = 'no piecewise'

        zeros = 0
        ones = 0
        techs = ['ec', 'ahp', 'chp', 'boiler', 'hrar']
        for y in techs:
            name = solution.metadata.loc[dict(cols_metadata='name', y=y)].to_pandas()
            colour = get_colour(colours[y])
            c = solution.metadata.loc[dict(cols_metadata='carrier', y=y)].to_pandas()
            prod = solution.e.loc[dict(y=y, c=c)].sum(dim='x')
            # ignore negligibly small production values
            if prod.sum() > 1e-6:
                showlegend = False if y in shownlegend else shownlegend.append(y)
                e_cap = solution.e_cap.loc[dict(y=y)].sum(dim='x')
                _zeros = zeros + len(prod.loc[prod == 0])
                _ones = ones + len(prod.loc[prod == e_cap])
                others = prod.loc[(prod < e_cap) & (prod > 0)]/e_cap
                # histogram in 10% intervals for the range (0, 1)
                fig.append_trace(go.Histogram(x=others.to_pandas(), opacity=.7, legendgroup=y,
                                              xbins=dict(start=0, end=1, size=0.1),
                                              autobinx=False, marker=dict(color=colour),
                                              name=y, showlegend=showlegend),
                                 solutions.index(row, column))
                # at no load (i.e. 0 load rate) give values as a line
                fig.append_trace(go.Scatter(x=[0,0], y=[zeros, _zeros], legendgroup=y,
                                            showlegend=False, marker=dict(symbol=141),
                                            line=dict(color=colour), name=y),
                                 solutions.index(row, column))
                # at full load (i.e. 100% load rate) give values as a line
                fig.append_trace(go.Scatter(x=[1,1],y=[ones, _ones], legendgroup=y,
                                            showlegend=False, marker=dict(symbol=141),
                                            line=dict(color=colour), name=y),
                                 solutions.index(row, column))
                zeros = _zeros
                ones = _ones
            showlegend = True
        fig['layout']['yaxis'+ str(solutions.index(row))]['title']=title
        fig['layout']['yaxis'+ str(solutions.index(row))]['range']=[0, 250]
    fig['layout'].update(barmode='relative', bargap=0)
    plot(fig, filename=os.path.join(current_directory, '../models/Output/load_histogram.html'), image='svg')

def plot_cost_error(output_folder):

    # get list of solution folders. NOTE: Output folder needs to be in 'models' subdirectory, alongside the run files
    solutions = [f.path for f in os.scandir(os.path.join(current_directory, '../models/Output')) if f.is_dir() ]
    fig = tools.make_subplots(rows=2, cols=1,
                                shared_xaxes=True, shared_yaxes=False,
                                vertical_spacing=0.02, subplot_titles=['Winter', 'Summer'])
    shownlegend = []

    for i in solutions:
        try:
            # NOTE: will open 'solution.nc', you may have more recent versions (e.g. 'solution_1.nc')
            # but it will ignore these unless you change it here.
            solution = calliope.read.read_netcdf(os.path.join(i, 'solution.nc'))
        except: #i.e. no solution file found
            continue

        e, e_t = get_error_netcdf(solution)
        time = solution.run_time
        # winter in first row of subfigures, summer in second row
        if 'winter' in i:
            row = 1
        else:
            row = 2
        nl = 0
        techs = ['chp','hrar','ec','ahp','boiler']
        net_cost = 0
        net_cost_nl = 0
        for y in solution.y:
            cost = float(solution.costs.loc[dict(y=y, kc='monetary')].sum(dim='x'))
            revenue = float(solution.revenue.loc[dict(y=y, kr='monetary')].sum(dim='x'))
            if y in techs:
                cost_nl = float(e.loc[dict(y=y, errors='sum_cost')].sum(dim='x')) + cost
            else:
                cost_nl = cost
            net_cost += cost - revenue
            net_cost_nl += cost_nl - revenue
        if 'piecewise_info' in solution.config_model.keys():
            N = solution.config_model.piecewise_info.N
            linearisation = solution.config_model.piecewise_info.linearisation
        else:
            N = 1
            linearisation = 'none'
            colour = get_colour('brown')
        _nl = nl + net_cost_nl
        if linearisation == 'opt':
            N = N - .05
            colour = get_colour('green')
        elif linearisation == 'eq':
            N = N + .05
            colour = get_colour('aqua')
        print(i, net_cost, net_cost_nl, time)
        showlegend = False if linearisation in shownlegend else shownlegend.append(linearisation)
        # Objective funciton value
        fig.append_trace(go.Bar(x=[N], y=[net_cost], legendgroup=linearisation,
                               showlegend=showlegend,
                               marker=dict(color=colour),
                               opacity=.7,
                               name=linearisation,
                               width=.1,
                               text=['run cost: £{}'.format(int(net_cost))],
                               hoverinfo='text'),
        j, 1)
        # system cost following applying nonlinear consumption curves ex-post
        fig.append_trace(go.Scatter(x=[N, N], y=[0, net_cost_nl],
                                    legendgroup=linearisation,
                                    showlegend=False,
                                    marker=dict(color=colour, symbol=141),
                                    line=dict(color=colour),
                                    name=linearisation,
                                    text='nonlinear cost: £{}'.format(int(net_cost_nl)),
                                    hoverinfo='text'),
        j, 1)
        # solution time
        fig.append_trace(go.Scatter(x=[N], y=[time], legendgroup=linearisation,
                                    showlegend=False,
                                    name=linearisation,
                                    text='time: {}'.format(time),
                                    hoverinfo='text',
                                    marker=dict(color=colour, line=dict(color='black'))),
        j, 1)
        showlegend = True
    fig['layout'].update(bargap=0)
    plot(fig, filename=os.path.join(current_directory, '../models/Output/load_histogram.html'), image='svg')


