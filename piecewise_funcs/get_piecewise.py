'''
This file produces a Calliope AttrDict for the piecewise breakpoints of each
technology in the study. It then translates that to a yaml file for use in Calliope
model runs.
'''

import csv
import os

from calliope.utils import AttrDict

import minimize_linearisation_error as min_err

current_directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# get nonlinear curves from file
data = {}
techs = ["Load", "ahp", "boiler", "chp", "ar", "hrar", "ec", "TES-H", "TES-C"]
data_types = ['power','gas','htp']
max_load = [1, 500, 750, 352.26, 1500, 1000, 2500, 360, 3000]
min_load = [0, 0.10, 0.20, 0.20, 0.25, 0.30, 0.45, 0.00, 0.00]

for i in techs:
    data[i] = {}
    data[i]['max_load'] = max_load[techs.index(i)]
    data[i]['min_load'] = min_load[techs.index(i)]
    for j in data_types:
        data[i][j] = []

for data_type in data_types:
    with open(os.path.join(current_directory,
                           "nonlinear_curves_"+data_type+".csv")) as csvfile:
        reader = csv.DictReader(csvfile)
        for i in reader:
            for k, v in i.items():
                if v:
                    data[k][data_type].append(float(v)/data[k]['max_load'])

# get piecewise breakpoints for equidistant and optimised cases
for k, v in data.items():
    if k is not 'Load':
        data[k]['results'] = {}
        for data_type in data_types:
            if 'cop' in data_type:
                y_type = 'eff'
            else:
                y_type = 'con'
            if data[k][data_type]:
                data[k]['results'][data_type] = \
                    min_err.compare_pieces(data['Load'][data_type],
                                           data[k][data_type],
                                           [1, 2, 3, 4, 5],
                                           min_x=0,
                                           plot_results=False,
                                           y_type=y_type,
                                           slope_constraint=True)
            else:
                del data[k][data_type]

# translate result to AttrDict
resultsAttrDict = AttrDict({'pieces':{}})
for k in data.keys():
    if k is not 'Load':
        for y, v in data[k]['results']['elec']['y_eq'].items():
            resultsAttrDict.pieces.set_key(str(y) + '.eq.' + k + '.power.con',
                                           [float(i) for i in v])
        for y, v in data[k]['results']['elec']['x_eq'].items():
            resultsAttrDict.pieces.set_key(str(y) + '.eq.' + k + '.power.prod',
                                           [float(i) for i in v])
        for y, v in data[k]['results']['elec']['y_opt'].items():
            resultsAttrDict.pieces.set_key(str(y) + '.opt.' + k + '.power.con',
                                           [float(i) for i in v])
        for y, v in data[k]['results']['elec']['x_opt'].items():
            resultsAttrDict.pieces.set_key(str(y) + '.opt.' + k + '.power.prod',
                                           [float(i) for i in v])
        if 'gas' in data[k]['results']:
            if k == 'hrar':
                gas = 'recovered_heat'
            else:
                gas = 'gas'
            for y, v in data[k]['results']['gas']['y_eq'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.eq.' + k + '.' +
                                               gas + '.con', [float(i) for i in v])
            for y, v in data[k]['results']['gas']['x_eq'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.eq.' + k + '.' +
                                               gas + '.prod', [float(i) for i in v])
            for y, v in data[k]['results']['gas']['y_opt'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.opt.' + k + '.' +
                                               gas + '.con', [float(i) for i in v])
            for y, v in data[k]['results']['gas']['x_opt'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.opt.' + k + '.' +
                                               gas + '.prod', [float(i) for i in v])
        if 'htp' in data[k]['results']:
            for y, v in data[k]['results']['htp']['y_eq'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.eq.' + k + '.htp.con',
                                               [float(i) for i in v])
            for y, v in data[k]['results']['htp']['x_eq'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.eq.' + k + '.htp.prod',
                                               [float(i) for i in v])
            for y, v in data[k]['results']['htp']['y_opt'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.opt.' + k + '.htp.con',
                                               [float(i) for i in v])
            for y, v in data[k]['results']['htp']['x_opt'].items():
                resultsAttrDict.pieces.set_key(str(y) + '.opt.' + k + '.htp.prod',
                                               [float(i) for i in v])

# translate AttrDict to yaml file
resultsAttrDict.to_yaml('piecewise.yaml')