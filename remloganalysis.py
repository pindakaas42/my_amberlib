#!/usr/bin/python
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def get_exchange(ex_line):
    splitline = ex_line.split()
    orig_wind = splitline[0]
    ex_partner = splitline[1]

    if splitline[7] == 'F':
        return dict(orig=orig_wind, ex_partner=orig_wind)
    if splitline[7] == 'T':
        return dict(orig=orig_wind, ex_partner=ex_partner)


def get_exchlines(openremfile):
    row = []
    for line in openremfile:
        if line[0] != '#':
            row.append(get_exchange(line)['ex_partner'])
        if line[0] == '#':
            yield row
            row = []


def get_orig_window_pos(rows):
    first_counter = 0

    for row in rows:
        if row == []:
            yield []
        else:
            if first_counter == 0:
                first_counter += 1
                start = {'window'+str(i+1): i+1 for i in range(len(row))}

            movements = [int(i)-int(j) for (i, j) in
                         zip(
                             row, [i+1 for i in range(len(row))]
                             )
                         ]
            for window in start:
                start[window] = movements[start[window]-1]+start[window]
            yield start


def remlog_plot():
    data = np.loadtxt('results.log', int)
    color = iter(cm.cool(np.linspace(0, 1, len(data[0, :]))))
    for i in data.T:
        plt.plot(i[::(len(data.T[0])/30)], color=next(color))
    return data


def make_results():
    ''''generates the file with all the swaps'''
    with open('rem.log', 'r') as remlog:
        rows = get_exchlines(remlog)
        windows_pos = get_orig_window_pos(rows)

        with open('results.log', 'w') as results:
            for pos in windows_pos:
                results.write(' '.join([str(pos['window'+str(k+1)])
                                        for k in range(len(pos))]) + '\n')
