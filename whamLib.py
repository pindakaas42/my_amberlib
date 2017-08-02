#!/usr/bin/python
import numpy as np
import pandas as pa
import sys
import re
import os
import glob
import myLib as lib
import pytraj
disangfilenames = sys.argv[1:]


def read_disangfiles(disangfilenames):
    '''Takes a list of disangfilenames. Saves the .disang filenames, position of
    minimum and force costant from the disang files and saves the triple for
    each file in a dict and these in a list. '''
    disangparams = []
    for filename in disangfilenames:

        disangparams.append(
            dict(name=filename, minpos=None, forceconst=None))

        with open(filename, 'r') as f:
            for line in f:

                match_minpos = re.search(r'r2=([0-9]+\.?[0-9]*)', line)
                if match_minpos is not None:
                    disangparams[-1]['minpos'] = float(match_minpos.group(1))

                match_forceconst = re.search('rk2=([0-9]+\.?[0-9]*)', line)
                if match_forceconst is not None:
                    disangparams[-1]['forceconst'] = float(
                        match_forceconst.group(1))
    return disangparams


def read_disang_angle_files(disangfilename):
    '''Takes a single disang ANGLE filename
    with more that just a standard distance restraint and gives a list of
    dicts, the dicts contains the disangparams of the file in the sequence
    of occurence in the disang file'''
    disangparams = []

    with open(disangfilename, 'r') as f:
        for line in f:
            if '&rst' in line:
                disangparams.append(dict(angle_or_dist=None,
                                         forceconst=None,
                                         atomgroups=None))
            match_minpos = re.search(r'([^g]r2|R0)=(-?[0-9]+\.?[0-9]*)', line)
            if match_minpos is not None:
                disangparams[-1]['angle_or_dist'] = float(match_minpos.group(2))

            match_forceconst = re.search(r'(rk2|K0)=([0-9]+\.?[0-9]*)', line)
            if match_forceconst is not None:
                disangparams[-1]['forceconst'] = float(
                    match_forceconst.group(2))

            if 'ig' in line:
                line = line.strip()
                pos_of_eq = line.find('=')
                val = line[pos_of_eq+1::]
                if disangparams[-1]['atomgroups'] is None:
                    disangparams[-1]['atomgroups'] = []
                disangparams[-1]['atomgroups'].append(
                    [i for i in val.split(',') if i != ''])
    return disangparams


def read_disang_angle_files_for_wham(disangfilenames):
    '''takes a list of disangfiles and takes the first entry wich needs to
    be the distance between atom groups term and returns the disangparams
    for all the files (works if there is more, angles and so on after that)'''
    disangparams = []
    for filename in disangfilenames:
        distparams = read_disang_angle_files(filename)[0]
        disangparams.append(dict(name=filename,
                                 minpos=distparams['angle_or_dist'],
                                 forceconst=distparams['forceconst']))
    return disangparams


def new_read_disangfiles(disangfilenames):
    '''Takes a list of disangfilenames. Saves the .disang filenames, position of
    minimum and force costant from the disang files and saves the triple for
    each file in a dict and these in a list. '''
    disangparams = []
    for filename in disangfilenames:

        with open(filename, 'r') as f:
            for line in f:
                if '&rst' in line:
                    disangparams.append(
                        dict(name=filename, minpos=None, forceconst=None))

                match_minpos = re.search(r'r2=([0-9]+\.?[0-9]*)', line)
                if match_minpos is not None:
                    disangparams[-1]['minpos'] = float(match_minpos.group(1))

                match_forceconst = re.search('rk2=([0-9]+\.?[0-9]*)', line)
                if match_forceconst is not None:
                    disangparams[-1]['forceconst'] = float(
                        match_forceconst.group(1))

                match_atomgroup = re.search(r'(igr[0-9])=([ 0-9,]*)', line)
                if match_atomgroup is not None:
                    disangparams[-1][match_atomgroup.group(1)] = \
                                                 match_atomgroup.group(2)
    return disangparams


def get_metadatafile(disangfilenames):
    s = ''
    for filedata in read_disangfiles(disangfilenames):
        s += ("{0} {1} {2} \n".format(filedata['name']+'.out',
                                      filedata['minpos'],
                                      2*float(filedata['forceconst']))
              )
    return s


def load_one_disangout(disangoutfilename):
    dat = pa.read_csv(disangoutfilename,
                      sep='\s+', usecols=[1], dtype={1: float})[:-1]
    dat = np.array(dat.values, dtype=np.dtype([('dist', float)]))
    return dat


def load_disangout(disangoutfilenames):
    ''' takes a list of disangoutfilenames and returnes a list of  tupels of
    the files contents and the name of the file'''
    disangdatalist = [(np.array(pa.read_csv(filename,
                                            sep='\s+',
                                            usecols=[1],
                                            dtype={1: float})[:-1],
                                dtype=np.dtype([('dist', float)])),
                       filename)
                      for filename in disangoutfilenames]
    return disangdatalist


def get_min_max(disangoutfilenames):
    ''' takes a list of disangoutfiles and return a dict with the biggest and
    the smallest value'''
    disangdatalist = load_disangout(disangoutfilenames)
    maxs = [np.max(filedata['dist']) for (filedata, name) in disangdatalist]
    mins = [np.min(filedata['dist']) for (filedata, name) in disangdatalist]
    return dict(maximum=max(maxs), minimum=min(mins))


def arsplit(array, parts):
    '''takes numpy array
    splits array in "parts" parts but cuts of the division rest if the
    array length is not a multiple of the number of parts
    returns an array with the parts'''

    cut_off_to = (len(array)/parts)*parts

    return (i for i in np.split(array[:cut_off_to], parts))


def cumu_split(array, parts, start=0):
    '''takes numpy array
    splits array in parts "parts" and return the parts added up cumutively
    return a list with the parts
    start basically just cuts of the first start-1 parts before creating the
    cumulated parts of the array'''
    tempsplit = arsplit(array, parts)
    tempsplit = list(tempsplit)
    return [np.concatenate((tempsplit[start:i+1]))
            for i in range(start, len(tempsplit))]


def cut_beginning_of(array, parts, start=0):
    '''takes numpy array
    splits array in parts "parts" and return the parts added up cumutively
    return a list with the parts
    start basically just cuts of the first start-1 parts before creating the
    cumulated parts of the array'''
    tempsplit = arsplit(array, parts)
    tempsplit = list(tempsplit)
    return [np.concatenate((tempsplit[start:]))]


def first_and_last_windowdatnames(disangparams):
    ''' takes a list of disangparams and returns the disang.out filenames
    belonging to the leftmost and rightmost windows'''

    smallestmin = min([params['minpos'] for params in disangparams])
    largestmin = max([params['minpos'] for params in disangparams])

    for params in disangparams:
        if(params['minpos'] == smallestmin):
            leftmost = params['name']+'.out'

        if(params['minpos'] == largestmin):
            rightmost = params['name']+'.out'

    return dict(leftmost=leftmost, rightmost=rightmost)


def find_quarter(distdata):
    '''takes raw unsorted distance data and returns the position
    of the near points of 25% of the maximum of the histogram belonging
    to that distance data'''

    hist = np.histogram(distdata, bins=100)
    histmax = max(hist[0])
    aFourthHmax = histmax/4
    idxMax = (np.abs(hist[0]-histmax)).argmin()

    idxClosestToFourth_left = (np.abs(hist[0][:idxMax]-aFourthHmax)).argmin()
    idxClosestToFourth_right = (np.abs(hist[0][idxMax:]-aFourthHmax)).argmin()

    left = hist[1][:-1][idxClosestToFourth_left]
    right = hist[1][:-1][idxClosestToFourth_right+idxMax]
    return dict(left=left, right=right)


def find_half(distdata):
    '''takes raw unsorted distance data and returns the position
    of the near points of 25% of the maximum of the histogram belonging
    to that distance data'''

    hist = np.histogram(distdata, bins=100)
    histmax = max(hist[0])
    aFourthHmax = histmax/2
    idxMax = (np.abs(hist[0]-histmax)).argmin()

    idxClosestToFourth_left = (np.abs(hist[0][:idxMax]-aFourthHmax)).argmin()
    idxClosestToFourth_right = (np.abs(hist[0][idxMax:]-aFourthHmax)).argmin()

    left = hist[1][:-1][idxClosestToFourth_left]
    right = hist[1][:-1][idxClosestToFourth_right+idxMax]
    return dict(left=left, right=right)


def get_quarter_bounds(disangoutfilenames):
    ''' takes a list of disangoutfiles and return a dict with the left
    and right quarter of the max'''
    disangdatalist = load_disangout(disangoutfilenames)
    quarters = [find_quarter(filedata['dist'])
                for (filedata, name) in disangdatalist]
    left_qu = [i['left'] for i in quarters]
    right_qu = [i['right'] for i in quarters]
    return dict(maximum=max(right_qu), minimum=min(left_qu))


def get_half_bounds(disangoutfilenames):
    ''' takes a list of disangoutfiles and return a dict with the left
    and right quarter of the max'''
    disangdatalist = load_disangout(disangoutfilenames)
    quarters = [find_half(filedata['dist'])
                for (filedata, name) in disangdatalist]
    left_qu = [i['left'] for i in quarters]
    right_qu = [i['right'] for i in quarters]
    return dict(maximum=max(right_qu), minimum=min(left_qu))


def create_wham_bashfiles(boundaries, maindirname, subdirs, temperature,
                          bins=200):
    ''' create a wham.sh file in each subfolder with boundaries of
    given as a parameter'''

    os.chdir(maindirname)
    for subdir in subdirs:
        os.chdir(subdir)
        with open('wham.sh', 'w') as f:
            f.write('#!/bin/bash \n wham '+str(boundaries['minimum'])+' ' +
                    str(boundaries['maximum']) + ' '+str(bins)+' ' +
                    '0.00000001 '+temperature+' 0 metadatafile '
                    '../freefile_'+subdir)
        os.system('chmod 755 wham.sh; ./wham.sh')
        os.chdir('../')
    os.chdir('../')


def split(distfilenames, maindirname, parts, splitfunction,
          dist_extr_func=read_disang_angle_files_for_wham):

    fileending = os.path.splitext(distfilenames[0])[-1]
    ismtmdfile = (fileending == '.mtmd')
    if ismtmdfile:
        dist_extr_func = mtmd_for_wham

    '''dist_extr_func must create a list of dicts with name=disang.out filename
    mispos= pos of umbrella window forceconst= umbrella force'''
    # make filestructure and create empty metadatafiles
    os.system('mkdir '+maindirname)
    subdirs = []
    # read in .disangfiles and get the related disang.out data
    # split the data and then save the parts in the correct subfolders
    # create metadatafile with corect distances forceconstts(*2)
    # and disangoutfiles
    disangparams = dist_extr_func(distfilenames)

    for params in disangparams:

        filename = params['name']+'.out'
        absdisangfilename = os.path.abspath(params['name'])
        disangoutdata = load_one_disangout(filename)
        splitdata = splitfunction(array=disangoutdata, parts=parts)

        for k, datapart in enumerate(splitdata):
            os.chdir(maindirname)
            number = format(k, "02d")
            subdirname = 'part'+number
            subdirs.append(subdirname)
            os.system('mkdir -p '+subdirname)
            os.chdir(subdirname)
            data_part_filename = ('window' +
                                  str(params['minpos']).replace('.', '_') +
                                  'part'+number+fileending)

            datapart = np.column_stack((datapart['dist'], datapart['dist']))
            np.savetxt(fname=data_part_filename+'.out', X=datapart,
                       fmt=['%.3f', '%.3f'])
            with open(absdisangfilename, 'r') as source:
                with open(data_part_filename, 'w') as disang:
                    for line in source:
                        disang.write(line)

            subdirs = list(set(subdirs))
            os.chdir('../../')


def split_andwham_new(distfilenames, maindirname, parts, splitfunction,
                      boundaryfunc,
                      dist_extr_func=read_disang_angle_files_for_wham,
                      bins=200,
                      ex_file_or_groupfile=None):

    fileending = os.path.splitext(distfilenames[0])[-1]
    ismtmdfile = (fileending == '.mtmd')
    iscv_file = (fileending == '.cv_in')
    if ismtmdfile:
        dist_extr_func = mtmd_for_wham
    if iscv_file:
        dist_extr_func = cv_for_wham

    '''dist_extr_func must create a list of dicts with name=disang.out filename
    mispos= pos of umbrella window forceconst= umbrella force'''
    # make filestructure and create empty metadatafiles
    os.system('trash '+maindirname)
    os.system('mkdir '+maindirname)
    subdirs = []
    # read in .disangfiles and get the related disang.out data
    # split the data and then save the parts in the correct subfolders
    # create metadatafile with corect distances forceconstts(*2)
    # and disangoutfiles
    disangparams = dist_extr_func(distfilenames)
    disangparams = sorted(disangparams, key=lambda s: s['minpos'])

    params_first_file = disangparams[0]
    if os.path.splitext(params_first_file['name'])[1] == '.mtmd':
        ex_fileinfo = lib.read_exfile_groupfile(ex_file_or_groupfile)[0]

        mtmdfiledict_2 = lib.read_mtmdfile(params_first_file['name'])[1]
        if lib.read_mtmdfile(params_first_file['name'])[1] == dict():
            mtmdfiledict_2 = lib.read_mtmdfile(params_first_file['name'])[0]

        atom_mask = mtmdfiledict_2['mtmdmask'].strip('"')
        top = pytraj.load_topology(ex_fileinfo['topology'])
        number_of_atoms = len(top.select(atom_mask))

    for params in disangparams:

        filename = params['name']+'.out'
        absdisangfilename = os.path.abspath(params['name'])
        disangoutdata = load_one_disangout(filename)
        splitdata = splitfunction(array=disangoutdata, parts=parts)

        for k, datapart in enumerate(splitdata):
            os.chdir(maindirname)
            number = format(k, "02d")
            subdirname = 'part'+number
            subdirs.append(subdirname)
            os.system('mkdir -p '+subdirname)
            os.chdir(subdirname)
            distconfigfilename = ('window' +
                                  str(params['minpos']).replace('.', '_') +
                                  'part'+number+fileending)
            data_part_filename = distconfigfilename+'.out'

            datapart = np.column_stack((datapart['dist'], datapart['dist']))
            np.savetxt(fname=data_part_filename, X=datapart,
                       fmt=['%.3f', '%.3f'])
            with open(absdisangfilename, 'r') as source:
                with open(distconfigfilename, 'w') as disang:
                    for line in source:
                        disang.write(line)

            if os.path.splitext(params['name'])[1] == '.disang':
                with open('metadatafile', 'a') as f:
                    f.write("{0} {1} {2} \n".format(data_part_filename,
                                                    params['minpos'],
                                                    2*float(
                                                        params['forceconst'])))
            if os.path.splitext(params['name'])[1] == '.mtmd':
                mtmdfcon = number_of_atoms*float(params['forceconst'])
                with open('metadatafile', 'a') as f:
                    f.write("{0} {1} {2} \n".format(data_part_filename,
                                                    params['minpos'],
                                                    mtmdfcon))

            if os.path.splitext(params['name'])[1] == '.cv_in':
                mtmdfcon = params['forceconst']
                with open('metadatafile', 'a') as f:
                    f.write("{0} {1} {2} \n".format(data_part_filename,
                                                    params['minpos'],
                                                    mtmdfcon))
            os.chdir('../../')
    subdirs = list(set(subdirs))

    firstandlastWindows = first_and_last_windowdatnames(disangparams)
    leftmost = firstandlastWindows['leftmost']
    rightmost = firstandlastWindows['rightmost']
    boundaries = boundaryfunc([leftmost, rightmost])
    infilename = lib.no_ext(disangparams[0]['name'])+'.in'
    temp0 = lib.infilereadin(infilename)['temp0']
    create_wham_bashfiles(boundaries, maindirname, subdirs, temp0, bins)


def create_meta_files(distfilenames, boundaryfunc,
                      bins=200, dist_extr_func=None,
                      ex_file_or_groupfile=None):
    ''' this function creates metadata files for wham i have implemented
    a different version for .mtmd files that accounts for the different
    implementation in the amber ff and one for the implementation of the
    .disang variety'''

    disangparams = dist_extr_func(distfilenames)

    # check if we have .mtmd files and if so extract the number of atoms
    params_first_file = disangparams[0]
    if os.path.splitext(params_first_file['name'])[1] == '.mtmd':

        ex_fileinfo = lib.read_groupfile(ex_file_or_groupfile)[0]

        mtmdfiledict_2 = lib.read_mtmdfile(['name'])[1]
        atom_mask = mtmdfiledict_2['mtmdmask']
        traj = pytraj.iterload(ex_fileinfo['rstfile'],
                               ex_fileinfo['topology'])
        top = traj.top
        number_of_atoms = len(top.select(atom_mask))

    for params in disangparams:
        if os.path.splitext(params['name'])[1] == '.disang':
            with open('metadatafile', 'a') as f:
                f.write("{0} {1} {2} \n".format(params['name']+'.out',
                                                params['minpos'],
                                                2*float(params['forceconst'])))
        if os.path.splitext(params['name'])[1] == '.mtmd':
            mtmdfcon = number_of_atoms*float(params['forceconst'])
            with open('metadatafile', 'a') as f:
                f.write("{0} {1} {2} \n".format(params['name']+'.out',
                                                params['minpos'],
                                                mtmdfcon))


def split_andwahm(distfilenames, maindirname, parts, splitfunction,
                  boundaryfunc, dist_extr_func=read_disang_angle_files_for_wham,
                  bins=200):

    ismtmdfile = distfilenames[0].split('.')[-1] == 'mtmd'
    if dist_extr_func == read_disangfiles and ismtmdfile:
        raise TypeError('wrong dist_extrc funciton MAAAAAAN')

    '''dist_extr_func must create a list of dicts with name=disang.out filename
    mispos= pos of umbrella window forceconst= umbrella force'''
    # make filestructure and create empty metadatafiles
    os.system('mkdir '+maindirname)
    subdirs = []
    # read in .disangfiles and get the related disang.out data
    # split the data and then save the parts in the correct subfolders
    # create metadatafile with corect distances forceconstts(*2)
    # and disangoutfiles
    disangparams = dist_extr_func(distfilenames)
    for params in disangparams:

        filename = params['name']+'.out'
        disangoutdata = load_one_disangout(filename)
        splitdata = splitfunction(array=disangoutdata, parts=parts)

        for k, datapart in enumerate(splitdata):
            os.chdir(maindirname)
            number = format(k, "02d")
            subdirname = 'part'+number
            subdirs.append(subdirname)
            os.system('mkdir -p '+subdirname)
            os.chdir(subdirname)
            data_part_filename = ('window' +
                                  str(params['minpos']).replace('.', '_') +
                                  'part'+number)

            datapart = np.column_stack((datapart['dist'], datapart['dist']))
            np.savetxt(fname=data_part_filename, X=datapart, fmt=['%.3f',
                                                                  '%.3f'])
            with open('metadatafile', 'a') as f:
                f.write("{0} {1} {2} \n".format(data_part_filename,
                                                params['minpos'],
                                                2*float(params['forceconst'])))
            os.chdir('../../')
    subdirs = list(set(subdirs))

    firstandlastWindows = first_and_last_windowdatnames(disangparams)
    leftmost = firstandlastWindows['leftmost']
    rightmost = firstandlastWindows['rightmost']
    boundaries = boundaryfunc([leftmost, rightmost])
    infilename = lib.no_ext(distfilenames[0])+'.in'
    temp0 = lib.infilereadin(infilename)['temp0']
    create_wham_bashfiles(boundaries, maindirname, subdirs, temp0, bins)


def join_disangouts(targetfolder, list_w_dirs):

    os.mkdir(targetfolder)
    disangfilelists = [glob.glob(i+'*disang') for i in list_w_dirs]
    disangfiles = sum(disangfilelists, [])

    disangparams = read_disang_angle_files_for_wham(disangfiles)
    sorted_disanparams = sorted(disangparams, key=lambda s: s['minpos'])

    for k, par in enumerate(sorted_disanparams):
        is_eq_prev = (par['minpos'] == sorted_disanparams[k-1]['minpos'])

        if not is_eq_prev or k == 0:
            filename = 'joined_window_'+str(par['minpos']).replace('.', '_')
            if k == 0:
                with open(targetfolder+'/'+filename+'.in', 'w') as infilecopy:
                    with open(lib.no_ext(par['name'])+'.in', 'r') as source:
                        for line in source:
                            infilecopy.write(line)

            with open(targetfolder+'/'+filename+'.disang', 'w') as disang:
                with open(par['name'], 'r') as disang_source:
                    disang.write(disang_source.read())

            with open(targetfolder+'/'+filename+'.disang.out', 'w') as summary:
                source = load_one_disangout(par['name']+'.out')
                data_forfile = np.column_stack((source['dist'], source['dist']))
                np.savetxt(fname=summary, X=data_forfile, fmt=['%.3f',
                                                               '%.3f'])
        else:
            with open(targetfolder+'/'+filename+'.disang.out', 'a') as summary:
                source = load_one_disangout(par['name']+'.out')
                data_forfile = np.column_stack((source['dist'], source['dist']))
                np.savetxt(fname=summary, X=data_forfile, fmt=['%.3f',
                                                               '%.3f'])
    with open(targetfolder+'/docfile', 'w') as docfile:
        docfile.write('created from .disang.out files in the folders: \n'
                      ' '.join(list_w_dirs))


def join_disangouts_backup(targetfolder, list_w_dirs):

    os.mkdir(targetfolder)
    disangfilelists = [glob.glob(i+'*disang') for i in list_w_dirs]
    disangfiles = sum(disangfilelists, [])

    disangparams = read_disangfiles(disangfiles)
    sorted_disanparams = sorted(disangparams, key=lambda s: s['minpos'])

    for k, par in enumerate(sorted_disanparams):
        is_eq_prev = (par['minpos'] == sorted_disanparams[k-1]['minpos'])

        if not is_eq_prev or k == 0:
            filename = 'joined_window_'+str(par['minpos']).replace('.', '_')
            if k == 0:
                with open(targetfolder+'/'+filename+'.in', 'w') as infilecopy:
                    with open(lib.no_ext(par['name'])+'.in', 'r') as source:
                        for line in source:
                            infilecopy.write(line)

            with open(targetfolder+'/'+filename+'.disang', 'w') as disang:
                with open(par['name'], 'r') as disang_source:
                    disang.write(disang_source.read())

            with open(targetfolder+'/'+filename+'.disang.out', 'w') as summary:
                source = load_one_disangout(par['name']+'.out')
                data_forfile = np.column_stack((source['dist'], source['dist']))
                np.savetxt(fname=summary, X=data_forfile, fmt=['%.3f',
                                                               '%.3f'])
        else:
            with open(targetfolder+'/'+filename+'.disang.out', 'a') as summary:
                source = load_one_disangout(par['name']+'.out')
                data_forfile = np.column_stack((source['dist'], source['dist']))
                np.savetxt(fname=summary, X=data_forfile, fmt=['%.3f',
                                                               '%.3f'])
    with open(targetfolder+'/docfile', 'w') as docfile:
        docfile.write('created from .disang.out files in the folders: \n'
                      ' '.join(list_w_dirs))


def find_dist_mtmdout(line, rmsdnumber=2):
    '''finds a line with rmsd reference 2(second in mtmdfile) and returns the
    dist or  if none is found'''
    s = r'Current RMSD from reference   '+str(rmsdnumber)+':[\s]*([\d]*.?[\d]+)'
    founddist = re.findall(s, line)

    if founddist != []:
        return founddist[0]
    else:
        return ''


def mtmdout_dists(outfile, rmsdnumber=2):
    with open(outfile, 'r') as f:
        for line in f:
            result = find_dist_mtmdout(line, rmsdnumber)
            if result != '':
                yield result


def find_time_mtmdout(line):
    s = r'NSTEP =[\s]*([\d])+[\s]*TIME(PS)'
    foundtime = re.findall(s, line)

    if foundtime != []:
        return foundtime[0]
    else:
        return ''


def print_disangout_from_mtmtout(mtmdrun_outfile, rmsdnumber=2):
    ''' takes .out files from mtmdruns and prints the results into the standard
    table format as fileprefix.mtmd.out'''
    with open(mtmdrun_outfile, 'r') as outfile:
        with open(lib.no_ext(mtmdrun_outfile)+'.mtmd.out', 'w') as disangout:
            for line in outfile:
                searchresult = find_dist_mtmdout(line, rmsdnumber)
                if searchresult != '':
                    disangout.write(searchresult+' '+searchresult+'\n')


def mtmd_for_wham(filenames):
    '''creates the equivalent dictionaries to the disangparams from mtmdfile'''
    disangparams = []
    for filename in filenames:
        mtmddicts = lib.read_mtmdfile(filename)
        mtmddict = mtmddicts[1]
        if mtmddicts[1] == dict():
            mtmddict = mtmddicts[0]
        disangparams.append(dict(name=lib.no_ext(filename)+'.mtmd',
                                 minpos=mtmddict['mtmdrmsd'],
                                 forceconst=mtmddict['mtmdforce']))
    return disangparams


def cv_for_wham(filenames):
    disangparams = []
    for filename in filenames:
        cv_dict = lib.read_cv_file(filename)
        disangparams.append(dict(name=lib.no_ext(filename)+'.cv_in',
                                 minpos=cv_dict['anchor_position'],
                                 forceconst=cv_dict['anchor_strength']))
    return disangparams
