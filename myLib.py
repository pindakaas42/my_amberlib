#!/usr/bin/python
import os
import re
import pytraj as pt
import numpy as np
import string
from collections import namedtuple
# from functools import wraps
# ###general utilities:
# takes pico seconds and converts to ntlim if dt=0.004


def angstr_molar(C):
    '''converts particles per cubic angtroem to molar'''
    return (C)/(6.023*10**(-4))


def cmdlinedict(arguments):
    '''take a list of commanline arguments of the form file.ending and
    create a dictionary of the form {ending:'file.ending'}'''
    filendings = set()
    files = dict()
    for i in arguments:
        filendings.add(os.path.splitext(i)[-1])

    for arg in arguments:
        for ending in filendings:
            if os.path.splitext(arg)[-1] == ending:
                files[ending] = arg
    return files


def cmdlinedict_with_multiples(arguments):
    '''take a list of commanline arguments of the form file.ending and
    create a dictionary of the form {ending:'file.ending'}'''
    filendings = set()
    files = dict()
    for i in arguments:
        filendings.add(os.path.splitext(i)[-1])
    for ending in filendings:
        files[ending] = []

    for arg in arguments:
        for ending in filendings:
            if os.path.splitext(arg)[-1] == ending:
                files[ending].append(arg)
    return files


def pico_to_nstlim(time_in_ps):
    return time_in_ps*250
# outputs the .sh file with amber commands to execute the sim.
# takes the runname, the run before  and the file object to write to


def no_ext(filename):
    return os.path.splitext(filename)[0]


def quickprint(filename, text, append=False):
    if append is False:
        with open(filename, 'w') as f:
            f.write(text)
    if append is True:
        with open(filename, 'a+') as f:
            f.write(text)


def print_rundict(rundict):
    ''' print the dict in the format key=value\n while adding the
    directory thefiles that are listed in filetypes thusly ../parentdir/file'''
    with open('rundict', 'w') as f:
        for key, value in rundict.iteritems():
            f.write("{0}={1}".format(key, value)+"\n")


def read_rundict(rundictfile):
    rundict = dict()
    filetypevars = {'infilename', 'crd', 'pdb', 'prmtop', 'ex_file_name'}

    with open(rundictfile, 'r') as f:
        for line in f:
            line = line.strip()
            pos_of_eq = line.find('=')
            val = line[pos_of_eq+1::]
            key = line[:pos_of_eq:]
            if key in filetypevars:
                val = paths_for_here(rundictfile, val)
            rundict[key] = val
    return rundict


def paths_for_here(original_dir, relative_to_original):
    '''relative_to_original needs to be relative to original_dir, the function
    returns a version of relative_to_original that is relative to the current
    working directory
    '''
    heredircwd = os.getcwd()
    original_dir = os.path.realpath(original_dir)
    if not os.path.isdir(original_dir):
        original_dir = os.path.dirname(original_dir)
    os.chdir(original_dir)
    relative_to_original = os.path.realpath(relative_to_original)
    relativepath = os.path.relpath(relative_to_original, heredircwd)
    os.chdir(heredircwd)
    return relativepath


def paths_for_there(rel_to_here, new_working_dir):
    ''' takes a path relative to the current working directory and a new
    working directory and returns that path as relative to the new working
    directory'''
    rel_to_here = os.path.realpath(rel_to_here)
    return os.path.relpath(rel_to_here, new_working_dir)


def ch_to_lrzworkdir():
    '''this function just creates the workingdir in the lrzwworkdir and
    changes to that for the execution of the rest of the script if this
    is done every time relative pathing should still work'''

    dir_copy = os.path.split(os.path.realpath('.'))[-1]
    os.chdir('/home/paul/lrzhw_workdir')
    os.system('mkdir -p '+dir_copy)
    os.chdir(dir_copy)


def make_ex(filename):
    os.system('chmod 755 '+filename)


class freezefunc:
    def __init__(self, func):
        self.func = func
        self.funcargs = {}
        if not hasattr(func, '__call__'):
            raise TypeError(('give a function object as argument,'
                             'meaning f not f()'))

    def __call__(self, **kwargs):
        self.funcargs.update(kwargs)
        return self.func(**self.funcargs)

    def updateargs(self, **kwargs):
        self.funcargs.update(kwargs)

# the creation of a File Object will create a empty physical file


class File:
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'w') as f:
            f.write("")

    def write(self, text):
        with open(self.filename, 'w') as f:
            f.write(text)

    def append(self, text):
        with open(self.filename, 'a') as f:
            f.write(text)

    def get_filename(self):
        return self.filename

    def make_ex(self):
        os.system('chmod 755 '+self.get_filename())


def autocorr(x, t=0):
    mean = np.mean(x)
    var = np.var(x)
    return np.dot(x[:len(x)-t]-mean, x[t:]-mean)/(var*(len(x)-t))

vecautocorr = np.vectorize(autocorr, excluded=[0])
# ###infiles and mdsim orders


# readin files

def find_value_mdout(line, value):
    '''finds a line with rmsd reference 2(second in mtmdfile) and returns the
    dist or  if none is found'''
    s = r''+str(value)+r'[\s]*=[\s]*(-?[\d]*.?[\d]+)'
    foundvalue = re.findall(s, line)

    if foundvalue != []:
        return foundvalue[0]
    else:
        return None


def mdout_value(outfile, variables):
    '''takes an mdoutfile an returns the given variables, the variables
    are given as strings or a list of strings'''
    if type(variables) != list:
        variables = [variables]
    result_array = []
    values = []
    with open(outfile, 'r') as f:
        for line in f:
            for variable in variables:
                if variable in line:
                    result = find_value_mdout(line, variable)
                    if result is not None:
                        values.append(result)
            if len(values) == len(variables):
                result_array.append(values)
                values = []
    return result_array


def infilereadin(infile):
    infiledict = dict()
    with open(infile, 'r') as f:
        for line in f:
            if '=' in line:
                (key, val) = line.split('=')
                key = key.strip()
                val = ','.join(val.split(',')[:-1])
                infiledict[key] = val
            if '/' in line or '&end' in line:
                break
    return infiledict


def infileread_disang(infile):
    infiledict = dict()
    with open(infile, 'r') as f:
        for line in f:
            if 'DISANG' in line or 'DUMPAVE' in line:
                (key, val) = line.split('=')
                key = key.strip()
                val = val.strip()
                infiledict[key] = val
    return infiledict


# def infilereadin_multiple_namelists(infile):
#     '''reads all namelists in the infile and puts each in a different dict'''
#     namelists = dict()
#     with open(infile, 'r') as f:
#         for line in f:
#             if '&cntr' in line:
#                 namelist['&cntr'] = dict()
#
#     return infiledict


def read_mtmdfile(mtmdfile):
    mtmddict1 = dict()
    mtmddict2 = dict()
    count = 0
    with open(mtmdfile, 'r') as f:
        for line in f:
            if '&tgt' in line:
                count += 1
            if '=' in line and count == 1:
                (key, val) = line.split('=')
                val = val.split(',')[0]
                mtmddict1[key] = val
            if '=' in line and count == 2:
                (key, val) = line.split('=')
                val = val.split(',')[0]
                mtmddict2[key] = val
    mtmddicts = [mtmddict1, mtmddict2]
    for a_dict in mtmddicts:
        for key, val in a_dict.iteritems():
            if key == 'mtmdrmsd':
                a_dict[key] = float(val)
            if key == 'mtmdforce':
                a_dict[key] = float(val)
            if key == 'refin':
                a_dict[key] = paths_for_here(mtmdfile, val.strip('"'))

    return mtmddict1, mtmddict2


def read_cv_file(cv_file):
    cv_dict = infilereadin(cv_file)

    for key, val in cv_dict.iteritems():
        if key == 'anchor_strength':
            cv_dict[key] = float(val)
        if key == 'anchor_position':
            cv_dict[key] = float(val)
    return cv_dict


def read_groupfile_double_for_readgr_readex(groupfile):
    '''returns all the lines contained in a groupfile or ex_sim.sh file
    as a list of dicts with the keys given by
    strings = dict(infile=r'-i', outfile=r'-o', topology=r'-p',
    incrdfile=r'-c', rstfile=r'-r', trajectory=r'-x',
    reffile=r'-ref', mtmdfile=r'-mtmd') '''

    with open(groupfile, 'r') as f:
        groupdictlist = []
        strings = dict(infile=r'-i', outfile=r'-o', topology=r'-p',
                       incrdfile=r'-c', rstfile=r'-r', trajectory=r'-x',
                       reffile=r'-ref', mtmdfile=r'-mtmd')
        gline = dict()
        for line in f:
            for filetype in strings:
                searchresult = re.findall('-O.*'+strings[filetype] +
                                          r' ([./]*[\w./]+)', line)
                if searchresult != []:
                    gline[filetype] = searchresult[0]
            if gline != {}:
                groupdictlist.append(dict(gline))

        for row in groupdictlist:
                for key in row:
                    row[key] = paths_for_here(groupfile, row[key])
        return groupdictlist


def read_groupfile(gr_or_exfile):
    ''' if the gr_or_exfile is a an exfile this just calls read_groupfile
    on the groupfile found in the ex_sim.sh file'''
    groupdictlist = read_groupfile_double_for_readgr_readex(gr_or_exfile)
    if groupdictlist == []:
        with open(gr_or_exfile, 'r') as f:
            for line in f:
                searchresult = re.findall(r'-groupfile ([./]*[\w./]+) -rem',
                                          line)

        groupfile_rel_to_here = paths_for_here(gr_or_exfile,
                                               searchresult[0])
        return read_groupfile_double_for_readgr_readex(groupfile_rel_to_here)
    else:
        return groupdictlist


def find_residuenumbers(pdb_file):
    '''finds residuenumbers of first and second molecules and returns them
    as a dict'''
    TER_number = 0
    chain_1_res = set()
    chain_2_res = set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if 'TER' in line[0:6]:
                TER_number += 1
            elif 'ATOM' in line[0:6]:
                if TER_number == 0:
                    chain_1_res.add(int(line[22:26]))
                if TER_number == 1:
                    chain_2_res.add(int(line[22:26]))

    return dict(chain_1_res=chain_1_res, chain_2_res=chain_2_res)


def find_heavyatoms(inputpdb, nuc_aci_seq):
    '''old style find heavyatoms function '''
    num_of_bases = len(nuc_aci_seq)*2
    strain1 = []
    strain2 = []
    with open(inputpdb, 'r') as f:
        for line in f:
            colum = line.split()

            if colum[0] == 'ATOM' and int(colum[4]) <= num_of_bases:
                is_primed = "'" in colum[2]
                is_phosphor = "P" in colum[2]
                is_H = "H" in colum[2]

                if not is_primed and not is_phosphor and not is_H:
                    is_strain1 = int(colum[4]) <= num_of_bases/2
                    is_strain2 = \
                        not is_strain1 and int(colum[4]) <= num_of_bases

                    if is_strain1:
                        strain1.append(colum[1])
                    if is_strain2:
                        strain2.append(colum[1])
    return strain1, strain2


def find_heavyatoms_new(inputpdb):
    ''' finds heavy atoms of the bases was testet with the new dict format'''
    dnadict = find_DNA_pdb_dictlist(inputpdb)
    is_heavy_at = (lambda x: "'" not in x and "P" not in x and "H" not in x)
    Heavy_Atoms_chain1 = [i for i in dnadict
                          if i['chain'] == 1 and is_heavy_at(i['atom_name'])]
    Heavy_Atoms_chain2 = [i for i in dnadict
                          if i['chain'] == 2 and is_heavy_at(i['atom_name'])]
    return Heavy_Atoms_chain1, Heavy_Atoms_chain2


def find_atoms_in_DNA(inputpdb, should_contain, shouldnt_contain):
    '''should_contain and shouldnt_contain are iterables that
    contain the strings the atom names should and should not contain'''
    residues_DNA = find_residuenumbers(inputpdb)
    (chain_1_res, chain_2_res) = (residues_DNA['chain_1_res'],
                                  residues_DNA['chain_2_res'])
    Atoms_found = []
    with open(inputpdb, 'r') as f:
        for line in f:
            is_TER_or_END = ('END' in line[0:6]) or ('TER' in line[0:6])
            if not is_TER_or_END:
                residuenumber = int(line[22:26])
                is_DNA = residuenumber in chain_1_res | chain_2_res
                if is_DNA:
                    atom_name = line[12:16].strip()
                    atom_number = line[6:11].strip()

                    pos_cond = False
                    neg_cond = True
                    for s in should_contain:
                        pos_cond = (s in atom_name) or pos_cond

                    for s in shouldnt_contain:
                        if s != '':
                            neg_cond = (s not in atom_name) and neg_cond
                        if s == '':
                            neg_cond = True

                    if lists_in_strig(atom_name,
                                      should_contain,
                                      shouldnt_contain):

                        if residuenumber in chain_1_res:
                            Atoms_found.append(dict(atom_number=atom_number,
                                                    res=residuenumber,
                                                    chain=1,
                                                    atom_name=atom_name))
                        if residuenumber in chain_2_res:
                            Atoms_found.append(dict(atom_number=atom_number,
                                                    res=residuenumber,
                                                    chain=2,
                                                    atom_name=atom_name))
    return Atoms_found


def lists_in_strig(somestring, should_contain, shouldnt_contain):

    pos_cond = False
    neg_cond = True
    for s in should_contain:
        pos_cond = (s in somestring) or pos_cond

    for s in shouldnt_contain:
        if s != '':
            neg_cond = (s not in somestring) and neg_cond
        if s == '':
            neg_cond = True
    return pos_cond and neg_cond


def find_DNA_pdb_dictlist(inputpdb):
    '''returns a list of all the atoms in the first two chain in a .pdb
    each in a dictionary containing atomnumber, resnumber, cahin number,
    and the full atom name'''
    residues_DNA = find_residuenumbers(inputpdb)
    (chain_1_res, chain_2_res) = (residues_DNA['chain_1_res'],
                                  residues_DNA['chain_2_res'])
    Atoms_found = []
    with open(inputpdb, 'r') as f:
        for line in f:
            is_TER_or_END = ('END' in line[0:6]) or ('TER' in line[0:6])
            if not is_TER_or_END:
                residuenumber = int(line[22:26])
                is_DNA = residuenumber in chain_1_res | chain_2_res
                if is_DNA:
                    atom_name = line[12:16].strip()
                    atom_number = line[6:11].strip()

                    if residuenumber in chain_1_res:
                        Atoms_found.append(dict(atom_number=atom_number,
                                                res=residuenumber,
                                                chain=1,
                                                atom_name=atom_name))
                    if residuenumber in chain_2_res:
                        Atoms_found.append(dict(atom_number=atom_number,
                                                res=residuenumber,
                                                chain=2,
                                                atom_name=atom_name))
    return Atoms_found


def find_p_atoms(inputpdb):
    residues_DNA = find_residuenumbers(inputpdb)

    (chain_1_res, chain_2_res) = (residues_DNA['chain_1_res'],
                                  residues_DNA['chain_2_res'])
    P_Atoms_in_strain1 = []
    P_Atoms_in_strain2 = []
    with open(inputpdb, 'r') as f:
        for line in f:
            is_TER_or_END = ('END' in line[0:6]) or ('TER' in line[0:6])
            if not is_TER_or_END:
                residuenumber = int(line[22:26])
                is_DNA = residuenumber in chain_1_res | chain_2_res
                if is_DNA:
                    atom_name = line[12:16].strip()
                    atom_number = line[6:11].strip()
                    is_phosphor = ('P' == atom_name)
                    if is_phosphor:
                        if residuenumber in chain_1_res:
                            P_Atoms_in_strain1.append(atom_number)
                        if residuenumber in chain_2_res:
                            P_Atoms_in_strain2.append(atom_number)
    return P_Atoms_in_strain1, P_Atoms_in_strain2


def find_p_atoms_pytraj(topology):
    '''returns a list of all the atoms in the first two chains in a topology
    each in a dictionary containing atomnumber, resnumber, cahin number,
    and the full atom name'''
    atomlist = get_atomdicts_pytraj(topology)
    P_Atoms_in_strain1 = [i['atom_number'] for i in atomlist
                          if i['atom_name'] == 'P' and i['chain'] == 1]
    P_Atoms_in_strain2 = [i['atom_number'] for i in atomlist
                          if i['atom_name'] == 'P' and i['chain'] == 2]

    return P_Atoms_in_strain1, P_Atoms_in_strain2


def create_mylist_only_dna(pytrajdict):
    '''pytrajdict being created via top = pt.load_topology(topologyfile)
    pytrajdict = top.to_dict()'''
    atom_names = pytrajdict['atom_name']
    resids = pytrajdict['resid']
    chainids = pytrajdict['mol_number']
    mydictlist = []
    for number, (name, res, chain) in enumerate(zip(atom_names,
                                                    resids,
                                                    chainids)):
        if chain+1 == 3:
            break
        mydictlist.append(dict(atom_number=str(number+1),
                               atom_name=str(name),
                               res=res+1, chain=chain+1))

    return mydictlist


def get_atomdicts_pytraj(topology):
    '''pytrajdict being created via top = pt.load_topology(topologyfile)
    pytrajdict = top.to_dict()'''
    top = pt.load_topology(topology)
    pytrajdict = top.to_dict()
    atom_names = pytrajdict['atom_name']
    resids = pytrajdict['resid']
    chainids = pytrajdict['mol_number']
    mydictlist = []
    for number, (name, res, chain) in enumerate(zip(atom_names,
                                                    resids,
                                                    chainids)):
        if chain+1 == 3:
            break
        mydictlist.append(dict(atom_number=str(number+1),
                               atom_name=str(name),
                               res=res+1, chain=chain+1))

    return mydictlist


def get_atom_num_and_pos(topology, mask, trajectory=None):
    '''retrun an array with a list of atom positions and numbers (starting at 1)
    corresponding to a cpptraj mask rerurns a tuple of
    coords and atom_numbers'''
    if isinstance(topology, basestring) == True:
        topology = pt.load_topology(topology)

    atom_numbers = topology.select(mask)+1
    if trajectory is not None:
        if isinstance(trajectory, basestring) == True:
            trajectory = pt.iterload(trajectory, topology)
        coords = np.array(trajectory[mask][-1].xyz)
    else:
        coords = None
    return coords, atom_numbers
# return config files as string


def get_fortranconf(infile, namelist='cntrl'):
    ''' returns the formated content of an infile from a dict that contains
    key=value with the contents of the infile ends lists with &end now hope
    that works too'''
    f = "\n&"+namelist+"\n"
    for key, value in infile.iteritems():
            if key != "runtime" and value is not '':
                f += "{0}={1}".format(key, value)+",\n"
    f += "&end\n"
    return f


get_infile = get_fortranconf


def get_infile_umb(disangfilename, disang_out):
            return("&wt type='DUMPFREQ',"
                   " istep1=50,/\n"  # timesteps of recording distance
                   "&wt type='END',/\n"
                   "DUMPAVE="+disang_out+" \n"
                   "DISANG="+disangfilename+" \n"
                   )


def get_colvar(topology, masks, trajectory=None,
               cv_type="'MULTI_RMSD'"):
    if cv_type == "'MULTI_RMSD'" and trajectory is None:
        raise ValueError('we need a trajectory for reference coords please')
    if isinstance(masks, basestring) == True:
        masks = [masks]

    atom_pos_nums = [get_atom_num_and_pos(topology, mask, trajectory)
                     for mask in masks]

    cv_i = ''
    for (coords, atom_numbers) in atom_pos_nums:
        for number in atom_numbers:
            cv_i += str(number)+', '
        cv_i += '0, '

    cv_ni = sum([len(atom_num)+1 for (coords, atom_num) in atom_pos_nums])

    cv_r = ''
    for (coords, atom_numbers) in atom_pos_nums:
        if coords is not None:
            for x, y, z in coords:
                cv_r += str(x)+', '+str(y)+', '+str(z)+', '
    cv_nr = str(3*sum([len(atom_num) for (coords, atom_num) in atom_pos_nums]))

    colvardict = dict(cv_type=cv_type,
                      cv_ni=cv_ni,
                      cv_nr=cv_nr,
                      cv_i=cv_i,
                      cv_r=cv_r,
                      )
    return colvardict


def get_disang_simple(dist1, dist2,
                      forceconst1, forceconst2,
                      atoms1, atoms2):
    s = ("&rst \n"
         "iat=-1,-1 \n"
         # declares that we want to define restraints
         # between two groups of Atoms, not single Atoms

         "iresid=0,irstyp=0,ifvari=0,ninc=0,imult=0,ir6=0,ifntyp=0, \n"

         "r1=0,r2="+dist1+",r3="+dist2+",r4=999, \n"

         "rk2="+forceconst1+",rk3="+forceconst2+", \n"
         )
    # the force constants are always directly k/2
    # meaning the actual force is 2 times rk
    s += ("igr1= "+atoms1+" \n"
          "igr2= "+atoms2+" \n"
          "/ \n"
          )
    return s


def get_disang(dist, forceconst, topology, find_atoms):
    dist = str(float(dist))
    forceconst = str(float(forceconst))
    igr1 = ''
    igr2 = ''
    for i in find_atoms(topology)[0]:
        igr1 += str(i)+", "
    for i in find_atoms(topology)[1]:
        igr2 += str(i)+", "
        return get_disang_simple(dist1=dist, dist2=dist,
                                 forceconst1=forceconst, forceconst2=forceconst,
                                 atoms1=igr1, atoms2=igr2)


def get_disang_angle_dist(angle_or_dist, forceconst, atomgroups):
    angle_or_dist = str(float(angle_or_dist))
    forceconst = str(float(forceconst))
    s = ("\n&rst \n")
    s += ("iat=")
    for i in range(len(atomgroups)):
        s += "-1,"
    s += "\n"
    # declares that we want to define restraints
    # between two groups of Atoms, not single Atoms

    s += ("iresid=0,irstyp=0,ifvari=0,ninc=0,imult=0,ir6=0,ifntyp=0, \n")

    if len(atomgroups) == 2:
        s += ("r1=0,r2="+angle_or_dist+",r3="+angle_or_dist+",r4=999, \n"
              "rk2="+forceconst+",rk3="+forceconst+", \n")
    if len(atomgroups) in [3, 4]:
        s += ("R0="+angle_or_dist+", \n"
              "K0="+forceconst+", \n")

    for k, atoms in enumerate(atomgroups):
        s += ('igr'+str(k+1)+'='+','.join(atoms)+', \n'
              )

    s += '/ \n'
    return s


def get_mtmdtxt(refin, mtmdforce, mtmdrmsd, mtmdmask):
    refin = refin.strip('"')
    mtmdmask = mtmdmask.strip('"')
    mtmdstr = ('&tgt \n'
               'refin="'+refin+'", \n'
               'mtmdforce='+str(mtmdforce)+', \n'
               'mtmdrmsd='+str(mtmdrmsd)+', \n'
               'mtmdmask="'+mtmdmask+'", \n'
               '/ \n')
    return mtmdstr


def get_exsim_cmd(infile, inputcrd, outputpref, topologyfile,
                  mtmdfile=None, ref=None, pbsa=False):

    if ref is None:
        ref = inputcrd
    outputpref = no_ext(outputpref)

    excmd = (' -O -i '+outputpref+'.in '
             '-o '+outputpref+'.out '
             '-p '+topologyfile + ' '
             '-c '+inputcrd + ' ')
    if pbsa is False:
        excmd += ('-r '+outputpref+'.rst7 ')

    if ('imin' not in infile or int(infile['imin']) == 0) and pbsa is False:
        excmd += ('-x '+outputpref+'.nc ')

    if 'itgtmd' in infile and int(infile['itgtmd']) == 2:
        excmd += ('-mtmd '+mtmdfile)

    if 'ntr' in infile and int(infile['ntr']) == 1:
        excmd += ('-ref '+ref)

    return (excmd+'\n')


def get_rex_qsubfiletxt(nodes, ppn, name):
    text = ('#!/bin/bash\n'
            '#PBS -N '+name+'\n'
            '#PBS -l nodes='+str(nodes)+':ppn='+str(ppn)+'\n'
            '#PBS -l walltime=999:00:00\n'
            '#PBS -q ethernet\n'
            '#PBS -M paul_westphaelinger@ewetel.net\n'
            'source $HOME/.bashrc\n'
            'cd $PBS_O_WORKDIR \n'
            'cat $PBS_NODEFILE > nodefile\n'
            'set -x \n')
    return text


def get_lrz_subfiletxt(nodes, ppn, jobclass, time):
    dirname = os.path.split(os.path.realpath('.'))[-1]
    text = ('#!/bin/bash\n'
            '#nocomment\n'
            '#@ job_type = mpich\n'
            '#@ class = '+jobclass+' \n'
            '#@ node = '+str(nodes)+'\n'
            '#@ tasks_per_node = '+str(ppn)+'\n'
            '#@ island_count = 1 \n'
            '#@ wall_clock_limit = '+str(time)+'\n'
            '#@ job_name = '+str(dirname)+'\n'
            '#@ network.MPI = sn_all,not_shared,us\n'
            '#@ output = job.$(schedd_host).$(jobid).out\n'
            '#@ error =  job.$(schedd_host).$(jobid).err\n'
            '#@ notification=always\n'
            '#@ notify_user=paul_westphaelinger@ewetel.net\n'
            '#@ queue\n'
            '. /etc/profile\n'
            '. /etc/profile.d/modules.sh\n'
            'module unload mpi.ibm\n'
            'module load mpi.intel/5.1\n'
            'module load amber/14\n'
            'MPICMD="mpiexec -n $LOADL_TOTAL_TASKS"\n'
            'WORKDIR="$WORK/mtmdtest/..."\n'
            'HOMEDIR="$LOADL_STEP_INITDIR"\n'
            )
    return text


def get_slurm_subfiletxt(nodes, ntasks, gpus='4', percore='2', restr=None,
                         time='00:01:00'):
    name = string.join([i for i in string.split(os.path.realpath('.'), '/')
                        if not any(['cow' in i,
                                    'paul' in i,
                                    'home' in i])], '/')
    text = ('#!/bin/bash\n'
            '#SBATCH --job-name='+str(name)+'\n'
            '#SBATCH --nodes='+str(nodes)+'\n'
            '#SBATCH --partition=barracuda \n'
            '#SBATCH --gres=gpu:'+str(gpus)+'\n'
            '#SBATCH --time='+time+'\n')
    if ntasks >= 16:
        text += '#SBATCH --ntasks=16\n'
    if ntasks < 16:
        text += '#SBATCH --ntasks='+str(ntasks)+'\n'

    if restr is not None:
            text += '#SBATCH --constraint="'+str(restr)+'"\n'

    text += 'module load amber/16\n'
    return text


# printing functions for differnt run types using different amber functions
def simple_conffiles(simrun, topologyfile, infiledict,
                     outputpref, inputcrd, ex_simfile):
    # print infile
    quickprint(filename=outputpref+'.in', text=get_infile(infiledict))


def new_disangrun(simrun, topologyfile, infiledict,
                  outputpref, inputcrd, ex_simfile,
                  disangfilename, disangparams):

    simple_conffiles(simrun, topologyfile, infiledict,
                     outputpref, inputcrd, ex_simfile)
    infilename = outputpref+'.in'
    quickprint(filename=infilename,
               text=get_infile_umb(disangfilename,
                                   disang_out=disangfilename+'.out'),
               append=True)
    # print exfile
    ex_filetext = get_exsim_cmd(infile=infiledict,
                                inputcrd=inputcrd,
                                outputpref=outputpref,
                                topologyfile=topologyfile)
    ex_simfile.append(simrun+ex_filetext)

    # print disang
    disangfiletext = get_disang_simple()
    quickprint(filename=no_ext(infilename)+'.disang', text=disangfiletext)


def std_runprint(simrun, outputpref, topologyfile, infiledict, incrd,
                 ex_simfile, rundict=None):

    quickprint(filename=outputpref+'.in', text=get_infile(infiledict))

    ex_simfile.append(simrun + get_exsim_cmd(infiledict, incrd,
                                             outputpref,
                                             topologyfile))

    if rundict is not None:
        rundict['ex_file_name'] = ex_simfile.get_filename()
        rundict['crd'] = outputpref+'.rst7'
        rundict['prmtop'] = topologyfile


def std_disang_run_pr(simrun, infilename, disangfilename, inputcrd,
                      ex_file, infiledict, dist, forceconst,
                      topology, inputpdb=None,
                      find_atoms=None, ref=None, rundict=None,
                      atomgroups=None):

    # print disang
    disangfiletext = get_disang(dist=dist, forceconst=forceconst,
                                topology=topology, find_atoms=find_atoms)
    quickprint(filename=no_ext(infilename)+'.disang', text=disangfiletext)

    # print infile
    infiletext = get_infile(infile=infiledict)
    quickprint(filename=infilename,
               text=infiletext+get_infile_umb(disangfilename,
                                              disang_out=disangfilename+'.out'))

    # print ex_ file
    ex_filetext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd, ref=ref,
                                outputpref=infilename,
                                topologyfile=topology)
    ex_file.append(simrun+ex_filetext)

    if rundict is not None:
        rundict['ex_file_name'] = ex_file.get_filename()
        rundict['crd'] = no_ext(infilename)+'.rst7'
        rundict['prmtop'] = topology


def std_remd_print(infilename, disangfilename, groupfile, inputcrd,
                   dist, forceconst, infiledict, inputpdb=None, topology=None,
                   find_atoms=None):

    # print disang
    disangfiletext = get_disang(dist=dist, forceconst=forceconst,
                                topology=topology, find_atoms=find_atoms)
    quickprint(filename=no_ext(infilename)+'.disang', text=disangfiletext)

    # print infile
    infiletext = get_infile(infile=infiledict)
    quickprint(filename=infilename,
               text=infiletext+get_infile_umb(disangfilename,
                                              disang_out=disangfilename+'.out'))

    # print groupfile
    groupfiletext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                  outputpref=infilename,
                                  topologyfile=topology)
    groupfile.append(groupfiletext)


def std_mtmd_run_pr(simrun, outputpref, inputcrd, ex_file, infiledict,
                    topology, mtmdparams1, mtmdparams2,
                    rundict=None):

    # print mtmd
    if mtmdparams2 == dict():
        mtmdfiletxt = get_mtmdtxt(**mtmdparams1)
    else:
        mtmdfiletxt = get_mtmdtxt(**mtmdparams1)+get_mtmdtxt(**mtmdparams2)
    quickprint(filename=outputpref+'.mtmd', text=mtmdfiletxt)

    # print infile
    infiletext = get_infile(infile=infiledict)
    quickprint(filename=outputpref+'.in',
               text=infiletext)

    # print ex_ file
    ex_filetext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                outputpref=outputpref,
                                topologyfile=topology,
                                mtmdfile=outputpref+'.mtmd')
    ex_file.append(simrun+ex_filetext)
    if rundict is not None:
        rundict['crd'] = outputpref+'.crd'
        rundict['ex_file_name'] = ex_file.get_filename()


def std_mtmd_remd_print(outputpref, groupfile, inputcrd,
                        infiledict, topology,
                        mtmdparams1, mtmdparams2, rundict=None):

    # print mtmd
    if mtmdparams2 == dict():
        mtmdfiletxt = get_mtmdtxt(**mtmdparams1)
    else:
        mtmdfiletxt = get_mtmdtxt(**mtmdparams1)+get_mtmdtxt(**mtmdparams2)
    quickprint(filename=outputpref+'.mtmd', text=mtmdfiletxt)

    # print infile
    infiletext = get_infile(infile=infiledict)
    quickprint(filename=outputpref+'.in',
               text=infiletext)

    # print groupfile
    groupfiletext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                  outputpref=outputpref,
                                  topologyfile=topology,
                                  mtmdfile=outputpref+'.mtmd')
    groupfile.append(groupfiletext)
    if rundict is not None:
        rundict['crd'] = outputpref+'.crd'
        rundict['ex_file_name'] = groupfile.get_filename()


def std_mtmd_disang_run_pr(simrun, outputpref, inputcrd, ex_file, infiledict,
                           topology, mtmdparams1, mtmdparams2,
                           disangparamlist, rundict=None):

    # print disang
    count = 0
    for params in disangparamlist:
        disangfiletext = get_disang_angle_dist(**params)

        if count == 0:
            quickprint(filename=outputpref+'.disang', text=disangfiletext,
                       append=False)
        if count > 0:
            quickprint(filename=outputpref+'.disang', text=disangfiletext,
                       append=True)
        count += 1

    # print mtmd
    mtmdfiletxt = get_mtmdtxt(**mtmdparams1)+get_mtmdtxt(**mtmdparams2)
    quickprint(filename=outputpref+'.mtmd', text=mtmdfiletxt)

    # print infile
    infiletext = get_infile(infile=infiledict)
    quickprint(filename=outputpref+'.in',
               text=infiletext+get_infile_umb(outputpref+'.disang',
                                              outputpref+'.disang.out'))

    # print ex_ file
    ex_filetext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                outputpref=outputpref,
                                topologyfile=topology,
                                mtmdfile=outputpref+'.mtmd')
    ex_file.append(simrun+ex_filetext)
    if rundict is not None:
        rundict['crd'] = outputpref+'.crd'
        rundict['ex_file_name'] = ex_file.get_filename()


def std_mtmd_disang_remd(simrun, outputpref, inputcrd, groupfile, infiledict,
                         topology, mtmdparams1, mtmdparams2, disangparamlist,
                         rundict=None):

    # print disang
    count = 0
    for params in disangparamlist:
        disangfiletext = get_disang_angle_dist(**params)

        if count == 0:
            quickprint(filename=outputpref+'.disang', text=disangfiletext,
                       append=False)
        if count > 0:
            quickprint(filename=outputpref+'.disang', text=disangfiletext,
                       append=True)
        count += 1

    # print mtmd
    mtmdfiletxt = get_mtmdtxt(**mtmdparams1)+get_mtmdtxt(**mtmdparams2)
    quickprint(filename=outputpref+'.mtmd', text=mtmdfiletxt)

    # print infile
    infiletext = get_infile(infile=infiledict)
    quickprint(filename=outputpref+'.in',
               text=infiletext+get_infile_umb(outputpref+'.disang',
                                              outputpref+'.disang.out'))

    # print groupfile
    groupfiletext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                  outputpref=outputpref,
                                  topologyfile=topology,
                                  mtmdfile=outputpref+'.mtmd')
    groupfile.append(groupfiletext)

    if rundict is not None:
        rundict['crd'] = outputpref+'.crd'
        rundict['groupfile'] = groupfile.get_filename()


def std_Tremd_print(infilename, groupfile, inputcrd,
                    infiledict, topology,):

    # print infile
    infiletext = get_infile(infile=infiledict)
    quickprint(filename=infilename,
               text=infiletext)

    # print groupfile
    groupfiletext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                  outputpref=infilename,
                                  topologyfile=topology)
    groupfile.append(groupfiletext)


umbr_param = namedtuple('umbr_param', ['dist', 'fcon', 'masks', 'ref_traj',
                                       'cv_type'])


def std_nfe_run_pr(simrun, outputpref, inputcrd, ex_file, infiledict,
                   topology, umbr_params, rundict=None):
    colvar_filetxt = ''
    for params in umbr_params:
        colvar_dict = dict()
        colvar_dict.update(get_colvar(topology, params.masks,
                                      trajectory=params.ref_traj,
                                      cv_type=params.cv_type))
        if params.dist is not None:
            colvar_dict.update(anchor_position=params.dist,
                               anchor_strength=params.fcon)
        colvar_filetxt += get_fortranconf(colvar_dict, 'colvar')

    cv_filename = outputpref+'.cv_in'
    quickprint(filename=cv_filename, text=colvar_filetxt)

    # print infile
    infile = File(outputpref+'.in')
    infiletext = get_fortranconf(infile=infiledict)
    infile.write(text=infiletext)
    # print infile the colvar part
    colvar_infilepart_dict = dict(output_file="'"+outputpref+".cv_in.out'",
                                  output_freq=50,
                                  cv_file="'"+cv_filename+"'")
    colvar_infile_part = get_fortranconf(colvar_infilepart_dict, namelist='pmd')
    infile.append(text=colvar_infile_part)

    # print groupfile
    ex_filetext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                outputpref=outputpref,
                                topologyfile=topology,)

    ex_file.append(simrun+ex_filetext)
    if rundict is not None:
        rundict['crd'] = outputpref+'.crd'
        rundict['ex_file_name'] = ex_file.get_filename()


def std_nfe_remd_print(outputpref, groupfile, inputcrd,
                       infiledict, topology,
                       umbr_params, rundict=None):
    colvar_filetxt = ''
    for params in umbr_params:
        colvar_dict = dict()
        colvar_dict.update(get_colvar(topology, params.masks,
                                      trajectory=params.ref_traj,
                                      cv_type=params.cv_type))
        if params.dist is not None:
            colvar_dict.update(anchor_position=params.dist,
                               anchor_strength=params.fcon)
        colvar_filetxt += get_fortranconf(colvar_dict, 'colvar')

    cv_filename = outputpref+'.cv_in'
    quickprint(filename=cv_filename, text=colvar_filetxt)

    # print infile
    infile = File(outputpref+'.in')
    infiletext = get_fortranconf(infile=infiledict)
    infile.write(text=infiletext)
    # print infile the colvar part
    colvar_infilepart_dict = dict(output_file="'"+outputpref+".cv_in.out'",
                                  output_freq=50,
                                  cv_file="'"+cv_filename+"'")
    colvar_infile_part = get_fortranconf(colvar_infilepart_dict, namelist='pmd')
    infile.append(text=colvar_infile_part)

    # print groupfile
    groupfiletext = get_exsim_cmd(infile=infiledict, inputcrd=inputcrd,
                                  outputpref=outputpref,
                                  topologyfile=topology,)

    groupfile.append(groupfiletext)
    if rundict is not None:
        rundict['crd'] = outputpref+'.crd'
        rundict['ex_file_name'] = groupfile.get_filename()


def pbsa_run(simrun, outputpref, topology, infiletuples, incrd,
             ex_simfile, rundict=None):

    quickprint(filename=outputpref+'.in',
               text=get_infile(infiletuples[0][0],
                               namelist=infiletuples[0][1]),)

    for infiledict, namelist in infiletuples[1:]:
        quickprint(filename=outputpref+'.in',
                   text=get_infile(infiledict, namelist=namelist),
                   append=True)

    ex_simfile.append(simrun + get_exsim_cmd(infiledict, incrd,
                                             outputpref,
                                             topology,
                                             pbsa=True))


# topologystuff


def print_topology(outputprefix, nuc_aci_seq, boxsize, salt, watermodel=None):

    output_pdb = nuc_aci_seq+"_vac.pdb"
    nabfilename = nuc_aci_seq+".nab"

    gen_nabfile(nuc_aci_seq=nuc_aci_seq,
                output_pdb=output_pdb,
                nabfilename=nabfilename)

    output_prmtop = outputprefix+".prmtop"
    output_inpcrd = outputprefix+".inpcrd"

    gen_leapscript(input_vacpdb=output_pdb,
                   output_prmtop=output_prmtop,
                   output_inpcrd=output_inpcrd,
                   leapscript_name="leap_script_"+outputprefix,
                   boxsize=boxsize,
                   additionalsalt=salt,
                   watermodel=watermodel)
    # HMassRepartition gives out a projectname.prmtop because this is what
    # the ex_sim file generator expects as the topology file
    HMassRepartition(input_prmtop=output_prmtop,
                     output_prmtop=output_prmtop)
    return locals()


# write the .nab file to generate vacuum pdbfiles of nuc_aci_seq
def gen_nabfile(nuc_aci_seq, output_pdb, nabfilename):
    with open(nabfilename, 'w') as f:
        f.write('molecule m;\n'
                'm = fd_helix( "abdna", "'+nuc_aci_seq+'" , "dna" );\n'
                'putpdb( "'+output_pdb+'", m, "-wwpdb");\n'
                )
    os.system('nab '+nabfilename+'\n'
              './a.out > /dev/null\n')


# write the script file for tleap
# standard salt for 21A box is 14 total ions
def gen_leapscript(input_vacpdb, output_pdb, output_prmtop,
                   output_inpcrd, leapscript_name,
                   boxsize, additionalsalt=None, watermodel=None,
                   residues_to_remove=None):
    with open(leapscript_name, 'w') as f:
        f.write('source leaprc.ff14SB\n')

        f.write('loadamberparams frcmod.ionsjc_tip3p\n')

        if watermodel is 'opc':
            f.write('loadamberparams frcmod.opc \n')

        f.write('dna1= loadpdb "'+input_vacpdb+'" \n'
                'savepdb dna1 pre_solvation_'+input_vacpdb+' \n')
        if residues_to_remove is not None:
            for k, res in enumerate(residues_to_remove):
                f.write('remove dna1 dna1.'+str(res)+'\n')

        if watermodel is None:
                f.write('solvateoct dna1 TIP3PBOX '+str(boxsize)+'\n')
        if watermodel is 'opc':
                f.write('solvateoct dna1 OPCBOX '+str(boxsize)+'\n')
        if watermodel is 'igb8':
                f.write('set default pbradii mbondi3\n')

        if watermodel != 'igb8':
            f.write('addions dna1 K+ 0\n')

        if additionalsalt is not None:
            additionalsalt = str(additionalsalt/2)
            f.write('addions dna1 K+ '+additionalsalt +
                    ' Cl- '+additionalsalt+' \n')

        f.write('savepdb dna1 '+output_pdb+' \n'
                'saveamberparm dna1 '+output_prmtop+' '
                ' '+output_inpcrd+' \n'
                'quit\n')
    os.system('tleap  -s -I $AMBERHOME/dat/leap/cmd/oldff \
              -f '+leapscript_name+' > /dev/null')
    os.system('mv leap.log '+leapscript_name+'.log')

    return dict(outputpdb_presolv='pre_solvation_'+input_vacpdb,
                outputpdb=input_vacpdb)


def gen_leapscript_amber15(input_vacpdb, output_pdb, output_prmtop,
                           output_inpcrd, leapscript_name,
                           boxsize, additionalsalt=None, watermodel=None,
                           residues_to_remove=None):
    with open(leapscript_name, 'w') as f:
        f.write('source leaprc.ff14SB\n')

        f.write('loadamberparams frcmod.ionsjc_tip3p\n')

        if watermodel is 'opc':
            f.write('loadamberparams frcmod.opc \n')

        f.write('dna1= loadpdb "'+input_vacpdb+'" \n'
                'savepdb dna1 pre_solvation_'+input_vacpdb+' \n')
        if residues_to_remove is not None:
            for k, res in enumerate(residues_to_remove):
                f.write('remove dna1 dna1.'+str(res)+'\n')

        if watermodel is None:
                f.write('solvateoct dna1 TIP3PBOX '+str(boxsize)+'\n')
        if watermodel is 'opc':
                f.write('solvateoct dna1 OPCBOX '+str(boxsize)+'\n')

        f.write('addions dna1 K+ 0\n')

        if additionalsalt is not None:
            additionalsalt = str(additionalsalt/2)
            f.write('addions dna1 K+ '+additionalsalt +
                    ' Cl- '+additionalsalt+' \n')

        f.write('savepdb dna1 '+output_pdb+' \n'
                'saveamberparm dna1 '+output_prmtop+' '
                ' '+output_inpcrd+' \n'
                'quit\n')
    os.system('tleap  -s -f '+leapscript_name+' > /dev/null')
    os.system('mv leap.log '+leapscript_name+'.log')

    return dict(outputpdb_presolv='pre_solvation_'+input_vacpdb,
                outputpdb=input_vacpdb)


def load_and_save_leapscript(input_pdb, output_pdb, output_prmtop, output_crd):
    '''load a inputpdb and saves as outputpdb'''
    script = ("source leaprc.ff14SB \n"
              "loadamberparams frcmod.ionsjc_tip3p \n"
              'dna1= loadpdb "'+input_pdb+'" \n'
              'saveamberparm dna1 '+output_prmtop+' '+output_crd+' \n'
              'savepdb dna1 '+output_pdb+' \n'
              'quit \n')
    return script


def remove_residues(residues_to_remove, input_pdb, output_pdb):

    tleap_mod_script = ('source leaprc.ff14SB \n' +
                        'dna = loadpdb '+input_pdb+'\n')

    for k, res in enumerate(residues_to_remove):
        if k < len(residues_to_remove)-1:
            if (res != residues_to_remove[k+1]-1):
                tleap_mod_script += ('remove dna dna.'+str(res+1)+'.P\n'
                                     'remove dna dna.'+str(res+1)+'.OP1\n'
                                     'remove dna dna.'+str(res+1)+'.OP2\n'
                                     )

        tleap_mod_script += 'remove dna dna.'+str(res)+' \n'

    tleap_mod_script += ('savepdb dna '+output_pdb+'\n' +
                         'quit')
    return tleap_mod_script


def ex_log_leap(leapscript_name):
    '''just executes the named leapscrit and logs the output to
    leapscript_name.log'''
    os.system('tleap -s -f '+leapscript_name+' > /dev/null')
    os.system('mv leap.log '+leapscript_name+'.log')


def HMassRepartition(input_prmtop, output_prmtop,
                     parmedscriptname='HMrepart.parmed'):
    '''creates input_prmtop_noHMassRepart.prmtop with the old prmtop
    and creates a repartitioned prmtop with the same name as the input'''
    with open(parmedscriptname, 'w') as f:
        f.write('HMassRepartition\n'
                'outparm  '+output_prmtop+' \n'
                'go\n')

    os.system('mv '+input_prmtop+' \
              '+no_ext(input_prmtop)+'_no_HMassRepart.prmtop\n'
              'parmed -p '+no_ext(input_prmtop)+'_no_HMassRepart.prmtop'
              ' -i '+parmedscriptname+' > parmed_log')


def change_base_pdb(input_pdb, modified_pdb, residue_to_replace, base):
    '''changes the residue whos number is given to the given residuename
    deletes all non backbone Atoms, so that the resulting pdb can only
    be used after reloading it into leap(with the forcefield loaded)'''
    base = base.upper()

    with open(input_pdb, 'r') as original:
        with open(modified_pdb, 'w') as modified:
            for line in original:

                is_TER_or_END = ('END' in line[0:6]) or ('TER' in line[0:6])
                if is_TER_or_END or (int(line[23:26]) != residue_to_replace):
                    modified.write(line)
                else:
                    atom_name = line[12:16]
                    if 'P' in atom_name or "'" in atom_name:
                        line = line[:19]+base+line[20:]
                        modified.write(line)
