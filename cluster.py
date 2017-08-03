import os
import myLib as lib
import pytraj as pt
import numpy as np


def compress_traj_pt(trajs, top, targetdir,
                     cutoff_fraction, compression,
                     outfile_name=None):
    '''cutoff_fraction is the fraction of frames left out at the beginning
    compression = 1 will give no compression'''
    top = pt.load_topology(top)
    noexttraj = os.path.splitext(os.path.basename(trajs[0]))[0]
    traj = pt.iterload(trajs, top)
    if cutoff_fraction > 1:
        traj = traj[len(traj)/cutoff_fraction::compression]
    else:
        traj = traj[::compression]

    if outfile_name is None:
        outfile_name = 'compress_'+noexttraj+'.nc'
    os.system('mkdir -p '+targetdir)
    traj.save(targetdir+'/'+outfile_name, overwrite=True)
    return lib.paths_for_here(targetdir, outfile_name)


def compress_traj(trajectory, topology, targetdir):
    noexttraj = os.path.splitext(os.path.basename(trajectory))[0]
    os.system('mkdir -p '+targetdir)
    os.chdir(targetdir)
    trajectory = lib.paths_for_here(targetdir, trajectory)
    topology = lib.paths_for_here(targetdir, topology)
    trajin = ('parm '+topology+' \n'
              'trajin '+trajectory+' 1 last 5'+' \n'
              'trajout '+noexttraj+'_compr.nc'+' \n')

    cpptrajfile = 'compress_'+noexttraj+'.cpptraj'
    with open(cpptrajfile, 'w') as f:
        f.writelines(trajin)
    os.system('cpptraj -i '+cpptrajfile)
    os.chdir('../')


def filter_seperated(trajectory, topology, targetdir, separated_frame, masks,
                     outprefix):
    cwd = os.getcwd()
    trajectory = lib.paths_for_there(trajectory, targetdir)
    topology = lib.paths_for_there(topology, targetdir)
    separated_frame = lib.paths_for_there(separated_frame, targetdir)
    os.system('mkdir -p '+targetdir)
    os.chdir(targetdir)

    cpptrajtext = ('parm '+topology+' \n'
                   'trajin '+trajectory+' \n'
                   'reference '+separated_frame+' [ref] \n'
                   'nativecontacts '+masks+' distance 3.3 out '
                   'contacts series ref [ref] \n'
                   'go \n')

    with open('cpptraj.in', 'w') as f:
        f.write(cpptrajtext)
    os.system('mpirun -np 4 cpptraj.MPI -i cpptraj.in')

    top = pt.load_topology(topology)
    traj = pt.iterload(trajectory, top)

    d = np.loadtxt('contacts', usecols=(0, 2), dtype=int)
    nocontact = []
    contact = []
    for i, k in d:
        if k == 0:
            nocontact.append(i-1)
        else:
            contact.append(i-1)

    pt.write_traj('nocont_'+outprefix+'.nc',
                  traj[nocontact], top=top, overwrite=True)
    conttraj = 'cont_'+outprefix+'.nc'
    pt.write_traj(conttraj,
                  traj[contact], top=top, overwrite=True)
    os.chdir(cwd)

    return lib.paths_for_here(targetdir, conttraj)


def cluster_epsilon(trajectory, topology, targetdir, epsilon='3.0',
                    mask=':1-8'):
    noexttraj = os.path.splitext(os.path.basename(trajectory))[0]
    nowdir = os.getcwd()
    os.chdir(targetdir)
    trajectory = lib.paths_for_here(nowdir, trajectory)
    topology = lib.paths_for_here(nowdir, topology)
    trajin = ('parm '+topology+' \n'
              'trajin '+trajectory+' \n'
              'autoimage \n'
              'cluster hieragglo epsilon '+epsilon+' averagelinkage'
              ' rms '+mask+' out cnumvtime_'+noexttraj+'.dat'
              ' summary avg_summary_'+noexttraj+'.dat'
              ' clusterout clus clusterfmt netcdf'
              ' repout repclus repfmt pdb \n')

    cpptrajfile = 'cluster_'+noexttraj+'.cpptraj'
    with open(cpptrajfile, 'w') as f:
        f.writelines(trajin)
    os.system('mpirun -np 4 cpptraj.MPI -i '+cpptrajfile)
    os.chdir(nowdir)


def arti_collv(trajs, topology):
    top = pt.load_topology(topology)
    traj = pt.iterload(trajs, top)
    atoms = lib.find_p_atoms_pytraj(topology)
    mask = '@'+','.join(atoms[0])+' @'+','.join(atoms[1])
    dist = pt.distance(traj, top=top, mask=mask)
    bins = np.linspace(min(dist), max(dist), 30)
    binned = np.digitize(dist, bins)
    trajs_indxs = dict()
    trajs_indxs = {i: None for i in binned}
    for k, val in enumerate(binned):
        if trajs_indxs[val] is None:
            trajs_indxs[val] = [k]
        else:
            trajs_indxs[val].append(k)
    return trajs_indxs, bins


def write_arti_collv(trajs, topology):
    trajs_indxs, bins = arti_collv(trajs, topology)
    traj = pt.iterload(trajs, topology)
    for key, val in trajs_indxs.iteritems():
        pt.write_traj('../art_cluster/window'+str(key)+'.nc',
                      traj, top=topology, frame_indices=val, overwrite=True)
    with open('../art_cluster/bins', 'w') as f:
        i = 0
        while i+1 < len(bins):
            f.write(str(bins[i])+' to ' + str(bins[i+1])+' bin '+str(i+1))
            i += 1
