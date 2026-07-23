#!/usr/bin/env python

import os
import glob
import shutil
EXAMPLE_DESCRIPTIVE_NAME = 'APD restitution in a single cell'
EXAMPLE_AUTHOR = 'Jason Bayer <jason.bayer@ihu-liryc.fr>'
EXAMPLE_DIR = os.path.dirname(__file__)
GUIinclude = True

from datetime import date
from carputils import settings
from carputils import tools
from matplotlib import pyplot
import numpy as np

def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')
    # Add arguments    
    group.add_argument('--Protocol',
                        default = 'S1S2',
                        choices  =['S1S2', 'Dynamic'],
                        help = 'Pacing protocol for restitution curve')
    group.add_argument('--prebeats',
                        type = int,
                        default = 20,
                        help='Number of pre-pacing beats at chosen pacing cycle length PCL')
    group.add_argument('--initial',
                        default = None,
                        help = 'Initialize with a stabilized limit cycle state vector precomputed for the chosen PCL')
    group.add_argument('--nbeats',
                        type = int,
                        default = 5,
                        help = 'Number of beats for S1 pacing at CI1')
    group.add_argument('--BCL',
                        type = int,
                        default = 500,
                        help = 'Reference basic cycle length (ms)')
    group.add_argument('--CI0',
                        type  =int,
                        default = 50,
                        help = 'Shortest coupling interval (ms)')
    group.add_argument('--CI1',
                        type = int,
                        help = 'Longest coupling interval (default: BCL)')
    group.add_argument('--CIinc',
                        type = int,
                        default = 25,
                        help = 'Decrement for coupling interval (ms)')
    group.add_argument('--imp',
                        default = 'AlievPanfilov',
                        help =  'Ionic model')
    group.add_argument('--params',
                        default = '',
                        help = 'Ionic model parameters')

    return parser

def plotResults(di,apd,xmin,xmax,ymin,ymax,webgui,idExp):
    if (webgui):
        datadic = {'labels':{'labelsAB':[]}, \
                'datasets':{'xlim':[xmin,xmax], 'ylim':[ymin,ymax],'labelXY':['Diastolic Interval (ms)', 'Action Potential Duration (ms)'],\
                'valueX':di, 'valueY1':apd, 'valueY2':[]}}
        
        with open(idExp + "_" + "matplotM.txt", 'w') as f:
            for key, value in datadic.items():
                for key2, value2 in value.items():
                    f.write('%s\n' % (value2))
    else:
        # Plot APD vs DI
        import matplotlib.pyplot as plt
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
    
        ax.plot(di, apd, 'rx-')
        
        ax.set_xlabel('Diastolic Interval (ms)')
        ax.set_ylabel('Action Potential Duration (ms)')
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin,xmax)
        
        pyplot.show()

def visualize(job, args):
    """
    """
    apdfile = os.path.join(job.ID, 'restout_APD_restitution.dat')
    file = open(apdfile, 'r')
    lines=file.readlines()
    di = []
    apd = []
    cnt = 0
    diffdi = 0
    diffapd = 0
    for line in lines:
        if line[0] == "#": 
            continue
        p = line.split()
        if int(cnt) > int(0):
            diffdi = float(di[int(cnt)-1])-float(p[5])
            diffapd = float(apd[int(cnt)-1])-float(p[3])
        if float(diffdi) > float(-5) and float(diffapd) > float(-10):
            apd.append(float(p[3]))
            di.append(float(p[5]))
        else:
            break
        cnt += 1
    file.close()

    print(apd)

    plotResults(di,apd,min(di)-10,max(di)+10,min(apd)-10,max(apd)+10,args.webGUI,args.ID)
#    plotResults(di,apd,0.0,args.CI1,0.0,args.CI1)

def jobID(args):
    today = date.today()
    ID = '02_APD_restitution_{}_CI0-{}_CI1-{}_{}_{}params'.format(today.isoformat(), args.CI0, args.CI1, args.Protocol, args.params)
    if args.params:
        ID += '_{}'.format(args.imp,args.params)
    return ID

@tools.carpexample(parser, jobID, clean_pattern=r'^(\d{4}-\d{2}-\d{2})|(.txt)|(.dat)')
def run(args, job):

    # Determine the threshold for the user input parameters
    stimcurr = 0
    thresh_achieved = False
    delta_curr = 2

    # check input args
    if not args.CI1:
        args.CI1 = args.BCL

    # set up ionic model
    imp_setup = ['--imp', args.imp]
    if args.params:
        imp_setup += ['--imp-par', args.params]

    # build baseline command line
    bcmd = imp_setup

    while not thresh_achieved:
        stimcurr += delta_curr
        print('Currently applied stimulus current: {}'.format(stimcurr))

        # define protocol
        pars = ['--duration', 100.1,
                '--numstim', 1,
                '--stim-start', 1,
                '--bcl', 100,
                '--stim-curr', stimcurr,
                '--stim-dur', 2,
                '--fout={}'.format(os.path.join(job.ID, 'thresh')),
                '--save-time', 100,
                '--save-file', os.path.join(job.ID, 'thresh_save.sv')]
        
        # run threshold
        job.bench(bcmd+pars)


        # Now read in the data
        if args.dry:
            thresh_achieved = True
        else:
            vmfile = os.path.join(job.ID, 'thresh.txt')
            Vm = np.loadtxt(vmfile)
            idx = np.argmax(Vm[:,1] > -10)
            if idx > 0:
                thresh_achieved = True

    stimcurr = stimcurr*2
    print('Chosen stimulus current: {}'.format(stimcurr))

    # Write the S1S2 restitution file
    if args.Protocol == 'S1S2':
        ropt = 'S1S2'
        if not args.dry:
            with open(os.path.join(job.ID, 'restitution_protocol.txt'), 'w') as fp:
                fp.write('  1   # protocol selection 1=S1S2 0=dynamic\n')
                fp.write('{:3d} # number of prepacing beats before starting protocol\n'.format(args.prebeats))
                fp.write('{:3d} # basic cycle length\n'.format(args.BCL))
                fp.write('{:3d} # S2 prematurity start\n'.format(args.CI1))
                fp.write('{:3d} # S2 prematurity end\n'.format(args.CI0))
                fp.write('{:3d} # number of beats preceding premature one\n'.format(args.nbeats))
                fp.write('{:3d} # decrement in S2 prematurity in ms\n'.format(args.CIinc))

    if args.Protocol == 'Dynamic' and not args.dry:
        ropt='dyn'
        if not args.dry:
            with open(os.path.join(job.ID, 'restitution_protocol.txt'), 'w') as fp:
                fp.write('  0   # protocol selection 1=S1S2 0=dynamic\n')
                fp.write('{:3d} # number of prepacing beats before starting protocol\n'.format(args.prebeats))
                fp.write('{:3d} # initial basic cycle length\n'.format(args.BCL))
                fp.write('{:3d} # final basic cycle length\n'.format(args.CI0))
                fp.write('{:3d} # number of beats preceding premature one\n'.format(args.nbeats))
                fp.write('{:3d} # decrement in S2 prematurity in ms\n'.format(args.CIinc))


    # Run bench with restitution file
    pars = ['--stim-curr', stimcurr,
            '--stim-dur', 2,
            '--restitute', ropt,
            '--res-file',  os.path.join(job.ID, 'restitution_protocol.txt'),
            '--res-trace', os.path.join(job.ID, 'restout_trace.txt'),
            '--fout={}'.format(os.path.join(job.ID, 'restout'))]

    if args.initial:
        pars += ['--restore', args.initial]

    # run threshold
    job.bench(bcmd+pars)

    # move results to folder with job.ID name
    for file in glob.glob(r'*.txt'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.dat'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.sv'):
        shutil.move(file, job.ID)

    #Process the data into two columns
    if args.visualize and not settings.platform.BATCH:
        visualize(job,args)
        apdfile = os.path.join(job.ID, 'restout_APD_restitution.dat')

        cmd = [settings.execs.APDRESTITUTION, apdfile, args.imp]
        job.bash(cmd)

if __name__ == '__main__':
    run()
