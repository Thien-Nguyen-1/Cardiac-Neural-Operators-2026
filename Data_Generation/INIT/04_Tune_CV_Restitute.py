#!/usr/bin/env python

EXAMPLE_DESCRIPTIVE_NAME = 'Computing conduction velocity restitution'
EXAMPLE_AUTHOR = 'Jason Bayer <jason.bayer@ihu-liryc.fr>'
GUIinclude = True

import sys
import os
import glob
import shutil
from datetime import date
from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import testing
from matplotlib import pyplot
import matplotlib.pyplot as plt

def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')

    # Add arguments 
    group.add_argument('--Gil',
                        type = float, default = 0.174,
                        help = 'Intracellular conductivity (S/m)')
    group.add_argument('--Gel',
                        type= float, default = 0.625,
                        help = 'Extracellular conductivity (S/m)')   
    group.add_argument('--nbeats',
                        type=int, 
                        default=5,
                        help='Number of beats for S1 pacing at CI1')
    group.add_argument('--CI0',
                        type=int, 
                        default=275,
                        help='Shortest coupling interval')
    group.add_argument('--CI1',
                        type=int, 
                        default=500,
                        help='Longest coupling interval')
    group.add_argument('--CIinc',
                        type=int, 
                        default=25,
                        help='Decrement for coupling interval')
    group.add_argument('--model',
                        type = str, default = 'AlievPanfilov',
                        help = 'Cell model')
    group.add_argument('--modelpar',
                        type = str, default = '',
                        help = 'Modifying parameters for the cell model')

    return parser

def plotResults(ci,cv,xmin,xmax,ymin,ymax, webgui, idExp):
    if (webgui):
        datadic = {'labels':{'labelsAB':[]}, \
                'datasets':{'xlim':[xmin,xmax], 'ylim':[ymin,ymax],'labelXY':['Coupling Interval (ms)', 'Conduction velocity (cm/s)'],\
                'valueX':ci, 'valueY1':cv, 'valueY2':[]}}
        
        with open(idExp + "_" + "matplotM.txt", 'w') as f:
            for key, value in datadic.items():
                for key2, value2 in value.items():
                    f.write('%s\n' % (value2))
    else:
        # Plot CV vs DI
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)

        ax.plot(ci, cv, 'rx-')

        ax.set_xlabel('Coupling Interval (ms)')
        ax.set_ylabel('Conduction velocity (cm/s)')
        ax.set_ylim(ymin,ymax)
        ax.set_xlim(xmin,xmax)

        pyplot.show()

def jobID(args):
    today = date.today()
    ID = '04_CVRestitute_{}_{}_Gil-{}_Gel-{}_CI0-{}_CI1-{}_params-{}'.format(today.isoformat(), args.model, args.Gil, args.Gel, args.CI0, args.CI1, args.modelpar)
    return ID

@tools.carpexample(parser, jobID, clean_pattern=r'^(\d{4}-\d{2}-\d{2})|(mesh)|(.sv)|(.log)|(.txt)|(.dat)|^(imp_)')
def run(args, job):

    #Run restituteCV
    cmd = [settings.execs.restituteCV, 
    #cmd = [ 'restituteCV',
            '--model', args.model,
            '--CI0', args.CI0,
            '--CI1', args.CI1,
            '--CIinc', args.CIinc,
            '--numCycles', args.nbeats,
            '--bcl', args.CI1,
            '--gi', args.Gil,
            '--ge', args.Gel ]
   
    if args.modelpar:
        cmd += ['--modelpar', args.modelpar]
    
    # run tuneCV 
    job.bash(cmd)

    #Process the data into two columns
    if args.visualize and not args.dry:
        cvfile = './CVrestitution_AlievPanfilov_bcl_' + str(args.CI1) + '_beats_' + str(args.nbeats) + '.dat'
        file = open(cvfile, 'r')
        lines=file.readlines()
        ci = []
        cv = []
        cnt = -2
        diffci = 0
        diffcv = 0
        for line in lines:
            if cnt > -1:
                p = line.split()
                if  p[1] != 'inf':
                    ci.append(float(p[0]))
                    cv.append(float(p[1]))                    
            cnt += 1
        file.close()
        plotResults(ci,cv,min(ci)-10,max(ci)+10,min(cv)-0.1,max(cv)+0.1,args.webGUI,args.ID)

    # move results to folder with job.ID name
    for file in glob.glob(r'*.txt'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.dat'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.sv'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.log'):
        shutil.move(file, job.ID)


if __name__ == '__main__':
    run()
