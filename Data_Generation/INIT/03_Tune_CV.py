#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
EXAMPLE_DESCRIPTIVE_NAME = 'Tuning velocities and anisotropy ratios'
EXAMPLE_AUTHOR = 'Caroline Mendonca Costa <caroline.mendonca-costa@kcl.ac.uk>'
EXAMPLE_DIR = os.path.dirname(__file__)
GUIinclude = False

import sys
from datetime import date
from carputils import settings
from carputils import tools
from carputils.carpio import txt
from matplotlib import pyplot


def parser():
    parser = tools.standard_parser()
    group = parser.add_argument_group('experiment specific options')
    group.add_argument('--resolution',
                        type = float, default = 100.,
                        help = 'Spatial resolution')
    group.add_argument('--velocity',
                        type = float, default = 0.6,
                        help = 'Desired conduction velocity (m/s)')
    group.add_argument('--gi',
                        type = float, default = 0.174,
                        help = 'Intracellular conductivity (S/m)')
    group.add_argument('--ge',
                        type= float, default = 0.625,
                        help = 'Extracellular conductivity (S/m)')
    group.add_argument('--model',
                        type = str, default = 'AlievPanfilov',
                        help = 'Cell model')
    group.add_argument('--modelpar',
                        type = str, default = '',
                        help = 'Modifying parameters for the cell model')
    group.add_argument('--ts',
                        type = int, default = 0,
                        help = 'Choose time stepping method. \n'
                               '0: explicit Euler,\n'
                               '1: crank nicolson,\n'
                               '2: second-order time stepping')
    group.add_argument('--dt',
                        type = float, default = 10.,
                        help = 'Integration time step on micro-seconds')
    group.add_argument('--lumping',
                        type = bool, default = False,
                        help = 'Use mass lumping')
    group.add_argument('--converge',
                        type = bool, default = False,
                        help = '0: Measure velocity with given setup or \n'
                               '1: Compute conductivities that yield desired velocity with given setup')
    group.add_argument('--compareTuning',
                        action = 'store_true',
                        help = 'run tuneCV with --resolution 100 200 and 400 with and without tuning and make '
                            'comparison plot')
    group.add_argument('--compareTimeStepping',
                        action = 'store_true',
                        help = 'run tuneCV with --resolution 100 200 and 400 with Explicit Euler and Crank Nicolson and make '
                               'comparison plot')
    group.add_argument('--compareMassLumping',
                        action = 'store_true',
                        help = 'run tuneCV with --resolution 100 200 and 400 with and without mass lumping and make '
                               'comparison plot')
    group.add_argument('--compareModelPar',
                        action = 'store_true',
                        help = 'run tuneCV with --resolution 100 200 and 400 with original Ten Tusher cell model and with reduced sodium '
                               'conductance and make comparison plot')
    group.add_argument('--ar',
                        type = float, default = 0.,
                        help = 'run tuneCV with for CV_f = 0.6 and CV_s = 0.3 m/s with given anisotropy ratio (ar)')
    return parser


def plotResults(res, cv1, cv2, label1, label2, ymin, ymax, webgui, idExp):
    if (webgui):
        datadic = {'labels':{'labelsAB':[label1,label2]}, \
                'datasets':{'xlim':[], 'ylim':[ymin,ymax],'labelXY':['Spatial resolution (um)','Conduction velocity (m/s)'],\
                'valueX':res, 'valueY1':cv1, 'valueY2':cv2}}
        
        with open(idExp + "_" + "matplotM.txt", 'w') as f:
            for key, value in datadic.items():
                for key2, value2 in value.items():
                    f.write('%s\n' % (value2))
        
    else:
        # Plot cv vs dx
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(res, cv1, 'rx-', label=label1)
        ax.plot(res, cv2, 'bx-', label=label2)

        ax.set_xlabel('Spatial resolution (um)')
        ax.set_ylabel('Conduction velocity (m/s)')
        ax.set_ylim(ymin, ymax)
        pyplot.legend(loc='upper right')

        pyplot.show()


def execute(args, job, resolution=None, converge=False, lumping = None,
            ts=None, model=None, gi = None, ge=None, modelpar=None, cvfile='results.dat'):

    # remove old results file
    if os.path.isfile(cvfile):
        os.remove(cvfile)

    # create command line
    cmd = [settings.execs.TUNECV,
           '--np',          args.np,
           '--resolution',  args.resolution if not resolution else resolution,
           '--gi',          args.gi if not gi else gi,
           '--ge',          args.ge if not ge else ge,
           '--ts',          args.ts if not ts else ts,
           '--dt',          args.dt,
           '--tol',         0.01,
           '--length',      0.25,    # reduce cable length to limit computation time
           '--lumping',     args.lumping if not lumping else lumping,
           '--model',       args.model if not model else model,
           '--stimS', 250
           ]

    if modelpar or args.modelpar:
        cmd += ['--modelpar', args.modelpar if not modelpar else modelpar]

    if converge or args.converge:
        cmd += ['--velocity', args.velocity,
                '--converge', True]

    # run tuneCV
    job.bash(cmd)

    try:
        # Read tuning results
        mcv = txt.read(cvfile)
    except:
        print('Could not read cvfile')
        return -1

    if args.ar > 0.:
        return mcv[1], mcv[2]
    else:
        return mcv[0]


def jobID(args):
    """
    Generate name of top level output directory.
    """
    today = date.today()
    if args.converge == 1:
        return '03_tuneCV_{}_{}_{}_{}_velocity_{}_params'.format(today.isoformat(), args.model, args.resolution, args.velocity, args.modelpar)
    else:
        return '03_tuneCV_{}_{}_{}'.format(today.isoformat(), args.model, args.resolution)


@tools.carpexample(parser, jobID, clean_pattern='{}*|(.log)|(.dat)|imp_*|(mesh)'.format(date.today()))
def run(args, job):

    # -------------------------------------------------------------------------
    if args.compareTuning:
        if '--resolution' in sys.argv:
            raise Exception('Cannot set --resolution with --compareTuning!')

        res = [100., 200., 400.]
        cv1 = []
        cv1.append(execute(args, job, resolution = res[0]))
        cv1.append(execute(args, job, resolution = res[1]))
        cv1.append(execute(args, job, resolution = res[2]))

        cv2 = []
        cv2.append(execute(args, job, resolution = res[0], converge = 1))
        cv2.append(execute(args, job, resolution = res[1], converge = 1))
        cv2.append(execute(args, job, resolution = res[2], converge = 1))

        label1 = 'CV without tuning'
        label2 = 'CV with tuning'
        ymin = 0.58
        ymax = 0.67

    # -------------------------------------------------------------------------
  
    elif args.compareTimeStepping:
            if '--resolution' in sys.argv or '--ts' in sys.argv:
                raise Exception('Cannot set --resolution or --ts with --compareTimeStepping')

            res = [100, 200, 400]
            cv1 = []
            cv1.append(execute(args, job, resolution = res[0], ts = 0))
            cv1.append(execute(args, job, resolution = res[1], ts = 0))
            cv1.append(execute(args, job, resolution = res[2], ts = 0))

            cv2 = []
            cv2.append(execute(args, job, resolution = res[0], ts = 1))
            cv2.append(execute(args, job, resolution = res[1], ts = 1))
            cv2.append(execute(args, job, resolution = res[2], ts = 1))

            label1 = 'CV with Explicit Euler'
            label2 = 'CV with Crank Nicolson'
            ymin = 0.60
            ymax = 0.66

        # -------------------------------------------------------------------------

    elif args.compareMassLumping:
        if '--resolution' in sys.argv or '--lumping' in sys.argv:
            raise Exception('cannot set --resolution or --lumping with --compareMassLumping')

        res = [100, 200, 400]
        cv1 = []
        cv1.append(execute(args, job, resolution = res[0], lumping = 0))
        cv1.append(execute(args, job, resolution = res[1], lumping = 0))
        cv1.append(execute(args, job, resolution = res[2], lumping = 0))

        cv2 = []
        cv2.append(execute(args, job, resolution = res[0], lumping = True))
        cv2.append(execute(args, job, resolution = res[1], lumping = True))
        cv2.append(execute(args, job, resolution = res[2], lumping = True))

        label1 = 'CV without Mass Lumping'
        label2 = 'CV with Mass Lumping'
        ymin = 0.48
        ymax = 0.68

    # -------------------------------------------------------------------------

    elif args.compareModelPar:
        if '--resolution' in sys.argv or '--model' in sys.argv or '--modelpar' in sys.argv:
            raise Exception('cannot set --resolution or --model or --modelpar with --compareModelPar')

        res = [100, 200, 400]
        cv1 = []
        cv1.append(execute(args, job, resolution = res[0], model = 'tenTusscherPanfilov'))
        cv1.append(execute(args, job, resolution = res[1], model = 'tenTusscherPanfilov'))
        cv1.append(execute(args, job, resolution = res[2], model = 'tenTusscherPanfilov'))

        cv2 = []
        cv2.append(execute(args, job, resolution = res[0], model = 'tenTusscherPanfilov', modelpar = 'GNa*0.5'))
        cv2.append(execute(args, job, resolution = res[1], model = 'tenTusscherPanfilov', modelpar = 'GNa*0.5'))
        cv2.append(execute(args, job, resolution = res[2], model = 'tenTusscherPanfilov', modelpar = 'GNa*0.5'))

        label1 = 'CV with original model'
        label2 = 'CV with reduce sodium conductance'
        ymin = 0.46
        ymax = 0.69

    elif args.ar:
        # user defined anisotropy ratio
        if '--velocity' in sys.argv or '--gi' in sys.argv or '--ge' in sys.argv or '--converge' in sys.argv:
            raise Exception('cannot set --velocity or --gi or --ge or --converge with --ar')
    # -------------------------------------------------------------------------

        # set default values
        gil = 0.174
        gel = 0.625
        git = 0.019
        get = (args.ar * gel * git) / gil

        args.velocity = 0.6
        gil_, gel_ = execute(args, job, gi = gil, ge = gel, converge = True)
        args.velocity = 0.3
        git_, get_ = execute(args, job, gi = git, ge = get, converge = True)

        print('g_if: %f\tg_is: %f\tg_ef: %f\tg_es: %f\n' % (gil_, git_, gel_, get_))
        args.visualize = False
    # -------------------------------------------------------------------------
    else:
        cv1 = execute(args, job)
        args.visualize = False


     # move results to folder with job.ID name
    for file in glob.glob(r'*.txt'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.dat'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.sv'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*It_*'):
        shutil.move(file, job.ID)


    # --- Do visualization ----------------------------------------------------
    if args.visualize and not settings.platform.BATCH:
        plotResults(res, cv1, cv2, label1, label2, ymin, ymax, args.webGUI, args.ID)


if __name__ == '__main__':
    run()
