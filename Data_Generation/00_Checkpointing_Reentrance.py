#!/usr/bin/env python

'''
Adapted from the checkpointing early access example in carputils. 
Use this alongside the mesh and tissue set up that you are using to find the ideal time to start your S2 stimulus.

'''
import os
EXAMPLE_DIR = str(os.path.dirname(__file__))

from datetime import date
from carputils.model import optionlist
from carputils.model.lat import Lat
from carputils.carpio import txt
from carputils import settings
from carputils import tools
from carputils import mesh
from carputils.job.command import ShellCommand
from pathlib import Path
from time import sleep

import pathlib
import shutil
import subprocess


def parser():
    parserobj = tools.standard_parser()
    group     = parserobj.add_argument_group('Experiment specific options')
    group.add_argument('--mesh',
                       type = str, default = 'Patch_10cm_250um',
                       help = 'Mesh to run the experiment on (default is %(default)s)')
    group.add_argument('--init',
                       type = str, default = 'checkpoints/INIT_500_BCL_AlievPanfilov_Model__Params.sv',
                       help = 'Single cell initial state (default is %(default)s)')
    group.add_argument('--im_param',
                        type = str,
                        default = '',
                        help = 'List of variable modifications for the ionic currents that you want to change')
    group.add_argument('--IMP',
                        type = str,
                        default = 'AlievPanfilov',
                        help = 'Ionic model name from limpet directory')
    group.add_argument('--Gil',
                        type = float, default = 0.174,
                        help = 'Intracellular conductivity (S/m)')
    group.add_argument('--Gel',
                        type= float, default = 0.625,
                        help = 'Extracellular conductivity (S/m)')   
    group.add_argument('--conmul',
                       type = float,
                       default = 1.0,
                       help = 'Multiplier for myocardium conductivities, default is %(default)s')
    group.add_argument('--experiment',
                        default='chkpt',
                        choices=['chkpt', 'restart' ],
                        type=str,
                        help='Choose experiment to perform. An initial simulation run with checkpoints is needed '
                             'before running a restart.')
    group.add_argument('--chkpt-approach',
                       choices=['intv', 'tsav'],
                       default='intv',
                       type=str,
                       help='Choose checkpointing approach, which could be an interval or dedicated saving timepoints.')
    intervalgroup = parserobj.add_argument_group('chkpt-approach "intv" specific arguments')
    intervalgroup.add_argument('--chkpt-start',
                               default=2500,
                               type=int,
                               help='Choose first checkpoint instance in time (ms)')
    intervalgroup.add_argument('--chkpt-intv',
                               default=10,
                               type=int,
                               help='Choose checkpointing interval (ms).')
    intervalgroup.add_argument('--chkpt-stop',
                               type=int,
                               default=3000,
                               help='Choose last checkpoint instance in time (ms).')
    tsavgroup = parserobj.add_argument_group('chkpt-approach "tsav" specific arguments')
    tsavgroup.add_argument('--tsav',
                           default=[3100],
                           nargs='+',
                           type=int,
                           help='Choose discrete checkpointing instances (ms).')
    group.add_argument('--restart-time',
                       default=3100,
                       type=int,
                       help='Choose checkpoint instance to continue the simulation (ms). '
                            'Note: the checkpoint file <path_to_simID>/chkpt/checkpoint.<restart-time>.0.roe OR '
                            '      the state file <path_to_simID>/chkpt/state.<restart-time>.0.roe must have been '
                            '      created during checkpointing!')
    group.add_argument('--S2',
                       default=3150,
                       type=int,
                       help='Choose time instance for S2 stimulus (ms). '
                            'Note: The S2 stimulus must be place after the checkpoint time otherwise '
                            '      it would be ignored by the simulator.')
    group.add_argument('--with-sentinel',
                       action='store_true',
                       help='Turn on sentinel to trigger simulator exit if no activation was detected.')
    group.add_argument('--detect-lats',
                       action='store_true',
                       help='Check for activation. If there are no activation events detected over the '
                                'prescribed time window, we checktpoint and terminate.')
    group.add_argument('--detect-lrts',
                       action='store_true',
                       help='Check for local repolarization events. If there are no repolarization events '
                            'detected over the prescribed time window, we checktpoint and terminate.')
    group.add_argument('--create-movie',
                       action='store_true',
                       help='Create movie from simulation run. Requires meshalyzer and ffmpeg to be present!')
    return parserobj

def create_movie(job, meshfile, datafile, viewfile):
    FFMPEGOPTS = ("-pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -movflags +faststart "
                  "-framerate 60 -vf 'yadif,pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=white,format=yuv420p' "
                  "-force_key_frames 'expr:gte(t,n_forced/2)' -y -crf 18")

    simID = os.path.dirname(datafile)
    movie_path = os.path.join(simID, 'movie')

    if not os.path.exists(movie_path):
        os.makedirs(movie_path, exist_ok=True)
    if not shutil.which('ffmpeg'):
        Warning("ffmpeg not available. Please install with system package manager (ffmpeg).")
        return
    if not shutil.which('convert'):
        Warning("convert not available. Please install with system package manager (image-magick).")
        return


    cmd  = settings.execs.MESHALYZER.__str__() + " "
    cmd += meshfile + " "
    cmd += datafile + " "
    cmd += viewfile + " "
    cmd += "--frame=0 "
    cmd += "--numframe=2000 "
    cmd += "--compSurf=+ "
    cmd += f"--PNGfile={movie_path} "

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
    retval = p.wait()

    # -------------------------------------------------------------------------
    # trim white space in the images
    cmd  = 'cd ' + movie_path + " "
    cmd += "& find ./ -name 'frame*.png' "
    cmd += '-exec convert {} -trim {} \;'

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
    retval = p.wait()

    # -------------------------------------------------------------------------
    # export mp4 file
    cmd  = 'cd ' + movie_path + ' '
    cmd += '&& ffmpeg ' + FFMPEGOPTS + ' '
    mp4_file = os.path.join(os.path.dirname(movie_path), os.path.basename(os.path.dirname(movie_path)) + '.mp4')
    cmd += mp4_file

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
    retval = p.wait()

    # -------------------------------------------------------------------------
    # convert mp4 file to gif
    #cmd  = 'cd ' + movie_path + ' '
    cmd  = 'ffmpeg -i ' + mp4_file + ' '
    cmd += '-r 50 -vf scale=512:-1 '
    cmd += mp4_file.replace('.mp4', '.gif')

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
    retval = p.wait()

    return None

def jobID(args):
    """
    Generate name of top level output directory.
    """
    # Return conducivity info:
    def hmean(gi, ge):
        """
        Compute harmonic mean conductivity
        """
        gm = gi*ge/(gi+ge)
        return gm

    gm = hmean(args.Gil*args.conmul, args.Gel*args.conmul)
    gm = round(gm, 2)
    today = date.today()
    if args.experiment == 'chkpt':
        ID = '06_S1_Stimulus_Checkpoint_{}_{}_{}-Gm_{}_params_{}-S1'.format(today.isoformat(), args.IMP, gm, args.im_param, args.chkpt_approach)
    elif args.experiment == 'restart':
        ID = '06_S2_Stimulus_Checkpoint_{}_{}_{}-Gm_{}_params_{}-S2'.format(today.isoformat(), args.IMP, gm, args.im_param, args.S2)
    return ID


@tools.carpexample(parser, jobID, clean_pattern='{}*|(.trc)'.format(date.today().year))
def run(args, job):
    
    cmd  = tools.carp_cmd()

    cmd += tools.gen_physics_opts(IntraTags=[1])

    # Define the mesh and simulation time
    cmd += ['-meshname', args.mesh,
                        '-gridout_i', 3,
                ]
    
    # INITIALISATION OF VARIABLES

    #Define the initial cell states for the model 
    cmd += ['-imp_region[0].im_sv_init', args.init]
    
    if not isinstance(args.tsav, list):
        args.tsav = [args.tsav]
    if args.chkpt_start and not args.chkpt_stop:
        args.chkpt_stop = args.chkpt_start + 300
    restart_statef = None
        
    #Define the physics regions
    cmd += ['-num_phys_regions', 2,
            '-phys_region[0].name', 'intracellular',
            '-phys_region[0].ptype', 0,
            '-phys_region[1].name', 'extracellular',
            '-phys_region[1].ptype', 1
            ]
        
    # Define the ionic model to use for the simulation
    cmd += ['-num_imp_regions',          1,
            '-imp_region[0].im',         args.IMP
    ]

    # Adjust ionic model parameters if specified by the user - useful for inducing spiral breakup or other dynamics
    if args.im_param:
            cmd += ['-imp_region[0].im_param', args.im_param]

    # Define outputs and postprocesses, save the final state
    cmd += ['-spacedt', 1.0,
                    '-timedt', 1.0,
                    '-compute_APD', 1,
                    '-dt', 10
                    ]
    

    # Define the outputs via trace files: return the recovery current 
    cmd += ['-num_gvecs', 1,
            '-gvec[0].imp', args.IMP,
            '-gvec[0].ID[0]', 'V',
            '-gvec[0].name', 'w', 
                    ]

    # Set isotropic monodomain conductivities for the tissue - adjust conditions for spiral breakup if needed
    cmd += [ '-num_gregions',	1,
                    
            '-gregion[0].name', 		"myocardium",
            '-gregion[0].num_IDs',           1,
            '-gregion[0].ID[0]', 		"100",		

            # mondomain conductivites will be calculated as half of the harmonic mean of intracellular
            # and extracellular conductivities

            '-gregion[0].g_il',       args.Gil,
            '-gregion[0].g_el',       args.Gel,
            '-gregion[0].g_it',       args.Gil,
            '-gregion[0].g_et',	      args.Gel,
            '-gregion[0].g_in',       args.Gil,
            '-gregion[0].g_en',	      args.Gel,
            '-gregion[0].g_mult',	  args.conmul,

                            ]


    # APPLY S1 STIMULUS AND ACTIVATE CHECKPOINTING
    # + distinguish between two approaches: chkpt interval approach
    # +                                     choose explicit time points for saving simulation state
    
    if args.experiment == 'chkpt':
        
        simID = 'chkpt'
        cmd += ['-simID',                      simID]
        cmd += ['+F',                       'S1.par']
        cmd += ['-imp_region[0].im_sv_init', args.init]
        cmd += ['-tend', args.chkpt_stop if args.chkpt_approach == 'intv' else max(args.tsav)]
        cmd += ['-stim[0].pulse.strength',       250]

        if args.chkpt_approach == 'intv':
            cmd += ['-chkpt_start', args.chkpt_start]
            cmd += ['-chkpt_intv',  args.chkpt_intv]
            cmd += ['-chkpt_stop',  args.chkpt_stop]

        elif args.chkpt_approach == 'tsav':
            # tsav approach with explicit instants, use less granularity here as this is less suitable
            cmd += ['-num_tsav',   len(args.tsav)]
            for idx in range(len(args.tsav)):
                cmd += [f'-tsav[{idx}]', args.tsav[idx]]

        #Calculate the APDs for 90% repolarization:
        cmd += ['-num_LATs', 2, 
                
                '-lats[0].ID ', 'LATS',
                '-lats[0].all', 0,
                '-lats[0].measurand', 0,
                '-lats[0].threshold',  '-20',
                '-lats[0].mode', 0,

                '-lats[1].ID',         'REPS',
                '-lats[1].all',         0,
                '-lats[1].measurand',  0,
                '-lats[1].threshold', '-70',
                '-lats[1].mode', 1,
                ]

        cmd += ['-compute_APD', 1,
                '-actthresh', '-10', 
                '-recovery_thresh', '-70'
                ]

    
    # APPLY S2 STIMULUS AND CHECK INDUCIBILITY
    elif args.experiment == 'restart':
        simID = os.path.join(job.ID, 'restart')
        cmd += ['-simID', simID] # when restarting at 3100ms the 6th stimulus would be missing
        cmd += ['+F',  'S2.par']
        cmd += ['-tend', args.restart_time + 2000]

        # Keep in mind, stimulus start must be later than checkpoint,
        # otherwise stimulus won't be triggered
        chkpt_file = os.path.join(EXAMPLE_DIR, 'chkpt', f'checkpoint.{args.restart_time}.0')
        state_file = os.path.join(EXAMPLE_DIR, 'chkpt', f'state.{args.restart_time}.0')

        # assign on or the other checkpoint file, if it exists
        if not os.path.exists(os.path.join(EXAMPLE_DIR, 'chkpt')):
            raise ValueError('Could not find the checkpoint folder. '
                             'Assuming you have not created any checkpoints!')
        if  not os.path.exists(chkpt_file + '.roe') and \
            not os.path.exists(state_file + '.roe'):

            raise ValueError('Could not find the given checkpoint files - {} or {}. '
                             'Choose an existing one!'.format(state_file + '.roe', chkpt_file + '.roe'))
        else:
            # Keep in mind, stimulus start must be later than checkpoint,
            # otherwise stimulus won't be triggered
            start_statef = chkpt_file if os.path.exists(os.path.exists(chkpt_file + '.roe')) else state_file

        if args.S2 == 3150:
            # restart from a checkpoint where inducibility is unsuccessful,
            # too early, refractory isoline is not on tissue
            sleep(0) # add pseudo pause
        elif args.S2 == 3220:
            # restart from a checkpoint where inducibility is successful
            sleep(0) # add pseudo pause
        elif args.S2 == 3290:
            # restart from a checkpoint where inducibility is unsuccessful, too late,
            # refractory isoline does not intersect with stimulus
            sleep(0) # add pseudo pause
        else:
            # user defined restart checkpoint instance
            sleep(0) # add pseudo pause

        # consistency check
        if args.S2 <= args.restart_time:
            raise Warning(f"S2 stimulus (@{args.S2}ms) must be set after the restart (@{args.restart_time}ms). Otherwise, it will be ignored!")

        cmd += ['-start_statef', start_statef]
        cmd += ['-stim[0].ptcl.start', args.S2]

        # add event detectors and a sentinel
        if args.with_sentinel:
            cmd += ['-sentinel_ID',                      0]
            cmd += ['-t_sentinel_start', args.restart_time]
            cmd += ['-t_sentinel',                      40]  # no LATs nor LRTs within 40 ms
            cmd += ['-tend', args.restart_time + 2000     ]

            if not args.detect_lats and not args.detect_lrts:
                raise ValueError('The sentinel feature depends on LAT or LRT detection. '
                                 'Please add one of the two detectors!')
    
    
    # COMPLETE COMMANDLINE PARAMETERS BEFORE OPENCARP SIMULATOR CALL
    # + add general commandline parameters
    
    cmd += ['-parab_solve', 0]
    cmd += ['-meshname', args.mesh]

    # turn on local activation detector
    detectors = []
    if args.detect_lats:
        lats = Lat(ID='LATs', measurand=0, method=1, all=0, threshold=-40., mode=0)
        detectors.append(lats)
    if args.detect_lrts:
        lrts = Lat(ID='LRTs', measurand=0, method=1, all=0, threshold=-70., mode=1)
        detectors.append(lrts)
    if detectors:
        cmd += optionlist(detectors)
    
    # LAUNCH OPENCARP SIMULATOR
    job.carp(cmd)

    # Calculate the APDs:

    if args.experiment == 'chkpt':
        LATs = txt.read(os.path.join('chkpt', 'init_acts_LATS-thresh.dat'))
        REPs = txt.read(os.path.join('chkpt', 'init_acts_REPS-thresh.dat'))
        APDs = REPs - LATs
        txt.write(os.path.join('chkpt', 'APDs.dat'), APDs)
    else:
        LATs = txt.read(os.path.join(job.ID, 'init_acts_LATS-thresh.dat'))
        REPs = txt.read(os.path.join(job.ID, 'init_acts_REPS-thresh.dat'))
        APDs = REPs - LATs
        txt.write(os.path.join(job.ID, 'APDs.dat'), APDs)



    # Remove the checkpoint folder if it exists, to avoid confusion with previous runs
    def hmean(gi, ge):
        """
        Compute harmonic mean conductivity
        """
        gm = gi*ge/(gi+ge)
        return gm

    gm = hmean(args.Gil*args.conmul, args.Gel*args.conmul)
    gm = round(gm, 2)
    today = date.today()
    checkpoint_dir = os.path.join(EXAMPLE_DIR, '06_S1_Stimulus_Checkpoint_{}_{}_{}-Gm_{}_params_{}-S1'.format(today.isoformat(), args.IMP, gm, args.im_param, args.chkpt_approach))
    if os.path.exists(checkpoint_dir):
        pathlib.Path.rmdir(Path(checkpoint_dir))

if __name__ == '__main__':
    run()    
