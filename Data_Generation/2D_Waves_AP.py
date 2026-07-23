# Script to generate 2D wave propagation in a slab of tissue using the Aliev-Panfilov model. 
# The script allows for different stimulation protocols, including planar waves, centrifugal waves, spiral initiation, and spiral breakup. 
# It also provides options for anisotropic conduction and allows for the modification of ionic model parameters.
import os
EXAMPLE_DIR = os.path.dirname(__file__)
CALLER_DIR = os.getcwd()
GUIinclude = False

from datetime import date
from carputils import tools
from carputils import ep
from carputils.carpio import txt
import sys


# Return conducivity info:
def hmean(gi, ge):
        """
        Compute harmonic mean conductivity
        """
        gm = gi*ge/(gi+ge)
        return gm


def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')

    # Add experiment arguments
    group.add_argument('--mesh',
                       type = str, default = '2D_10cm_250um',
                       help = 'Mesh to run the experiment on (default is %(default)s)')
    group.add_argument('--init',
                       type = str, default = '',
                       help = 'Single cell initial state (default is %(default)s)')
    group.add_argument('--tend',
                       type = float, default = 1000,
                       help = 'Duration of simulation (default is %(default)s) ms')
    group.add_argument('--pls',
                       type = float, default = 1,
                       help = 'Number of pulses (default is %(default)s)')
    group.add_argument('--bcl',
                        type = float, default = 500,
                        help = 'BCL pulses (default is %(default)s ms, 80bpm )')
    group.add_argument('--prepacing_bcl',
                        type = float, default = 500,
                        help = 'Pre-pacing BCL pulses (default is %(default)s ms, 80bpm )')
    group.add_argument('--prepacing_stimstr',
                        type = float, default = 250,
                        help = 'Strength for the pre-pacing stimulation (default is %(default)s muA)')
    group.add_argument('--prepacing_stimdur',
                        type = float, default = 2,
                        help = 'Duration for the stimulation (default is %(default)s  ms)')
    group.add_argument('--strength',
                        type = float, default = 250,
                        help = 'Strength for the stimulation (default is %(default)s muA)')
    group.add_argument('--duration',
                        type = float, default = 2,
                        help = 'Duration for the stimulation (default is %(default)s  ms)')
    group.add_argument('--model',
                        type = str,
                        default = "AlievPanfilov", 
                        help='ionic model (default is %(default)s)')
    group.add_argument('--output_res',
                       type = float, default = 5,
                       help = 'Resolution of output for .igb files (default is %(default)s ms)')
    group.add_argument('--check',
                       type = float, default = 500,
                       help = 'Time interval for saving the state of the simulation (default is %(default)s ms, 1 second )')
    group.add_argument('--aniso',
                       action = 'store_true',
                       help = 'Flag to us anisotropic conductivity values')
    group.add_argument('--im_param',
                        type = str,
                        default = '',
                        help = 'List of variable modifications for the ionic currents that you want to change')
    group.add_argument('--conmul',
                       type = float,
                       default = 1.0,
                       help = 'Multiplier for myocardium conductivities, default is %(default)s')

    # Protocol Arguments 
    group.add_argument('--protocol',
                       choices = ['planar', 'centrifugal', 'spiral_init', 'spiral', 'breakup'],
                       default = 'planar',
                       help = 'Option for simple wave excitement, choose from %(choices)s, default is %(default)s, propagating left to right')
    group.add_argument('--start_state', 
                       default = '',
                       help = 'Starting state for continued simulation, default is %(default)s')
    group.add_argument('--Gil',
                        type = float, default = 0.174,
                        help = 'Intracellular conductivity (S/m)')
    group.add_argument('--Gel',
                        type= float, default = 0.625,
                        help = 'Extracellular conductivity (S/m)')  
    group.add_argument('--second_stim_start', 
                       default = 325,
                       help= 'Start time for second stimulus for cross stimulus spiral propagation, default = %(default)s')
    group.add_argument('--second_stim_pls',
                       default = 1,
                       help = 'Number of second stimulus pulses in cross stimulus spiral propagation, default is %(default)s')
    group.add_argument('--second_stim_bcl',
                       default = 500,
                       help = 'BCL of second stimulus pulses in cross stimulus spiral propagation, default is %(default)s')
   

    return parser

def jobID(args):
    today = date.today()

    gm = hmean(args.Gil*args.conmul, args.Gel*args.conmul)
    gm = round(gm, 2)
    
    # Signal for anisotropic conduction:
    if args.aniso:
        aniso_flag = 'Aniso'
    else: 
        aniso_flag = ''

    if args.protocol == 'spiral_init':
        ID = '{}_2D_spiral_init_{}_{}-bcl_{}-pls_{}-uA_{}-ms_{}-start_{}-param_{}-gm_{}_{}-tend'.format(today.isoformat(), args.model, args.bcl,
                                        args.pls, args.strength, args.duration, args.second_stim_start, args.im_param, gm, aniso_flag, args.tend)
        return ID
    elif args.protocol == 'spiral':
        ID = '{}_2D_spiral_{}_{}-bcl_{}-pls_{}-uA_{}-ms_{}-S2start_{}-param_{}-gm_{}_{}-tend'.format(today.isoformat(), args.model, args.bcl,
                                        args.pls, args.strength, args.duration, args.second_stim_start, args.im_param, gm, aniso_flag, args.tend)
        return ID
    elif args.protocol == 'centrifugal':
        ID = '{}_2D_centrifugal_{}_{}-bcl_{}-pls_{}-uA_{}-ms_{}-S2start_{}-param_{}-gm_{}_{}-tend'.format(today.isoformat(), args.model, args.bcl,
                                        args.pls, args.strength, args.duration, args.second_stim_start, args.im_param, gm, aniso_flag, args.tend)
        return ID
    elif args.protocol == 'breakup':
        ID = '{}_2D_spiral_break_{}_{}-bcl_{}-pls_{}-uA_{}-ms_{}-S2start_{}-param_{}-gm_{}_{}-tend'.format(today.isoformat(), args.model, args.bcl,
                                        args.pls, args.strength, args.duration, args.second_stim_start, args.im_param, gm, aniso_flag, args.tend)
        return ID
    else:
        ID = '{}_2D_planar_{}_{}-bcl_{}-pls_{}-uA_{}-ms_{}-param_{}-gm_{}_{}-tend'.format(today.isoformat(), args.model, args.bcl,
                                        args.pls, args.strength, args.duration, args.im_param, gm, aniso_flag, args.tend)
        return ID      


@tools.carpexample(parser, jobID)


def run(args, job):
    
        # Generate general command line
        cmd = tools.carp_cmd()

        # Set output directory
        cmd += ['-simID', job.ID]

        # Ensure the job folder exists
        os.makedirs(job.ID, exist_ok=True)

        log_path = os.path.join(job.ID, "simulation_output.dat")
        logfile = open(log_path, "w")

        class Tee:
                def __init__(self, *files):
                        self.files = files
                def write(self, data):
                        for f in self.files:
                                f.write(data)
                def flush(self):
                        for f in self.files:
                                f.flush()

        sys.stdout = Tee(sys.stdout, logfile)
        sys.stderr = Tee(sys.stderr, logfile)


        # Add some example-specific command line options
        cmd += ['-meshname', args.mesh,
                        '-tend', args.tend ,
                        '-gridout_i', 3,
                       # '-gridout_e', 3
                ]
        #Define the physics regions
        cmd += ['-num_phys_regions', 2,
                        '-phys_region[0].name', 'intracellular',
                        '-phys_region[0].ptype', 0,
                        '-phys_region[1].name', 'extracellular',
                        '-phys_region[1].ptype', 1
                        ]
        # Define the cell model and the number of regions for the simulation
        cmd += ['-num_imp_regions',          1,
                '-imp_region[0].im',         args.model
        ]  
        
        #Define outputs and postprocesses, save the final state
        cmd += ['-spacedt', args.output_res,
                        '-timedt', 1.0,
                        '-compute_APD', 1,
                        '-dt', 10
                        ]
        
        # Save the checkpoints of the simulation 
        cmd += ['-chkpt_start', 0,
                '-chkpt_intv',  args.check,
                '-chkpt_stop', args.tend
                ]
        
        #Define the outputs via trace files
        cmd += ['-num_gvecs', 1,
                '-gvec[0].imp', args.model,
                '-gvec[0].ID[0]', 'V',
                '-gvec[0].name', 'w', 
                        ]

        # Set monodomain conductivities for the tissue and the bath - adjust conditions for spiral breakup if needed
        # Toggle for isotropic vs anisotropic conduction for the slab (2D)

        if args.aniso:
                cmd += [ '-num_gregions',	1,
                                
                        '-gregion[0].name', 		"myocardium",
                        '-gregion[0].num_IDs',           1,
                        '-gregion[0].ID[0]', 		"100",		

                        # mondomain conductivites will be calculated as half of the harmonic mean of intracellular
                        # and extracellular conductivities

                        '-gregion[0].g_il',       0.174,
                        '-gregion[0].g_el',       0.625,
                        '-gregion[0].g_it',       0.019,
                        '-gregion[0].g_et',	  0.236,
                        '-gregion[0].g_in',       0.019,
                        '-gregion[0].g_en',	  0.236,
                        '-gregion[0].g_mult',	  args.conmul,

                                        ]
        else:
                cmd += [ '-num_gregions',			1,
                        
                        '-gregion[0].name', 		"myocardium",
                        '-gregion[0].num_IDs',           1,
                        '-gregion[0].ID[0]', 		"100",		

                        # mondomain conductivites will be calculated as half of the harmonic mean of intracellular
                        # and extracellular conductivities

                        '-gregion[0].g_il',       args.Gil,
                        '-gregion[0].g_el',       args.Gel,
                        '-gregion[0].g_it',       args.Gil,
                        '-gregion[0].g_et',	  args.Gel,
                        '-gregion[0].g_in',       args.Gil,
                        '-gregion[0].g_en',	  args.Gel,
                        '-gregion[0].g_mult',	  args.conmul,

                                ]
                
                gm = hmean(args.Gil*args.conmul, args.Gel*args.conmul)
                
                print("======================================")
                print("Isotropic Conductivities:")
                print(f"Gi = {args.Gil*args.conmul}")    
                print(f"Ge = {args.Gel*args.conmul}") 
                print(f"Gm = {gm}") 
                print("======================================")   

        if args.protocol == 'spiral' or args.protocol == 'breakup':
                
                # Define the starting state of the tissue - i.e continue a simulation from a previous state. 
                if args.start_state:
                        cmd += [
                                '-start_statef', args.start_state
                                ]
                else:
                        print("Error: For spiral or breakup protocols, a starting state must be provided. Please specify a starting state file using the --start_state argument.")
                        sys.exit(1)
                
                # Run example
                job.carp(cmd)


        else:
                # Define the starting state of the tissue (if specified) - i.e continue a simulation from a previous state (without adding any new stimulus)
                if args.start_state:
                        cmd += [
                                '-start_statef', args.start_state
                                ]
                # Define the initial stages for the model based on cell model limit cycle:
                elif args.init:
                        cmd += [
                                '-imp_region[0].im_sv_init', args.init
                                ]
                # Prepace the model
                else:
                        cmd += ['-prepacing_beats', 100,
                                '-prepacing_bcl', args.prepacing_bcl,
                                '-prepacing_stimdur', args.prepacing_stimdur,
                                '-prepacing_stimstr', args.prepacing_stimstr
                                ]
                
                ## Define the stimulus depending on the protocol ##

                if args.protocol == 'spiral_init':

                #Set up electrode across the left wall and dump the electrode to a vtx file.             
                        cmd += ['-num_stim',  2,
                                
                                # Define the electrode geometry based on the mesh (all along the left edge, thin in the x direction)

                                # Define the stimulus pulse 
                                '-stimulus[0].name', 'left_wall'
                                '-stimulus[0].stimtype',   0,
                                '-stimulus[0].strength',   args.strength,
                                '-stimulus[0].duration',   args.duration,
                                '-stimulus[0].npls',       args.pls,
                                '-stimulus[0].bcl',        args.bcl,
                                #'-stimulus[0].geometry', 100,

                                #Define electrode based on mesh geomtery (left wall)
                                #'-stimulus[1].ctr_def', 1,
                                '-stimulus[0].x0', -50000,
                                '-stimulus[0].y0', -50000,
                                '-stimulus[0].z0', 0,

                                '-stimulus[0].xd', 500,
                                '-stimulus[0].yd', 100000,
                                '-stimulus[0].zd', 0,
                                

                                # Dump the electrode file
                                '-stimulus[0].dump_vtx_file', 1,


                                # Define the S2 stimulus electrode geometry based on the mesh (bottom quadrant of the mesh)

                                # Define the stimulus pulse 
                                '-stimulus[1].name', 'S2_stimulus'
                                '-stimulus[1].stimtype',   0,
                                '-stimulus[1].strength',   args.strength,
                                '-stimulus[1].duration',   args.duration,
                                '-stimulus[1].npls',       args.second_stim_pls,
                                '-stimulus[1].bcl',        args.second_stim_bcl,
                                '-stimulus[1].start',       args.second_stim_start,
                                #'-stimulus[1].geometry',     200,    

                                # Define the electrode geometry based on the mesh (bottom half of the mesh)
                                #'-stimulus[1].ctr_def', 1,
                                '-stimulus[1].x0', -50000,
                                '-stimulus[1].y0', -50000,
                                '-stimulus[1].z0', 0,

                                '-stimulus[1].xd', 50000,
                                '-stimulus[1].yd', 50000,
                                '-stimulus[1].zd', 0,
                                

                                # Dump the electrode file
                                '-stimulus[1].dump_vtx_file', 1

                        ]
                        
                elif args.protocol == 'centrifugal':
                # Pulse from a small patch in the bottom corner 
                        cmd += [    '-num_stim',  1,

                                # Define the stimulus pulse 
                                '-stimulus[0].stimtype',   0,
                                '-stimulus[0].strength', args.strength,
                                '-stimulus[0].duration',   args.duration,
                                '-stimulus[0].npls',       args.pls,
                                
                                # Define the electrode geometry based on the mesh 
                                #'-stimulus[0].ctr_def', 1,
                                '-stimulus[0].x0', -50000,
                                '-stimulus[0].y0', -50000,
                                '-stimulus[0].z0', 0,

                                '-stimulus[0].xd', 500,
                                '-stimulus[0].yd', 500,
                                '-stimulus[0].zd', 0,
                                

                                # Dump the electrode file
                                '-stimulus[0].dump_vtx_file', 1
                                ]
                else:
                #Default Planar Wave: Set up electrode across the left wall and dump the electrode to a vtx file.             
                        cmd += ['-num_stim',  1,
                                

                                # Define the stimulus pulse 
                                '-stimulus[0].stimtype',   0,
                                '-stimulus[0].strength', args.strength,
                                '-stimulus[0].duration',   args.duration,
                                '-stimulus[0].npls',       args.pls,
                                #'-stimulus[0].geometry', 100,

                                # Dump the electrode file
                                '-stimulus[0].dump_vtx_file', 1,

                                #Define electrode based on mesh geomtery (left wall)
                                #'-stimulus[1].ctr_def', 1,
                                '-stimulus[0].x0', -50000,
                                '-stimulus[0].y0', -50000,
                                '-stimulus[0].z0', 0,

                                '-stimulus[0].xd', 500,
                                '-stimulus[0].yd', 100000,
                                '-stimulus[0].zd', 0,

                        ]       

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
        
                # Run example
                job.carp(cmd)

                # Calculate the APDs:
                LATs = txt.read(os.path.join(job.ID, 'init_acts_LATS-thresh.dat'))
                REPs = txt.read(os.path.join(job.ID, 'init_acts_REPS-thresh.dat'))
                APDs = REPs - LATs
                txt.write(os.path.join(job.ID, 'APDs.dat'), APDs)

if __name__ == '__main__':
    run()