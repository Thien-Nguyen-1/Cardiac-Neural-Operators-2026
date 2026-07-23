#!/usr/bin/env python

import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
EXAMPLE_DESCRIPTIVE_NAME = 'Adjusting action potential duration'
EXAMPLE_AUTHOR = 'Jason Bayer <jason.bayer@ihu-liryc.fr>'
EXAMPLE_DIR = os.path.dirname(__file__)
GUIinclude = True

from datetime import date
from carputils import settings
from carputils import tools

def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')
    # Add arguments
    group.add_argument('--Mode',
                        default = 'default',
                        choices = ['default', 'adjust'],
                        help = 'Use default or adjustment parameters')
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
    group.add_argument('--nbeats',
                        type = int,
                        default = 1,
                        help = 'Number of beats to before outputing APD')
    group.add_argument('--bcl',
                        type = int,
                        default = 500,
                        help = 'Basic cycle length of pacing')
    return parser

def jobID(args):
    today = date.today()
    ID = '05_APD_Adjustment_{}_{}_{}_Gil_{}_Gel_{}_bcl_{}_params'.format(today.isoformat(), args.IMP, args.Gil, args.Gel, args.bcl, args.im_param)
    return ID


@tools.carpexample(parser, jobID, clean_pattern='{}*|(.trc)'.format(date.today().year))

def run(args, job):

    cmd  = tools.carp_cmd(os.path.join(EXAMPLE_DIR, 'Mesh_1.5cm_Cable_100um', 'sim.par'))

    cmd += ['-stim[0].elec.vtx_file', os.path.join(EXAMPLE_DIR, 'Mesh_1.5cm_Cable_100um', 'stim')]
    cmd += ['-meshname', os.path.join(EXAMPLE_DIR, 'Mesh_1.5cm_Cable_100um', 'mesh')]

    if args.Mode == 'default':
        simid = os.path.join(job.ID, 'APD_default')
        cmd += ['-simID', simid,
                '-imp_region[0].im', args.IMP,
                '-stim[0].ptcl.bcl', args.bcl,
                '-stim[0].ptcl.npls', args.nbeats,
                '-stim[0].pulse.strength', 250,
                '-tend', args.bcl*args.nbeats+2500 ]
       
    if args.Mode == 'adjust':
        simid = os.path.join(job.ID, 'APD' + args.im_param)
        cmd += ['-simID', simid,
                '-imp_region[0].im', args.IMP,
                '-imp_region[0].im_param', str(args.im_param),
                '-stim[0].ptcl.bcl', args.bcl,
                '-stim[0].ptcl.npls', args.nbeats,
                '-stim[0].pulse.strength', 250,
                '-tend', args.bcl*args.nbeats+ 500]
        
    #Calculate the APDs for 90% repolarization:
    cmd += ['-num_LATs', 2, 
            
            '-lats[0].ID ', 'LATS',
            '-lats[0].all', 0,
            '-lats[0].measurand', 0,
            '-lats[0].threshold',  '-10',
            '-lats[0].mode', 0,

            '-lats[1].ID',         'REPS',
            '-lats[1].all',         0,
            '-lats[1].measurand',  0,
            '-lats[1].threshold', '-70',
            '-lats[1].mode', 1,
            ]
    
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
    #Run simulation
    job.carp(cmd)

    #calculate the apd
    cmd = [settings.execs.igbapd,
           '-t', (args.nbeats-1)*args.bcl,
           '--repol=90',
           '--vup=-10',
           '--peak-value=plateau',
           '--plateau-start=15',
           '--output-file={}'.format(os.path.join(simid, 'apd.dat')),
            os.path.join(simid, 'vm.igb')]

    
    job.bash(cmd)


    # Add in CV calculation:
    activation_file = os.path.join(simid, 'init_acts_LATS-thresh.dat')

    # Check if the activation times file exists; if not, create it from example values
    if os.path.exists(activation_file):
        with open(activation_file, 'r') as f:
            times = [float(line.strip()) for line in f if line.strip()]
    else:
        print("No LAT file found")
        

    if len(times) >= 2:
        activation_time = times[-1] - times[0]
        cable_length_cm = 1.5 # cm
        conduction_velocity = cable_length_cm / activation_time  # cm/ms

        # Convert to m/s for standard units
        conduction_velocity_mps = conduction_velocity * 10  # (cm/ms) × 10 = m/s

        # Save to results file
        cv_file = os.path.join(simid, 'conduction_velocity_results.txt')
        with open(cv_file, 'w') as f:
            f.write("=== Conduction Velocity Calculation ===\n")
            f.write(f"Number of nodes: {len(times)}\n")
            f.write(f"First activation time: {times[0]:.6f} ms\n")
            f.write(f"Last activation time: {times[-1]:.6f} ms\n")
            f.write(f"Total activation time: {activation_time:.6f} ms\n")
            f.write(f"Cable length: {cable_length_cm:.2f} cm\n")
            f.write(f"Conduction velocity: {conduction_velocity_mps:.4f} m/s\n")
        print(f"Conduction velocity saved to {cv_file}")

    else:
        print("Not enough activation times to calculate conduction velocity.")


    # GUI
    if args.webGUI:
        folderNameExp = f'/experiments/03_study_prep_APD_{job.ID}'
        datagui = os.path.join(simid, 'apd.dat')
        os.mkdir(folderNameExp)
        cmdMesh = f'meshtool collect -imsh=/tutorials/02_EP_tissue/03_study_prep_APD/Mesh_1.5cm_Cable/mesh -omsh={folderNameExp}/{job.ID} -nod={datagui} -ifmt=carp_txt -ofmt=ens_bin'
        outMesh = os.system(cmdMesh)
        if(outMesh == 0):
            print('Meshtool - conversion successfully')
        else:
            print('Meshtool - conversion failure')
    
    #Visualize with meshalyzer
    elif args.visualize and not settings.platform.BATCH:
        geom = os.path.join(EXAMPLE_DIR, 'Mesh_1.5cm_Cable_100um', 'mesh')
        data = os.path.join(simid, 'apd.dat')
        view = os.path.join(EXAMPLE_DIR, 'Mesh_1.5cm_Cable_100um', 'apd_state.mshz')
        job.meshalyzer(geom, data, view)

        # ---- APD plot ----
        apd_file = os.path.join(simid, "apd.dat")

        if os.path.exists(apd_file):
            apd = np.loadtxt(apd_file)

            # If only one column, assume APD values only
            if apd.ndim == 1:
                x = np.linspace(0, 1.5, len(apd))   # cable position (cm)
                apd_values = apd
            else:
                x = apd[:, 0]
                apd_values = apd[:, 1]

            plt.figure(figsize=(8,4))
            plt.plot(x, apd_values, lw=2)
            plt.xlabel("Cable Position (cm)")
            plt.ylabel("APD90 (ms)")
            plt.title("Action Potential Duration Along Cable")
            plt.tight_layout()
            plt.savefig(os.path.join(simid, "APD_distribution.png"), dpi=300)
            plt.close()
    

    # move results to folder with job.ID name
    for file in glob.glob(r'*.txt'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.dat'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.sv'):
        shutil.move(file, job.ID)

if __name__ == '__main__':
    run()
