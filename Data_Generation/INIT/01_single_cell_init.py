#Single Cell initialization script

import os
import glob
import shutil
from datetime import date
import re
import numpy as np
from carputils import settings
from carputils import tools

def parser():
    parser = tools.standard_parser()
    group = parser.add_argument_group('script specific options')

    group.add_argument('--imp', type=str, default='AlievPanfilov',
                       choices=['tenTusscherPanfilov', 'Courtemanche', 'AlievPanfilov', 'AlievPanfilovDynamic'],
                       help='Ionic model (default is %(default)s)')
    group.add_argument('--params',
                        default = '',
                        help = 'Ionic model parameters')
    group.add_argument('--pls', type=float, default=100,
                       help='Number of pulses to test (default is %(default)s )')
    group.add_argument('--bcl', type=float, default=500,
                       help='BCL for sinus rhythm (default is %(default)s ms, 120bpm )')
    group.add_argument('--strength', type=float, default=250,
                       help='Strength for the transmembrane current stimulation (default is %(default)s muA)')
    group.add_argument('--duration', type=float, default=2,
                       help='Duration for the stimulation (default is %(default)s  ms)')
    group.add_argument('--vis_var',
                       default = 'V',
                       type = str,
                       nargs = '+',
                       help = 'Variable(s) to visualize, if empty Vm will be plotted.\n'
                              'Separate multiple variables with spaces.')
    group.add_argument('--overlay',
                        action = 'store_true',
                        help = 'Overlays all existing experiments. ')
    return parser

def jobID(args):
    today = date.today()
    return '01_single_cell_init_{}_{}_{}bcl_{}pls_{}params'.format(today.isoformat(), args.imp, args.bcl, args.pls, args.params)


def replace_sv_str(sv_names):
    for i, item in enumerate(sv_names):
        sv_names[i] = item.replace("sv->", "")

def get_corrected_vis_var(variables, sv_names):
    absent_variables = []
    for input_item in variables:
        if input_item not in sv_names:
            absent_variables.append(input_item)

    if len(absent_variables) == 0:
        return variables
    else:
        msg1 = f'The following input variables are missing from the trace header file:\n{absent_variables}\n\n'
        msg2 = f'Your input was:\n{variables}\n\n'
        msg3 = f'The trace header file contains:\n{sv_names}\n'
        
        new_variables = input(msg1 + msg2 + msg3 + '\n' + 'Please input correct variables(omitting any "sv->") separated by whitespaces.\n').split()
        return get_corrected_vis_var(new_variables, sv_names)

def visualization(args, path):
    import matplotlib.pyplot as plt
    variables = args.vis_var

    Vidx = []

    if args.overlay:
        txt_files = []
        dat_files = []
        Vidx_lst = []
        txt_legend = []
        sv_names_lst = []
        sv_data_lst = []
        directories = [x for x in os.listdir('.') if os.path.isdir(x)]

        for dir in directories:
            txt_files.append(glob.glob(os.path.join(dir, '*_trace_header.txt')))
            dat_files.append(glob.glob(os.path.join(dir, '*.dat')))
            pattern = r''+re.escape(args.EP)+ r"(.*)$"
            matches = re.search(pattern, str(dir), re.DOTALL)
            txt_legend.append(matches.group())
        for txtitem in txt_files:
            sv_names_lst.append(open(txtitem[0]).read().splitlines())
        for datitem in dat_files:
            sv_data_lst.append(np.loadtxt(datitem[0]))

        # rename state variables in-place
        for index_lst, sv_names in enumerate(sv_names_lst):
            replace_sv_str(sv_names)
            variables = get_corrected_vis_var(variables, sv_names)
            for index, item in enumerate(sv_names):
                n_item = item.replace("sv->", "")
                sv_names[index] = n_item

                # shall this item be plot?
                if n_item in variables:
                    Vidx.append(index)
            sv_names_lst[index_lst] = sv_names
            Vidx_lst.append(Vidx)


        fig, axes = plt.subplots(1, len(Vidx_lst), sharex=True, sharey=False)
        axes = [axes] # for the case just one dataset was found, we need to make it iterable by making it a list
        idx = 0
        for i, ax in enumerate(axes):
            for j in range(0, len(Vidx_lst)):
                ax.plot(sv_data_lst[j][:, 0], sv_data_lst[j][:, Vidx[idx]+1])
                ax.set_title(sv_names_lst[0][Vidx[i]])
                ax.set_xlabel('Time (ms)')
            idx += 1
        plt.legend(txt_legend)
        plt.show()

    else:
        # Read trace file
        trace_file = glob.glob(os.path.join(path, "AlievPanfilov_*.txt"))
        if len(trace_file) == 0:
            raise RuntimeError("No Aliev-Panfilov trace file found.")
        trace = np.loadtxt(trace_file[0])
        time = trace[:,0]
        Vm   = trace[:,1]
        V    = trace[:,2]

        # Read AP stats
        stats_file = glob.glob(os.path.join(path, "*AP_stats.dat"))

        if len(stats_file):
            ap_stats = np.loadtxt(
                stats_file[0],
                comments="#",
                usecols=(0,2,3)
            )
            beat = ap_stats[:,0]
            steady = ap_stats[:,1]
            APD = ap_stats[:,2]
        else:
            beat = None

        fig, ax = plt.subplots(3,1, figsize=(10,9))
        # Vm
        ax[0].plot(time, Vm)
        ax[0].set_ylabel("Vm (mV)")
        ax[0].set_title("Membrane Voltage")

        # Recovery variable
        ax[1].plot(time, V)
        ax[1].set_ylabel("V")
        ax[1].set_title("Recovery Variable")

        # APD convergence
        if beat is not None:
            ax[2].plot(beat, APD, 'o-')
            ax[2].set_xlabel("Beat")
            ax[2].set_ylabel("APD (ms)")
            ax[2].set_title("APD Convergence")
            ax[2].grid(True)

        plt.tight_layout()
        plt.show()

        '''
        # no overlay
        txt_files = glob.glob(os.path.join(path, '*.txt'))
        dat_files = glob.glob(os.path.join(path, '*.dat'))
        sv_names = open(txt_files[0]).read().splitlines()
        sv_data = np.loadtxt(dat_files[0])

        # rename state variables in-place
        replace_sv_str(sv_names)
        variables = get_corrected_vis_var(variables, sv_names)
        for index, item in enumerate(sv_names):
            n_item = item.replace("sv->", "")
            sv_names[index] = n_item

            # shall this item be plot?
            if n_item:
                if n_item in variables:
                    Vidx.append(index)

        if len(Vidx) > 1:
            fig, axes = plt.subplots(1, len(Vidx), sharex = True, sharey = False)

            for i, ax in enumerate(axes.flatten()):
                ax.plot(sv_data[:,0], sv_data[:, Vidx[i]+1])
                ax.set_title(sv_names[Vidx[i]])
                ax.set_xlabel('Time (ms)')
            plt.show()
        else:
            plt.plot(sv_data[:,0], sv_data[:, Vidx[0]+1])
            plt.xlabel('Time (ms)')
            plt.title(sv_names[Vidx[0]])
            plt.legend([args.EP+args.EP_par+args.plug_in+args.plug_par])
            plt.grid(True)
            plt.show()

        '''
    return



@tools.carpexample(parser, jobID)
def run(args, job):
    cmd = [
        settings.execs.BENCH,
        '--imp', args.imp,
    ]

    if args.params is not None:
            cmd += ['--imp-par', args.params]
        
    tend = args.pls*args.bcl

    cmd += ['--stim-curr', str(args.strength),
            '--numstim', str(args.pls),
            '--bcl', str(args.bcl),
            '--stim-dur', str(args.duration),
            '--duration', str(tend),
            '--save-ini-file', 'INIT_{}_BCL_{}_Model_{}_Params.sv'.format(str(args.bcl), args.imp, args.params),
            '--save-ini-time', tend,
            '--fout={}_{}'.format(str(args.imp),str(args.params)),
            '--APstatistics']

        
    job.mpi(cmd, 'Running {}'.format(args.imp))
   
    # move results to folder with job.ID name
    for file in glob.glob(r'*.txt'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.dat'):
        shutil.move(file, job.ID)
    for file in glob.glob(r'*.sv'):
        shutil.move(file, job.ID)

    # Do visualization
    if args.visualize and not settings.platform.BATCH:   
        # detailed visualization of all relevant traces
        if not args.dry:
            visualization(args,job.ID)

if __name__ == '__main__':
    run()


        
        