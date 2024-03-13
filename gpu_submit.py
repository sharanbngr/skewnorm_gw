import sys, os
from configparser import ConfigParser


if __name__ == "__main__":
    if len(sys.argv) != 2:
            raise ValueError('Provide the config file as an argument')
    else:

        config = ConfigParser()
        config.read(sys.argv[1])

        runargs = {}

        # collate args
        spin_model = config.get('model', 'spin_model')
        backend = config.get('params', 'backend')
        runidx = config.get('params', 'rundix')


        outdir = './' + spin_model + '_' + backend + '_' + runidx
        configfile = sys.argv[1]

        # create directory and copy the config file
        os.system('mkdir -p ' + outdir)
        os.system('cp '  + configfile  + ' ' + outdir + '/config.ini')


        submitfile = file1 = open(outdir + '/submit.sh', 'w')


        submitfile.write('#!/bin/sh \n')
        submitfile.write('#SBATCH -A b1094 \n')
        submitfile.write('#SBATCH -p ciera-gpu \n')
        submitfile.write('#SBATCH --job-name=skewnorm_' + runidx + ' \n')
        submitfile.write('#SBATCH --time=2:00:00\n')
        submitfile.write('#SBATCH --output=' + outdir + '/skewnorm_slurm.out\n')
        submitfile.write('#SBATCH --error=' + outdir + '/skewnorm_slurm.err\n')
        submitfile.write('#SBATCH --mem=20G\n')
        submitfile.write('#SBATCH --gres=gpu:a100:1\n')
        submitfile.write('#SBATCH --nodes=1 \n')
        submitfile.write('#SBATCH --ntasks-per-node=1\n')
        submitfile.write('\n')
        submitfile.write('mamba init\n')
        submitfile.write('mamba activate numpyro-test\n')
        submitfile.write('module use /hpc/software/spack_v17d2/spack/share/spack/modules/linux-rhel7-x86_64/\n')
        submitfile.write('module load cuda/11.2.2-gcc\n')
        submitfile.write('\n')
        submitfile.write('cd /projects/p31963/sharan/pop/skewnorm_gw\n')
        submitfile.write('python skewnorm.py ' + outdir +  '/config.ini\n')
        submitfile.close()

        os.system('chmod u+x ' + outdir + '/submit.sh')

        print('submit file created ...')
        print('Now submit with sbatch ' + outdir + '/submit.sh')
