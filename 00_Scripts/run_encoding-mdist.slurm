#! /bin/bash -l
#SBATCH --job-name=encoding-mdist
#SBATCH --account=a_mattingley
#SBATCH --partition=general
#SBATCH --array=0-38
#SBATCH --output=../06_Logs/encoding-mdist_perVar_%A_%a.out
#SBATCH --error=../06_Logs/encoding-mdist_perVar_%A_%a.out
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB

source /home/uqdrange/.bashrc
CONDAENV=/home/uqdrange/Software/intdec
conda activate $CONDAENV

echo 'START'
srun python /home/uqdrange/scratch/DR-RisSen-05/00_Scripts/DR-RisSen-05_analyses_EEG.py $SLURM_ARRAY_TASK_ID
# move data from TMPDIR to RDM
# rsync -avuz  --progress $TMPDIR/ /QRISdata/Q3911/DR-MulFea-26/ 
echo 'DONE'