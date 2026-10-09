#!/bin/bash
# Out-of-sample backtest, 8 stocks, 2017Q2: fit April -> test May, fit May -> test June,
# for E1m (K = 4, 1 ms dead time; BAC K = 3) and E0 (K = 4), at execution latencies 0, 0.5, 2, 5 ms.
# One array task per stock.
#
#     sbatch scripts/cluster/oos_panel.sbatch
#
# Output: ~/hawkes-taq-data/stopping/fits/*.pkl (TAQ-derived, not committed) and
#         results/*.csv (aggregates; commit these).
#
#SBATCH --job-name=stop_oos
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stop_oos_%a_%A.o
#SBATCH --error=logs/stop_oos_%a_%A.e
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hyoeun@illinois.edu

set -euo pipefail
mkdir -p logs results
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HAWKES_TAQ=${HAWKES_TAQ:-~/hawkes-taq}

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
NBBO=$DATA/nbbo_2017Q2
FITS=$DATA/stopping/fits
K=4; [ "$SYM" = BAC ] && K=3

for DEF in E1m E0; do
    for M in 201704 201705; do
        python scripts/fit_month.py --nbbo $NBBO --sym $SYM --month $M --definition $DEF --K $K --out $FITS &
    done
    wait
done

for DEF in E1m E0; do
    for PAIR in 201704:201705 201705:201706; do
        FM=${PAIR%%:*}; TM=${PAIR##*:}
        for LAT in 0 0.0005 0.002 0.005; do
            python scripts/run_backtest.py --nbbo $NBBO --sym $SYM --test-month $TM \
                --fit $FITS/${SYM}_${DEF}_K${K}_${FM}.pkl --exec-lat $LAT --out results &
        done
        wait
    done
done
