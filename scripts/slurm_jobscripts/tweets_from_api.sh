#!/bin/bash

#SBATCH --job-name=download_journalists
#SBATCH --time=08-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --output=log.out
#SBATCH --mail-type=BEGIN,END,FAIL


python scripts/download_journalist_tweets.py
