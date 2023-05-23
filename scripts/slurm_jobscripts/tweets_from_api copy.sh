#!/bin/bash

#SBATCH --job-name=orgs
#SBATCH --account=patgwall
#SBATCH --partition=standard
#SBATCH --time=08-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --mail-type=BEGIN,END,FAIL


python scripts/download_conversation_tweets.py