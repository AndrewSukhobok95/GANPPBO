# GANspace + PPBO project

on Pannikki:
ps aux | grep 12345
jupyter notebook --no-browser --port=12345

on kosh:
ssh -L 12345:localhost:12345 -N -f -l sukhoba1 rename

on your computer:
ssh -L 12345:localhost:12345 -N -f -l sukhoba1 kosh.aalto.fi

## Set Up

1. Install anaconda or miniconda
2. Create environment: `conda create -n ganppbo python=3.7`
3. Activate environment: `conda activate ganppbo`
4. Install dependencies: `conda env update -f environment.yml --prune`
5. Activate widget extension: `jupyter nbextension enable --py widgetsnbextension`

## Original project info

Currently, the code of both projects is commited to this repository. 
Later ganspace is going to be transformed into submodule. 
PPBO will stay a part of this code since it was changed.

Original repositories:
- https://github.com/harskish/ganspace
- https://github.com/AaltoPML/PPBO

Origial README files:
- [GANSpace README file](ganspace/README.md)
- [PPBO README file](PPBO/README.md)

Additional info:
- [Changes performed to PPBO project](md_files/ppbo_changes.md)





