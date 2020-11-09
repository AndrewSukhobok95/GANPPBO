# GANspace + PPBO project



## Idea

Probabilistic Interactive User Model for Interactive AI

Based on 2 projects:
- GANSpace
- Projective Preferential Bayesian Optimization (PPBO)

![original projects](md_files_and_imgs/ganspace_and_ppbo.png)

**Idea Formulation:** Adjust image generation model (GAN) to follow user’s preferences using Bayesian Optimization.

![original projects](md_files_and_imgs/ganppbo.png)

## Set Up

1. Install anaconda or miniconda
2. Create environment: `conda create -n ganppbo python=3.7`
3. Activate environment: `conda activate ganppbo`
4. Open project directory: `cd GANPPBO`
5. Install dependencies: `conda env update -f environment.yml --prune`
6. Setup submodules: `git submodule update --init --recursive`
7. Activate widget extension: `jupyter nbextension enable --py widgetsnbextension`

## Original project info

Currently, the code of ganspace projects is committed to this repository. 
Later ganspace is going to be transformed into submodule.

Original repositories:
- https://github.com/harskish/ganspace
- https://github.com/AaltoPML/PPBO

Origial README files:
- [GANSpace README file](ganspace/README.md)
- [PPBO README file](base_modules/PPBO/README.md)






