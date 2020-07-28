# Projective Preferential Bayesian Optimization (PPBO)
[Projective Preferential Bayesian Optimization](https://arxiv.org/abs/2002.03113) implementation in Python (Mikkola, P., Todorović, M., Järvi, J., Rinke, P., and Kaski, S.; 2020). The PPBO framework lends itself to preference learning and prior elicitation, particularly in high-dimensional and otherwise complex settings.

The repository includes the interface for eliciting the user preferneces over Camphor/Cu(111) test case described in the "User experiment" section of the paper. 

## Instructions to experiment with Camphor/Cu(111)

*Procedure - Linux*

Clone the repository: <br />
#> git clone https://github.com/P-Mikkola/PPBO <br />
or download from the link above as a zip file, and unpack it

Go to the PPBO folder: <br />
#> cd PPBO

[optional] Get the virtualenv package for python3: <br />
#> pip3 install virtualenv

[optional] Create a python virtual environment <br />
#> python3 -m virtualenv env

[optional] Activate the virtual environment <br />
#> source env/bin/activate

Install python packages: <br />
#> pip3 install -r requirements.txt <br />
OR if you used virtualenv: <br />
#> pip install -r requirements.txt

Run the jupyter notebook system <br />
#> jupyter notebook

Find the Jupyter-notebook *Camphor-Copper.ipynb*, and click it.<br />
In the notebook, click ![Screenshot_2020-05-22 Camphor-Copper - Jupyter Notebook](https://user-images.githubusercontent.com/57790862/82723533-47d17600-9cd8-11ea-9978-46f4551af440.png)!







