
# Gelogical modeling with GemPy
Geological modeling allows the identification of key structural and stratigraphic features that can host hydrocarbon accumulations in evolving basins, such as folds, faults, anticlines and stratigraphic traps. Therefore, in this section the basic functions of GemPy will be discussed to generate geological-numerical models with structural and stratigraphic complexity, integrating an analysis of quantitative techniques and metrics to determine their quality and reliability.

#### Installation of gempy libraries and visualization of a 3D geological model in Jupyter Notebook
Do it in six simple steps using Anaconda (open-source package and environment management system):

1. Install anaconda
   
2. Open the anaconda powershell prompt

3. Install the virtual environment "gempy".
NOTE: if it exists previously, you must remove it with:
```
conda remove --name gempy --all
```
```
conda create --name gempy python==3.10
```

5. enter the created environment
```
conda activate gempy
```

7. Install theano 
```
conda install theano
```

8. Install gempy
```
pip install gempy==2.3.1
```

10. Install Jupyter
```
pip install jupyterlab
```
```
pip install notebook
```

12. Install some dependencies for 3D visualization

```
sudo apt install xorg xorg gcc libx11-dev libxt-dev libxext-dev make libtirpc-dev
```
```
conda install -c conda-forge libstdcxx-ng
```
```
conda install -c conda-forge libffi
```

You can find more information in GemPy repository: https://github.com/gempy-project/gempy






