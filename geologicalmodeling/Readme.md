
# Gelogical-Numerical Modelling of emergent basins with GemPy
hola, este es el repositorio para los modelos geologicos con GEMPY


#### Installation of gempy libraries and visualization of a 3D geological model in Jupyter Notebook
Do it in six simple steps using Anaconda (open-source package and environment management system):

1. Install anaconda
   
2. Open the anaconda powershell prompt

3. Install the virtual environment "gempy".
NOTE: if it exists previously, you must remove it with:
```
>> conda remove --name gempy --all
```
```
>> conda create --name gempy python==3.10
```

5. enter the created environment´´
´´conda activate gempy´´

6. Install theano 
´´conda install theano´´

7. Install gempy
´´pip install gempy==2.3.1´´

8. Install Jupyter
´´pip install jupyterlab´´
´´pip install notebook´´

9. Install some dependencies for 3D visualization
´´sudo apt install xorg xorg gcc libx11-dev libxt-dev libxext-dev make libtirpc-dev´´
´´conda install -c conda-forge libstdcxx-ng´´
´´conda install -c conda-forge libffi´´









