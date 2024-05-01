
# Modelamiento geológico-numérico con GemPy

El modelamiento geológico permite la identificación de características estructurales y estratigráficas clave que pueden albergar acumulaciones de hidrocarburos en cuencas en evolución, como pliegues, fallas, anticlinales y trampas estratigráficas. Por eso, en este apartado se discutirán las funciones básicas de GemPy para generar modelos geológico-numéricos con complejidad estructural y estratigráfica, integrando un análisis de técnicas y métricas cuantitativas para determinar su calidad y confiabilidad.

#### Instalación de la librería GemPy y visualización de un modelo geológico 3D en Jupyter Notebook
Hazlo en estos sencillos pasos utilizando Anaconda (paquete de código abierto y sistema de gestión de entornos):

1. Instala Anaconda (https://www.anaconda.com/download)
   
2. Abre la terminal de Anaconda
   
4. Instala un ambiente virtual llamado "gempy".
NOTA: si ya existe previamente debes removerlo con:
```
conda remove --name gempy --all 
```
```
conda create --name gempy python==3.10 #crearlo
```

5. Ingresa al nuevo ambiente
```
conda activate gempy
```

7. Instala la librería theano
```
conda install theano
```

8. Instala la versión de GemPy
```
pip install gempy==2.3.1
```

10. Instala Jupyter
```
pip install jupyterlab
```
```
pip install notebook
```

12. Instala otras librerías para la visualización 3D

```
sudo apt install xorg xorg gcc libx11-dev libxt-dev libxext-dev make libtirpc-dev
```
```
conda install -c conda-forge libstdcxx-ng
```
```
conda install -c conda-forge libffi
```

Puedes encontrar más información en el repositorio oficial de GemPy: https://github.com/gempy-project/gempy

## Metodología para la generación de modelos geológicos con diferentes propiedades geofísicas

<p align="center">
<img src="https://github.com/hdspgroup/SeismicEnhancement/blob/c5b83807e60b2b858a555fa01750c5bd55bc243e/geologicalmodeling/images/methodology.png" width="1000">
</p>




