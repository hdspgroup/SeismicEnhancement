# SeismicEnhancement
![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

Repositorio oficial del proyecto de investigación VIE3925

### Unconditional Generation
| Model     | Notebook |
| --------- | -------- |
| DDPM      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdspgroup/SeismicEnhancement/blob/main/notebooks/diffusion/train_unconditional.ipynb)    |
| VAE       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdspgroup/SeismicEnhancement/blob/main/notebooks/variational/train_vae.ipynb)    |
| WGAN      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdspgroup/SeismicEnhancement/blob/main/notebooks/adversarial/train_gan.ipynb)    |

---
![](/imgs/banner.png)


### 3D Geological Modeling
| Model     | Notebook |
| --------- | -------- |

#### Installation of gempy libraries and visualization of a 3D geological model in Jupyter

Do it in six simple steps using Anaconda (open-source package and environment management system):

1. Open the anaconda powershell prompt
2. Install the virtual environment "gempy".
NOTE: if it exists previously, you must remove it with: conda remove --name gempy --all
>> conda create --name gempy python==3.10

3. enter the created environment
>> conda activate gempy

4. Install theano 
>> conda install theano

5. Install gempy
>> pip install gempy

6. Install Jupyter
>>pip install jupyterlab
>>pip install notebook
