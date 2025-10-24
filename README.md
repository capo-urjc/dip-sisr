<img src="https://github.com/user-attachments/assets/53b39e41-0230-4d47-a404-f2daa143b3cd" alt="URJCLOG JPG-removebg-preview" width="20%">
<a href="https://onlinelibrary.wiley.com/doi/10.1111/exsy.70142" target="_blank">
    <img src="https://img.shields.io/badge/View%20Paper-DOI-blue?style=for-the-badge&logo=wiley" alt="View Paper DOI">
  </a>


## Table of contents

- [Instalation](#instalation)
- [Structure](#Structure)
- [Usage](#usage)
- [Licencse](#licencia)
- [Citation](#citation)

## Instalation

1. Clone the repository:

   ```bash
   git clone https://github.com/capo-urjc/dip-sisr

2. Navigate to the project directory:

   ```bash
   cd dip-sisr

3. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt


## Structure
The project structure must be as follows:

dip-sisr/

├── DATA/  
│   └── KODAK/  
│       └── 1.png  
├── datasets/  
│   └── NaturalColor.py  
├── folder_structure.py  
├── LICENSE  
├── losses/  
│   └── functions.py  
├── main.py  
├── models/  
│   ├── common.py  
│   ├── downsampler.py  
│   ├── resnet.py  
│   ├── skip.py  
│   ├── texture_nets.py  
│   └── unet.py  
├── README.md  
└── utils/  
    ├── cast_to_precision.py  
    ├── common_utils.py  
    ├── dip_utils.py  
    ├── normalize.py  
    ├── os_utils.py  
    ├── os_utils_dip.py  
    ├── quality_measures.py  
    ├── results_saver.py  
    └── torch_utils.py  


## Usage
To run this model, the file 
   ```bash
   main.py
   ```

must be executed.  

## Citation
```bibtex
@article{abalo2025unsupervised,
  title={Unsupervised Deep Image Prior-Based Neural Networks for Single Image Super-Resolution: Comparative Analysis and Modelling Guidelines},
  author={Abalo-Garc{\'\i}a, Alejandra and Ram{\'\i}rez, Iv{\'a}n and Schiavi, Emanuele},
  journal={Expert Systems},
  volume={42},
  number={11},
  pages={e70142},
  year={2025},
  publisher={Wiley Online Library}
}



