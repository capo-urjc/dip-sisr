<img src="https://github.com/user-attachments/assets/53b39e41-0230-4d47-a404-f2daa143b3cd" alt="URJCLOG JPG-removebg-preview" width="20%">


## Table of contents

- [Instalation](#instalation)
- [Structure](#Structure)
- [Usage](#usage)
- [Licencia](#licencia)

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



