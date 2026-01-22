# DataFlow 2026

```bash
git clone https://github.com/GinHikat/FomoKaguya2026.git

cd FomoKaguya2026

pip install -r requirement.txt

cp .env.example .env
```

secret/ggsheet_credentials.json and .env will be given later

After getting the .env and ggsheet credentials, run this to download the data

```bash
python download_asset.py

#Choose the id of dataset
```

Some notes about using shared_functions

```python
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.gg_sheet_drive import *
```

Then follow the instruction docstring in the beginning of the file @shared_functions/gg_sheet_drive
