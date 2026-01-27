# DataFlow 2026

```bash
git clone https://github.com/GinHikat/FomoKaguya2026.git

cd FomoKaguya2026

pip install -r requirement.txt

cp .env.example .env
```

secret/ggsheet_credentials.json and .env will be given later

Run this inside the main folder to download the data

```bash
python set_up_dataset.py
```

Some notes about using the processing functions (you may not need this tutorial)

```python
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from processing.ingestion_pipeline import *

processor = Processor()
```
