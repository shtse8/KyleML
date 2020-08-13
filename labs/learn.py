import os
from pathlib import Path
path = os.path.join(os.path.dirname(__file__), "weights", "a2c", "model.h5")
# Path(path).mkdir(parents=True, exist_ok=True)
print(os.path.dirname(path), __file__)