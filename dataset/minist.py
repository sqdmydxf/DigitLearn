import gzip
import struct
from pathlib import Path

from icecream import ic

current_dir = Path(__file__).parent

def load_data():
    with gzip.open(current_dir / 'train-images-idx3-ubyte.gz', 'rb') as f:
        magic, num_images = struct.unpack('>II', f.read(8))
        ic(magic, num_images)
        num_rows, num_cols = struct.unpack('>II', f.read(8))
        ic(num_rows, num_cols)

load_data()

