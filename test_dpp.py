import numpy as np
from dpp_functions import *

items = np.random.normal(size=(100, 25))

print sample_from_dpp(items)
