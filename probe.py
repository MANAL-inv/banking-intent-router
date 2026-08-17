import platform, sys

print("machine:", platform.machine())
print("python :", sys.version.split()[0])

import numpy, sklearn, pandas
print("numpy  :", numpy.__version__)
print("sklearn:", sklearn.__version__)
print("pandas :", pandas.__version__)

import torch
print("torch  :", torch.__version__)

import transformers, sentence_transformers, datasets
print("transformers:", transformers.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
print("datasets:", datasets.__version__)

x = torch.randn(2, 3)
print("tensor op ok:", (x @ x.T).shape)
