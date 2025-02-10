# import cupy
# print("CUDA Available:", cupy.cuda.is_available())

# import spacy
# spacy.require_gpu()
# print("spaCy is using GPU:", spacy.prefer_gpu())

from numba import cuda
print(cuda.gpus)

