import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector
from preprocessing.normalization.functionals import within_subject_functional_normalization


from constants import file_path, meta_columns, observation_columns

log_dir = "logs"
metadata_filename = "metadata.tsv"
tensor_filename = "tensor.tsv"

df = pd.read_csv(file_path)

meta = df[meta_columns]
vec = df[observation_columns]

participant_ids = meta["Participant"].values

vec = within_subject_functional_normalization(vec, participant_ids, "standard")


meta.to_csv(os.path.join(log_dir, metadata_filename),
            sep='\t', index=False)

vec.to_csv(os.path.join(log_dir, tensor_filename),
            sep='\t', index=False, header=False)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.metadata_path = metadata_filename
embedding.tensor_path = tensor_filename
projector.visualize_embeddings(log_dir, config)
