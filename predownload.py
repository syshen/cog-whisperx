from pyannote.audio import Pipeline, Model
import torch
import sys

model_name="pyannote/speaker-diarization-3.0"
use_auth_token=sys.argv[1]
Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)
Model.from_pretrained("pyannote/segmentation", use_auth_token=use_auth_token,  strict=False)
