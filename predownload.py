from pyannote.audio import Pipeline
from typing import Optional, Union
import torch
import sys

model_name="pyannote/speaker-diarization-3.0"
use_auth_token=sys.argv[1]
Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)
Pipeline.from_pretrained("pyannote/segmentation", use_auth_token=use_auth_token)