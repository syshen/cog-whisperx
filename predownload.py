from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

model_name="pyannote/speaker-diarization-3.0"
use_auth_token="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"
Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token)