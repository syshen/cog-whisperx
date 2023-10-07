# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/src/hf_models/hub'
os.environ['TORCH_HOME'] = '/src/torch_models'
os.environ['PYANNOTE_CACHE'] = '/src/torch_models/pyannote'
from cog import BasePredictor, Input, Path
import torch
import whisperx
import json
import time


class Predictor(BasePredictor):
    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        device: str = Input(description="Device name", default="cuda"),
        language: str = Input(description="Language", default="en"),
        model_name: str = Input(description="Whisper model", default="large-v2"),
        compute_type: str = Input(description="Compute type", default="float16"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        only_text: bool = Input(description="Set if you only want to return text; otherwise, segment metadata will be returned as well.", default=False),
        hf_token: str = Input(description="Hugging Face Access Token to access PyAnnote gated models", default=None),
        diarize: bool = Input(description="Apply diarization to assign speaker labels to each segment/word (default: False)", default=False),
        min_speakers: int = Input(description="Minimum number of speakers to in audio file", default=None),
        max_speakers: int = Input(description="Maximum number of speakers to in audio file", default=None),
        debug: bool = Input(description="Print out memory usage information.", default=False)
    ) -> str:
        self.device = device
        self.model = whisperx.load_model(model_name, device=self.device, language=language, compute_type=compute_type)

        # self.alignment_model, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)
        """Run a single prediction on the model"""
        with torch.inference_mode():
            t1 = time.perf_counter(), time.process_time()

            result = self.model.transcribe(str(audio), batch_size=batch_size) 
            if debug:
              print (result)

            t2 = time.perf_counter(), time.process_time()
            if debug:
              print(f"> Finish transcription")
              print(f"  Real time: {t2[0] - t1[0]:.2f} seconds")
              print(f"  CPU time: {t2[1] - t1[1]:.2f} seconds")

            if diarize:
              if hf_token is None: 
                print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
              else:
                
                self.model = None
                diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=self.device)
                diarize_segments = diarize_model(str(audio), min_speakers=min_speakers, max_speakers=max_speakers)

                t3 = time.perf_counter(), time.process_time()
                if debug:
                  print("> Finish diarization")
                  print(f"  Real time: {t3[0] - t2[0]:.2f} seconds")
                  print(f"  CPU time: {t3[1] - t2[1]:.2f} seconds")

                result = whisperx.diarize.assign_word_speakers(diarize_segments, result)
                if debug:
                  print(result)
                if only_text:
                  return ''.join([f"{val['speaker']}:\n{val['text'].strip()}\n\n" for val in result["segments"]])

            if only_text:
                return ''.join([val['text'] for val in result["segments"]])

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        return json.dumps(result['segments'])

