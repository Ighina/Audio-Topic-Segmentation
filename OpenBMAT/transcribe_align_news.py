# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:32:42 2022

@author: User
"""

from speechbrain.pretrained import EncoderDecoderASR
import os

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech")

for root, dirs, files in os.walk("news_segments"):
    for file in files:
        raise NotImplementedError()
        transcription = asr_model.transcribe_file(os.path.join(root, file))
        0/0