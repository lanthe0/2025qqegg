# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
用于通过ASRT语音识别系统预测一次语音文件的程序
"""

import os

# 导入语音模型相关的模块
from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义音频参数
AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
OUTPUT_SIZE = 1428

# 初始化声学模型
sm251bn = SpeechModel251BN(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
)

# 初始化特征提取器
feat = Spectrogram()

# 初始化语音识别模型
ms = ModelSpeech(sm251bn, feat, max_label_length=64)

# 加载声学模型
ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')

# 通过声学模型识别语音文件
res = ms.recognize_speech_from_file('233.wav')
print('*[提示] 声学模型语音识别结果：\n', res)

# 初始化语言模型
ml = ModelLanguage('model_language')

# 加载语言模型
ml.load_model()

# 将识别结果的拼音转换为文本
str_pinyin = res
res = ml.pinyin_to_text(str_pinyin)
print('语音识别最终结果：\n', res)