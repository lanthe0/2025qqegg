import pyaudio
import wave

def record_audio(output_filename, record_seconds, sample_rate=44100, chunk_size=1024, channels=1):
    # 初始化PyAudio
    audio = pyaudio.PyAudio()
    
    # 打开音频流
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    
    print("开始录音...")
    
    frames = []
    
    # 录制音频数据
    for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)
    
    print("录音结束")
    
    # 停止和关闭流
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # 保存音频文件
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

if __name__ == "__main__":
    output_filename = "output.wav"
    record_seconds = 5  # 录制5秒
    record_audio(output_filename, record_seconds)