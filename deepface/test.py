from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import matplotlib
import time

# 将matplotlib的后端设置为'TkAgg'，以兼容不同的环境
matplotlib.use('TkAgg')

# 开始从电脑摄像头（设备0）捕获视频
cap = cv2.VideoCapture(0)

# 设置录制视频的帧率（根据你的摄像头能力调整）
frame_rate = 4.3

# 初始化OpenCV的VideoWriter，以指定的帧率保存视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_video.avi', fourcc, frame_rate, (640, 480))

# 设置一个matplotlib图形以显示实时情绪检测结果
plt.ion()  # 打开交互模式以实时更新
fig, ax = plt.subplots()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
bars = ax.bar(emotion_labels, [0]*7, color='lightblue')  # 为每种情绪初始化条形
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.title('Real-time Emotion Detection')
ax.set_xticklabels(emotion_labels, rotation=45)

# 初始化imageio writer以将实时图表更新保存为GIF
gif_writer = imageio.get_writer('emotion_chart.gif', mode='I', duration=0.1)

# 列表以存储每帧的累积情绪统计数据
emotion_statistics = []

# 更新实时图表的函数
def update_chart(detected_emotions, bars, ax, fig):
    # 清除当前轴并重新设置条形图
    ax.clear()
    ax.bar(emotion_labels, [detected_emotions.get(emotion, 0) for emotion in emotion_labels], color='lightblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence')
    plt.title('Real-time Emotion Detection')
    ax.set_xticklabels(emotion_labels, rotation=45)
    fig.canvas.draw()
    fig.canvas.flush_events()

# 开始计时以测量摄像头的活跃时间
webcam_start_time = time.time()

try:
    while True:
        ret, frame = cap.read()  # 从摄像头读取一帧
        if not ret:
            break  # 如果没有捕获到帧，则退出循环

        try:
            # 检测帧中的情绪
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if not results:
                continue
            
            # DeepFace.analyze 返回的是一个列表，取第一个元素
            result = results[0]
            emotions = result['emotion']
            current_emotions = {emotion: emotions.get(emotion, 0) / 100 for emotion in emotion_labels}

            # 如果检测到脸，则显示情绪并更新图表
            if current_emotions:
                emotion_statistics.append(current_emotions)
                
                # 找到最大概率的情绪
                emotion_type = max(current_emotions, key=current_emotions.get)
                emotion_score = current_emotions[emotion_type]
                emotion_text = f"{emotion_type}: {emotion_score:.2f}"

                # 检查并打印 region 的结构
                if 'region' in result:
                    region = result['region']
                    if isinstance(region, dict) and all(key in region for key in ['x', 'y', 'w', 'h']):
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                update_chart(current_emotions, bars, ax, fig)

                out.write(frame)  # 将帧写入视频文件

                # 将条形图的当前状态保存为GIF中的一帧
                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                print(f"Canvas width: {width}, height: {height}")
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
                print(f"Image size: {image.size}, expected size: {width * height * 4}")
                # 确保图像大小匹配
                if image.size == width * height * 4:
                    image = image.reshape((height, width, 4))[:, :, :3]
                    gif_writer.append_data(image)

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow('Emotion Detection', frame)  # 显示帧

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except KeyboardInterrupt:
    print("用户中断")

finally:
    webcam_end_time = time.time()  # 当摄像头窗口关闭时结束计时
    print(f"摄像头活跃时间: {webcam_end_time - webcam_start_time:.2f} 秒")

    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)

    out.release()
    gif_writer.close()

    emotion_df = pd.DataFrame(emotion_statistics, columns=emotion_labels)

    plt.figure(figsize=(10, 10))
    for emotion in emotion_labels:
        plt.plot(emotion_df[emotion].cumsum(), label=emotion)
    plt.title('随时间累积的情绪统计')
    plt.xlabel('帧')
    plt.ylabel('累积置信度')
    plt.legend()
    plt.savefig('cumulative_emotions.jpg')
    plt.close()