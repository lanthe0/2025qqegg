import cv2
import torch
import numpy as np
from torchvision import transforms
from models import VGG  # 确保 models.py 在同一目录下或正确导入
import torch.nn.functional as F

# 表情类别
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# 预处理转换（确保是 3 通道输入）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


# 加载模型

def load_model():
    model = VGG('VGG19')

    checkpoint = torch.load('models/PrivateTest_model.t7', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['net'])

    model.eval()

    return model


# 初始化摄像头

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")

    exit()

# 加载模型

model = load_model()

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ 无法获取摄像头画面")

        break

    # 转为灰度图

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(gray, (48, 48))

    # ⚠️ 关键修正：将灰度图转换为 3 通道（伪 RGB）

    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    # 预处理图像

    face_tensor = transform(face).unsqueeze(0)  # 增加 batch 维度

    # 模型预测

    with torch.no_grad():

        output = model(face_tensor)

        probabilities = F.softmax(output, dim=1)  # 计算 softmax 概率

        prediction = torch.argmax(probabilities, 1).item()

    # 获取表情类别

    predicted_emotion = class_names[prediction]

    # 显示预测结果

    cv2.putText(frame, predicted_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Facial Expression Recognition", frame)

    # 按 'q' 退出

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源

cap.release()

cv2.destroyAllWindows()