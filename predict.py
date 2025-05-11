import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 모델 로드 정확도 (80.9%)
model = tf.keras.models.load_model('cats_and_dogs_small.h5',compile=False)

# 예측하고 싶은 이미지 경로
img_path = '2022.jpg'

# 이미지 로드 및 전처리
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)
img_array /= 255.0  # 모델 학습 시 rescale 했기 때문에 맞춰줌

# 예측
prediction = model.predict(img_array)

# 결과 출력
if prediction[0] < 0.5:
    print("이 이미지는 고양이(Cat)입니다.")
else:
    print("이 이미지는 개(Dog)입니다.")