import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# 데이터 경로 설정
base_dir = 'output'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# 예: PetImages/Cat 또는 PetImages/Dog 폴더
from PIL import Image
import os

def clean_corrupt_images(image_folder):
    num_deleted = 0
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        try:
            with Image.open(img_path) as img:
                img.verify()  # 파일 구조 검사
            with Image.open(img_path) as img:
                img.load()  # 실제 데이터까지 로딩해보기 (여기서 Truncated(잘림) File Read 감지됨)
        except Exception as e:
            print(f"손상된 파일 발견: {img_path} (사유: {e})")
            try:
                os.remove(img_path)
                num_deleted += 1
            except Exception as del_err:
                print(f"삭제 실패: {del_err}")
    print(f"🧹 총 {num_deleted}개의 손상된 이미지 삭제 완료")

# 예시 실행
clean_corrupt_images('output/train/Cat')
clean_corrupt_images('output/train/Dog')
clean_corrupt_images('output/val/Cat')
clean_corrupt_images('output/val/Dog')


# 이미지 데이터 전처리 및 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# CNN 모델 구축
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

# 모델 저장
model.save('cats_and_dogs_small.h5')