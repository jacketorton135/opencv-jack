from sklearn.model_selection import train_test_split  # 匯入 sklearn 用於分割訓練和測試數據
import glob  # 匯入 glob 模組，用於檔案模式匹配
import numpy as np  # 匯入 NumPy 庫，用於數據處理
import os.path as path  # 匯入 os.path 模組，用於處理路徑
import os  # 匯入 os 模組，用於操作檔案和目錄
import cv2  # 匯入 OpenCV 庫，用於影像處理
import tensorflow as tf  # 匯入 TensorFlow 庫，用於建立深度學習模型

# 定義影像所在的目錄路徑
IMAGEPATH = 'images'  # 使用原始字串來防止反斜杠問題

# 獲取 IMAGEPATH 目錄下的所有子目錄名稱（每個子目錄代表一個類別）
dirs = [d for d in os.listdir(IMAGEPATH) if os.path.isdir(path.join(IMAGEPATH, d))]  # 確保只選擇資料夾

# 初始化空列表，用於存放影像資料和標籤
X = []
Y = []

print(f"資料夾中的子目錄: {dirs}")  # 輸出子目錄名稱列表

i = 0  # 初始化標籤計數器

# 遍歷每個子目錄（每個子目錄代表一個類別）
for name in dirs:
    print(f"處理類別: {name}")
    
    # 獲取子目錄下所有影像檔案的完整路徑
    file_paths = glob.glob(path.join(IMAGEPATH, name, '*.*'))
    
    # 檢查是否有影像檔案
    if len(file_paths) == 0:
        print(f"警告：'{name}' 類別下沒有影像檔案。")
    
    # 遍歷每個影像檔案
    for img_path in file_paths:
        img = cv2.imread(img_path)  # 讀取影像
        if img is None:  # 檢查是否成功讀取影像
            print(f"無法讀取影像檔案: {img_path}")  # 打印無法讀取的影像文件路徑
            continue  # 跳過該影像
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)  # 調整影像大小至 224x224
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將影像從 BGR 轉換為 RGB 格式
        X.append(im_rgb)  # 將影像資料添加到 X 列表中
        Y.append(i)  # 將影像的標籤（對應子目錄的索引）添加到 Y 列表中
    
    i += 1  # 每處理完一個子目錄，將標籤計數器加一

# 將影像資料和標籤列表轉換為 NumPy 陣列
X = np.asarray(X)
Y = np.asarray(Y)

# 確認是否有影像資料
print(f"影像數量: {X.shape[0]}")  # 輸出影像數量
print(f"標籤數量: {Y.shape[0]}")  # 輸出標籤數量

# 如果影像資料為空，終止程式
if X.shape[0] == 0:
    print("沒有有效的影像資料，請檢查影像資料夾")
    exit()

# 將影像資料轉換為浮點型，並進行標準化處理
X = X.astype('float32')
X = X / 255.0  # 將像素值歸一化至 [0, 1] 範圍

# 確保影像維度為 (樣本數量, 224, 224, 3)
X = X.reshape(X.shape[0], 224, 224, 3)

# 獲取分類數目，即子目錄的數量
category = len(dirs)

# 將資料集分為訓練集和測試集，測試集佔 5%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, stratify=Y)

# 輸出訓練集的資料形狀
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")

# 將標籤轉為 One-hot 向量
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)

# 建立資料增強（ImageDataGenerator）
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=25,
    width_shift_range=[-3, 3],
    height_shift_range=[-3, 3],
    zoom_range=0.3,
    horizontal_flip=True,  # 添加水平翻轉
    data_format='channels_last'
)

# 使用更小的批量大小
trainData = datagen.flow(x_train, y_train2, batch_size=32)  # 批量大小設定為 32

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(category, activation='softmax')  # 根據類別數量調整輸出層
])

# 設定學習率和優化器
learning_rate = 0.001
opt1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt1,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 訓練模型
history = model.fit(trainData, epochs=100)

# 保存模型權重
# 重新訓練模型並保存完整模型
model.save("model_complete.h5")  # 這會保存模型的架構和權重


# 測試
score = model.evaluate(x_test, y_test2, batch_size=32)  # 測試批量大小也減少至 32
print("Test score:", score)

# 預測
predict = model.predict(x_test)

# 顯示前幾個預測結果
for i in range(4):  # 顯示四個測試樣本的預測結果
    img = x_test[i]
    img_resized = cv2.resize(img * 255, (224, 224), interpolation=cv2.INTER_AREA).astype('uint8')
    im_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    predicted_class = np.argmax(predict[i])
    label = dirs[predicted_class]
    
    cv2.putText(im_bgr, f"{label} ({predict[i][predicted_class]:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(f"Predicted: {label}", im_bgr)
    cv2.waitKey(0)

cv2.destroyAllWindows()

