import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from openpyxl import Workbook


IMG_SIZE = (64,64)
BATCH_SIZE = 8

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

model = models.Sequential([
    layers.Input(shape=(64,64,3)),

    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128,3,activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

class ExcelLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.append([
            "Epoch",
            "Train_Accuracy",
            "Train_Loss",
            "Val_Accuracy",
            "Val_Loss"
        ])

    def on_epoch_end(self, epoch, logs=None):
        self.ws.append([
            epoch+1,
            format(logs["accuracy"], ".4f"),
            format(logs["loss"], ".4f"),
            format(logs["val_accuracy"], ".4f"),
            format(logs["val_loss"], ".4f"),
        ])
        self.wb.save("training_log.xlsx")
        print("✔ Log Excel tersimpan")

class StopAtOne(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= 1.0:
            print("\n Accuracy sudah 1.0 — training dihentikan")
            self.model.stop_training = True

excel_logger = ExcelLogger()
stopper = StopAtOne()

print("\nTRAINING START\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=1000,   
    callbacks=[excel_logger, stopper]
)

model.save("model_sepatu_sendal.keras")

print("\n✅ TRAINING SELESAI")
print("✅ Model tersimpan")
print("✅ Excel log tersimpan")
