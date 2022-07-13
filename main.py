import numpy as np
from reader import read_train_set, read_test_set
from keras.applications.vgg16 import VGG16
from keras.metrics import categorical_accuracy
from sklearn.metrics import classification_report, accuracy_score
from utils import integer_encoding
from models import LeNet

train_generator = read_train_set()
test_generator, classes = read_test_set()

model = VGG16(weights=None, classes=len(
    train_generator.class_indices), input_shape=(150, 150, 3))


# model = LeNet(classes=len(
    # train_generator.class_indices))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[categorical_accuracy])

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=len(train_generator) + 1,
#     epochs=20
# )

# model.save_weights('lenet.h5')

model.load_weights('vgg16_keras.h5')
print(model.summary())

print()

y_pred = model.predict_generator(
    test_generator, steps=len(classes), verbose=1)
y_pred = np.argmax(y_pred, axis=1)

print(len(integer_encoding(classes)))
print(np.unique(integer_encoding(classes)))
print(len(y_pred))
print(np.unique(y_pred))
print()
print(accuracy_score(integer_encoding(classes), y_pred) * 100)
print(classification_report(integer_encoding(classes), y_pred))
