import json
import matplotlib.pyplot as plt

# Reading histore json
with open('training_history_relu.json', 'r') as f:
    history = json.load(f)

# Visualization of data
epochs = [i for i in range(1, len(history['loss']) + 1)]
plt.plot(epochs, history['accuracy'], color='blue')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.title('Visualization of Training Accuracy Result')
plt.show()

plt.plot(epochs, history['loss'], color='red')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Visualization of Training Loss Result')
plt.show()

plt.plot(epochs, history['val_accuracy'], color='blue')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Visualization of Validation Accuracy Result')
plt.show()

plt.plot(epochs, history['val_loss'], color='red')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('Visualization of Validation Loss Result')
plt.show()
