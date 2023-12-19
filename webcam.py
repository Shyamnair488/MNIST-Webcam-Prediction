import cv2
import numpy as np
import tensorflow as tf

model_path = "C:/Users/shyam/Desktop/Projects/MNIST/mnist_simple_model.h5"
model = tf.keras.models.load_model(model_path)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 28, 28))
    return reshaped

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture from the webcam.")
            break

        processed_frame = preprocess_image(frame)
        prediction = model.predict(processed_frame)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        cv2.putText(frame, f"Prediction: {predicted_class} - Confidence: {confidence:.2f}%", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Number Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
