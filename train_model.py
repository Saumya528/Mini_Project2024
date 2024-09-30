from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load extracted features and labels
with open("features_labels.pkl", 'rb') as f:
    X, y = pickle.load(f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
model = SVC()
model.fit(X_train, y_train)

# Test model accuracy
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the trained model to disk
with open('D://Downloads/SAUMYA MITRA/IIT Jodhpur/Mini_Project2024/SpeechRecognitionSample/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")