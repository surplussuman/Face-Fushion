import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load embeddings and labels
embeddings_rgb = np.load('embeddings/embeddings_rgb.npy')
embeddings_ir = np.load('embeddings/embeddings_ir.npy')
labels = np.load('embeddings/labels.npy')

# Concatenate RGB and IR embeddings
embeddings = np.concatenate((embeddings_rgb, embeddings_ir), axis=1)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, random_state=42)

# Train an SVM model
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model and label encoder
joblib.dump(clf, 'models/face_fusion_model.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
