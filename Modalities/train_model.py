'''from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the embeddings and labels
embeddings_rgb = np.load('embeddings_rgb.npy')
embeddings_ir = np.load('embeddings_ir.npy')
labels = np.load('labels.npy')

# Concatenate the RGB and IR embeddings
embeddings = np.concatenate((embeddings_rgb, embeddings_ir), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Train a classifier
clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
'''
'''import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_embeddings_and_labels(embedding_path_rgb, embedding_path_ir):
    embeddings_rgb = []
    embeddings_ir = []
    labels = []

    for label in os.listdir(embedding_path_rgb):
        rgb_label_path = os.path.join(embedding_path_rgb, label)
        ir_label_path = os.path.join(embedding_path_ir, label)

        if not os.path.exists(ir_label_path):
            continue

        for file in os.listdir(rgb_label_path):
            if file.endswith('.npy'):
                # Load RGB embedding
                #rgb_embedding_path = os.path.join(rgb_label_path, file)
                rgb_embedding = np.load(embeddings_rgb.npy)

                # Load corresponding IR embedding
                # ir_embedding_path = os.path.join(ir_label_path, file)
                # if not os.path.exists(ir_embedding_path):
                #     continue
                ir_embedding = np.load(embeddings_ir.npy)

                embeddings_rgb.append(rgb_embedding)
                embeddings_ir.append(ir_embedding)
                labels.append(label)

    return np.array(embeddings_rgb), np.array(embeddings_ir), np.array(labels)

# Example usage
embedding_path_rgb = 'embeddings_rgb.npy'
embedding_path_ir = 'embeddings_ir.npy'

embeddings_rgb, embeddings_ir, labels = load_embeddings_and_labels(embedding_path_rgb, embedding_path_ir)

# Combine embeddings
combined_embeddings = np.concatenate((embeddings_rgb, embeddings_ir), axis=1)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(combined_embeddings, encoded_labels, test_size=0.2, random_state=42)

# Train SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model and label encoder
model_path = 'face_fusion_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump((classifier, label_encoder), f)

print("Face fusion model saved to", model_path)'''

'''import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load embeddings and labels
embeddings_rgb = np.load('embeddings_rgb.npy')
embeddings_ir = np.load('embeddings_ir.npy')
labels = np.load('labels.npy')

# Concatenate RGB and IR embeddings
fused_embeddings = np.concatenate((embeddings_rgb, embeddings_ir), axis=1)

# Encode labels // Using Label encoder
# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels)

# Using Train test Split
X_train, X_test, Y_train, Y_test = train_test_split(fused_embeddings, labels, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='linear', probability=True)
#svm.fit(embeddings, labels_encoded) //Label encoder
svm.fit(X_train, Y_train) 

# Save the trained SVM model
joblib.dump(svm, 'face_fusion_model.pkl')

# Save the label encoder
# joblib.dump(label_encoder, 'label_encoder.pkl')

# Evaluate the model (optional)
predictions = svm.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
'''

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load embeddings and labels
embeddings_rgb = np.load('embeddings_rgb.npy')
embeddings_ir = np.load('embeddings_ir.npy')
labels = np.load('labels.npy')

# Combine embeddings
embeddings = np.concatenate((embeddings_rgb, embeddings_ir), axis=1)

# Train a more advanced classifier (e.g., SVM)
classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', probability=True))
])

classifier.fit(embeddings, labels)

# Save the trained model
import joblib
joblib.dump(classifier, 'face_fusion_model.pkl')
