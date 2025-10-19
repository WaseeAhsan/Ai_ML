# প্রয়োজনীয় লাইব্রেরি ইম্পোর্ট করা হলো 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib 

# 1. ডেটা লোড করা 
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model/iris_model.joblib')

print("Model trained and saved successfully.")