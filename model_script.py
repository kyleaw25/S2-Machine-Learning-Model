#artificial intelligence s2
#
#machine learning model creation script
#
#creating with ai assistance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# load your dataset
df = pd.read_csv("stellar_dataset.csv")

# set features and target
X = df[[
    "star_temp",
    "star_radius",
    "star_mass",
    "star_metallicity",
    "star_surfaceGravity"
]]

y = df["planet_amount"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model (random forest works well for this)
model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(n_estimators=300, random_state=42)
)

# train model
model.fit(X_train, y_train)

# evaluate model
preds = model.predict(X_test)
preds = np.clip(preds, 0, None)  # make sure predictions aren’t negative

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

#printing test results when model is created
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# save the model
joblib.dump(model, "exoplanetPrediction_model.joblib")
print("Model saved as exoplanetPrediction_model.joblib")
