import joblib
import numpy as np

# load the model
model = joblib.load("exoplanetPrediction_model.joblib")

#star to predict, edit values to recieve prediction!
sample = np.array([[  
    4800,    # star_temp
    1.0,     # star_radius
    1.0,     # star_mass
    1.0,     # star_metallicity
    1.0      # star_surfaceGravity
]])

prediction = model.predict(sample)
prediction = max(0, prediction[0])  # avoid negatives

print("Predicted planets:", round(prediction))

