import joblib
import numpy as np

# load the model
model = joblib.load("exoplanetPrediction_model.joblib")

#star to predict
sample = np.array([[  
    34153,    # star_temp
    1.3155,     # star_radius
    2.2923,     # star_mass
    3.46,     # star_metallicity
    4.91      # star_surfaceGravity
]])

prediction = model.predict(sample)
prediction = max(0, prediction[0])  # avoid negatives

print("Predicted planets:", round(prediction))
