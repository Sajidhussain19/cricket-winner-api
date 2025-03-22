from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load team mappings
with open("team_mapping.pkl", "rb") as f:
    team_mapping = pickle.load(f)

inverse_team_mapping = {v: k for k, v in team_mapping['team1'].items()}

app = FastAPI()

class MatchInput(BaseModel):
    team1: str
    team2: str
    toss_winner: str
    toss_decision: str

@app.post("/predict")
def predict_winner(input_data: MatchInput):
    try:
        team1_enc = team_mapping['team1'][input_data.team1]
        team2_enc = team_mapping['team2'][input_data.team2]
        toss_winner_enc = team_mapping['toss_winner'][input_data.toss_winner]
        toss_decision_enc = 1 if input_data.toss_decision == "bat" else 0

        features = np.array([[team1_enc, team2_enc, toss_winner_enc, toss_decision_enc]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        predicted_winner = inverse_team_mapping[prediction[0]]

        return {"predicted_winner": predicted_winner}
    except KeyError as e:
        return {"error": f"Invalid input: {e}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Cricket Match Winner Prediction API is running!"}
