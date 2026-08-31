from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load XGBoost / ML model
model = joblib.load("forecasting_model.pkl")

# Load training data for feature computation
full_df = pd.read_csv("train.csv", parse_dates=["date"])

# -----------------------------
# Feature Engineering Function
# -----------------------------
def build_features(store, item, date):

    target_date = pd.to_datetime(date)

    # Filter only chosen store & item
    df = full_df[(full_df["store"] == store) & (full_df["item"] == item)].copy()
    df = df.sort_values("date")

    # Last known real date
    last_known_date = df["date"].max()

    # Loop until target date
    while last_known_date < target_date:

        next_date = last_known_date + pd.Timedelta(days=1)

        # Add new row
        new_row = {
            "date": next_date,
            "store": store,
            "item": item,
            "sales": None
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Sort again
        df = df.sort_values("date")

        # -------------------------
        # Generate features
        # -------------------------
        df["year"] = df["date"].dt.year.astype("int64")
        df["month"] = df["date"].dt.month.astype("int64")
        df["day"] = df["date"].dt.day.astype("int64")
        df["week"] = df["date"].dt.isocalendar().week.astype("int64")
        df["dayofweek"] = df["date"].dt.dayofweek.astype("int64")

        df["lag_1"] = df["sales"].shift(1)
        df["lag_7"] = df["sales"].shift(7)
        df["lag_30"] = df["sales"].shift(30)

        df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()
        df["rolling_30"] = df["sales"].shift(1).rolling(30).mean()

        # Extract the row we need to predict now
        feature_row = df.iloc[-1][[
            "store", "item",
            "year", "month", "day",
            "week", "dayofweek",
            "lag_1", "lag_7", "lag_30",
            "rolling_7", "rolling_30"
        ]]

        # Convert to DataFrame and enforce numeric dtype
        feature_df = pd.DataFrame([feature_row])
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
        feature_df = feature_df.fillna(0)

        # -------------------------
        # Predict today's sales
        # -------------------------
        prediction = model.predict(feature_df)[0]

        # Insert prediction into the dataframe
        df.at[df.index[-1], "sales"] = prediction

        # Move to next day
        last_known_date = next_date

    # Return features for final target date
    final_row = df[df["date"] == target_date].iloc[-1]
    final_features = final_row[[
        "store", "item",
        "year", "month", "day",
        "week", "dayofweek",
        "lag_1", "lag_7", "lag_30",
        "rolling_7", "rolling_30"
    ]]

    # Convert cleanly to DataFrame
    final_features = pd.DataFrame([final_features])
    final_features = final_features.apply(pd.to_numeric, errors="coerce").fillna(0)

    return final_features

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Predict Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        store = int(data["store_id"])
        item = int(data["item_id"])
        date = data["date"]

        features = build_features(store, item, date)
        prediction = model.predict(features)[0]

        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# Run Flask (Python 3.13 FIX)
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)