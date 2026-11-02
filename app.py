from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

DATA_FILE = Path(__file__).parent / "Dataset" / "student-mat.csv"


@st.cache_resource
def train_model():
    # same steps as the notebook: load, encode text columns, split 80/20, train
    df = pd.read_csv(DATA_FILE, sep=",")
    df_model = pd.get_dummies(df, drop_first=True)

    X = df_model.drop("G3", axis=1)
    y = df_model["G3"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    # the "typical student" row used to fill the columns we don't ask about
    typical = X_train.astype(float).median()
    importance = pd.Series(model.feature_importances_, index=X.columns)
    return model, X.columns, typical, mae, r2, importance


def weekly_range(daily_hours):
    # the dataset stores study time as weekly ranges (1 to 4)
    weekly = daily_hours * 7
    if weekly < 2:
        return 1
    if weekly <= 5:
        return 2
    if weekly <= 10:
        return 3
    return 4


model, columns, typical, mae, r2, importance = train_model()

st.title("🎓 Student Performance Predictor")
st.write("Answer 6 short questions to get the expected final marks (out of 20).")

st.subheader("Academic record")
c1, c2 = st.columns(2)
g1 = c1.slider("Marks in the first term exam", 0, 20, 10)
g2 = c2.slider("Marks in the second term exam", 0, 20, 10)

st.subheader("Study habits")
c3, c4 = st.columns(2)
daily_hours = c3.slider("Study hours per day", 0.0, 12.0, 2.0, step=0.5)
absences = c4.slider("Classes missed this year", 0, 93, 4)

st.subheader("Motivation and lifestyle")
c5, c6 = st.columns(2)
higher = c5.radio("Plans to pursue higher education?", ["Yes", "No"], horizontal=True)
goout = c6.slider("Going out with friends (1 = rarely, 5 = very often)", 1, 5, 3)

if st.button("Predict final marks", type="primary"):
    row = typical.copy()
    row["G1"] = g1
    row["G2"] = g2
    row["studytime"] = weekly_range(daily_hours)
    row["absences"] = absences
    row["higher_yes"] = 1 if higher == "Yes" else 0
    row["goout"] = goout
    row = row[columns].to_frame().T

    pred = float(min(max(model.predict(row)[0], 0), 20))
    low = max(pred - mae, 0)
    high = min(pred + mae, 20)

    st.success(f"Predicted final marks: {pred:.1f} out of 20")
    st.write(f"Likely range: {low:.1f} to {high:.1f} (the model is usually off by about {mae:.1f} marks)")

with st.sidebar:
    st.header("About the model")
    st.write("Random Forest trained on 316 students and tested on 79 unseen students.")
    st.metric("Average error (MAE)", f"{mae:.2f} marks")
    st.metric("R² score", f"{r2:.2f}")
    st.caption("Data: Kaggle 'Math Students' dataset (395 students)")

with st.expander("Which factors matter most?"):
    st.bar_chart(importance.sort_values(ascending=False).head(10))