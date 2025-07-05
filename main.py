# ==========================
# Imports
# ==========================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta, timezone
from tabulate import tabulate
from skyfield.api import load
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import joblib


# ==========================
# Constants and Initialization
# ==========================
planets = load('de421.bsp')
ts = load.timescale()

planet_objects = {
    'mercury': planets['mercury'],
    'venus': planets['venus'],
    'earth': planets['earth'],
    'mars': planets['mars'],
    # Add others if needed
}

look_back = 60  # Number of days to use as history for LSTM


# ==========================
# Helper Functions
# ==========================

def days_until(date_str):
    """Returns number of days from today until a target date"""
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    return (target_date - today).days


def prepare_training_data(planet, years=10):
    """Generates and scales training data for LSTM based on planetary ephemeris"""
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365 * years)

    dates = [start_date + timedelta(days=i) for i in range(365 * years)]
    times = ts.utc([d.year for d in dates], [d.month for d in dates], [d.day for d in dates])
    positions = planet.at(times).xyz.au.T  # Shape: (n_days, 3)

    # Create sequences
    X, y = [], []
    for i in range(look_back, len(positions)):
        X.append(positions[i - look_back:i])
        y.append(positions[i])

    X, y = np.array(X), np.array(y)

    # Normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X.reshape(-1, 3)).reshape(X.shape)
    y_scaled = scaler.transform(y)

    return X_scaled, y_scaled, scaler


def create_lstm_model(input_shape):
    """Returns compiled LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def get_planet_model(planet_name, planet_obj):
    """Loads or trains an LSTM model for a planet"""
    try:
        model = joblib.load(f'{planet_name}_lstm_model.pkl')
        scaler = joblib.load(f'{planet_name}_scaler.pkl')
        print(f"Loaded pre-trained model for {planet_name}")
    except:
        print(f"Training new LSTM model for {planet_name}...")
        X, y, scaler = prepare_training_data(planet_obj)
        model = create_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        joblib.dump(model, f'{planet_name}_lstm_model.pkl')
        joblib.dump(scaler, f'{planet_name}_scaler.pkl')
    return model, scaler


def predict_positions(planet_name, days_to_predict):
    """Predicts future positions for a planet"""
    model, scaler = planet_models[planet_name]
    planet_obj = planet_objects[planet_name]

    # Get recent historical data
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=look_back)
    dates = [start_date + timedelta(days=i) for i in range(look_back)]
    times = ts.utc([d.year for d in dates], [d.month for d in dates], [d.day for d in dates])
    positions = planet_obj.at(times).xyz.au.T

    # Scale and prepare sequence
    sequence = scaler.transform(positions)[np.newaxis, ...]

    predicted = []
    for _ in range(days_to_predict):
        next_scaled = model.predict(sequence, verbose=0)
        predicted.append(next_scaled[0])
        sequence = np.roll(sequence, -1, axis=1)
        sequence[0, -1] = next_scaled[0]

    return scaler.inverse_transform(np.array(predicted)).T


def collect_planet_info(name, x, y):
    """Computes revolution and position info"""
    start = (x[0], y[0])
    end = (x[-1], y[-1])
    angles = np.unwrap(np.arctan2(y, x))
    revolutions = (angles[-1] - angles[0]) / (2 * np.pi)
    planet_data.append([
        name.capitalize(),
        f"({start[0]:.3f}, {start[1]:.3f})",
        f"({end[0]:.3f}, {end[1]:.3f})",
        f"{revolutions:.3f}"
    ])


# ==========================
# User Input & Model Setup
# ==========================
input_date = input("Enter a date (YYYY-MM-DD): ").strip()
days_to_simulate = days_until(input_date)

planet_models = {name: get_planet_model(name, obj) for name, obj in planet_objects.items()}


# ==========================
# Predictions
# ==========================
planet_predictions = {
    name: predict_positions(name, days_to_simulate)
    for name in planet_objects
}


# ==========================
# Extract Coordinates
# ==========================
me_x, me_y, _ = planet_predictions['mercury']
v_x, v_y, _ = planet_predictions['venus']
e_x, e_y, _ = planet_predictions['earth']
m_x, m_y, _ = planet_predictions['mars']


# ==========================
# Plotting Setup
# ==========================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_title('LSTM-Based Animation of Planets')
ax.grid(True)

# Plot orbits
ax.plot(me_x, me_y, color="#708090", alpha=0.5, linewidth=0.5)
ax.plot(v_x, v_y, color="#ff7f00", alpha=0.5, linewidth=0.5)
ax.plot(e_x, e_y, color="#1f77b4", alpha=0.5, linewidth=0.5)
ax.plot(m_x, m_y, color="#b22222", alpha=0.5, linewidth=0.5)

# Markers
for x, y in [(me_x, me_y), (v_x, v_y), (e_x, e_y), (m_x, m_y)]:
    ax.plot(x[0], y[0], marker='*', color='black', markersize=5)
    ax.plot(x[-1], y[-1], marker='X', color='black', markersize=5)

# Planet dots
sun_dot, = ax.plot(0, 0, color="#FFD700", marker='o', markersize=20, label='Sun')
mercury_dot, = ax.plot([], [], color="#708090", marker='o', markersize=4, label='Mercury')
venus_dot, = ax.plot([], [], color="#ff7f00", marker='o', markersize=6, label='Venus')
earth_dot, = ax.plot([], [], color="#1f77b4", marker='o', markersize=6, label='Earth')
mars_dot, = ax.plot([], [], color="#b22222", marker='o', markersize=5, label='Mars')

ax.legend()


# ==========================
# Animation
# ==========================
def update(frame):
    mercury_dot.set_data([me_x[frame]], [me_y[frame]])
    venus_dot.set_data([v_x[frame]], [v_y[frame]])
    earth_dot.set_data([e_x[frame]], [e_y[frame]])
    mars_dot.set_data([m_x[frame]], [m_y[frame]])
    return mercury_dot, venus_dot, earth_dot, mars_dot


ani = FuncAnimation(fig, update, frames=days_to_simulate, interval=100, blit=True)
plt.show()


# ==========================
# Summary Table
# ==========================
planet_data = []
collect_planet_info("mercury", me_x, me_y)
collect_planet_info("venus", v_x, v_y)
collect_planet_info("earth", e_x, e_y)
collect_planet_info("mars", m_x, m_y)

headers = ["Planet", "Start (x, y)", "End (x, y)", "Revolutions"]
print(tabulate(planet_data, headers=headers, tablefmt="fancy_grid"))
