import matplotlib.pyplot as plt

import pandas as pd
# use forward slashes to avoid escape-sequence warnings on Windows
df = pd.read_csv('data/TrainDistancesDelft_NSPrices_NSTravelTime.csv')
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
  'Distance (km)': 'distance_km',
  'NS 2nd class fare (€) (no subscription, e-ticket)': 'ns_fare_eur',
  'travel time (min)': 'travel_time_min'
})
# ensure City header is normalized to lowercase 'city'
df = df.rename(columns={'City': 'city'})
df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
df['ns_fare_eur'] = pd.to_numeric(df['ns_fare_eur'], errors='coerce')
df['travel_time_min'] = pd.to_numeric(df['travel_time_min'], errors='coerce')


# Print city plus the three train columns (distance, fare, travel time)
print("\nTrain columns (city, distance_km, ns_fare_eur, travel_time_min):")
wanted = ['city', 'distance_km', 'ns_fare_eur', 'travel_time_min']
missing = [c for c in wanted if c not in df.columns]
if missing:
    print(f"Missing columns, can't print requested fields: {missing}")
    print("Available columns:", list(df.columns))
else:
    print(df[wanted].head(50).to_string(index=False))


def train_cost_per_city(df: 'pd.DataFrame') -> 'pd.Series':
  """Return train cost (NS 2nd class fare) per city as a Series indexed by city.

  The function expects the DataFrame to contain a 'city' column and 'ns_fare_eur'.
  """
  if 'city' not in df.columns or 'ns_fare_eur' not in df.columns:
    raise KeyError("DataFrame must contain 'city' and 'ns_fare_eur' columns")
  # Return a copy to avoid accidental modifications
  return df.set_index('city')['ns_fare_eur'].copy()


def Train_Travel_Time_per_city(df: 'pd.DataFrame') -> 'pd.Series':
  """Return total travel time per city (train travel time + 10 minutes bike ride).

  Expects 'city' and 'travel_time_min' columns. Returns minutes as numeric.
  """
  if 'city' not in df.columns or 'travel_time_min' not in df.columns:
    raise KeyError("DataFrame must contain 'city' and 'travel_time_min' columns")
  # return raw train travel time (minutes) without additional bike time
  total = pd.to_numeric(df['travel_time_min'], errors='coerce')
  return pd.Series(total.values, index=df['city'].values, name='train_travel_time_min')


# Print outputs of the two functions for inspection
print("\nTrain cost per city (EUR):")
try:
  costs = train_cost_per_city(df)
  print(costs.to_string())
except Exception as e:
  print("Could not compute train cost per city:", e)

print("\nTrain travel time per city (minutes, raw train time):")
try:
  times = Train_Travel_Time_per_city(df)
  print(times.to_string())
except Exception as e:
  print("Could not compute train travel time per city:", e)

