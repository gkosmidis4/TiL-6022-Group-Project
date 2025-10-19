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


# --- simplified plotting -------------------------------------------------
import os

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)
# Set to True if you want the script to show plots interactively after saving.
# In some environments (headless servers, or certain IDE terminals) showing
# may not work; set to False to only save files.
SHOW_PLOTS = False


def simple_scatter(x, y, labels, xlabel, ylabel, title, fname):
  """Create a simple scatter plot with city name labels and grid ticks every 10."""
  plt.figure(figsize=(7, 5))
  plt.scatter(x, y, c='C0')
  # annotate points with city names (simple offset)
  for xi, yi, lab in zip(x, y, labels):
    plt.text(xi + 0.5, yi + 0.5, lab, fontsize=8)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  # set simple grid with ticks every 10
  xmin, xmax = plt.xlim()
  ymin, ymax = plt.ylim()
  xticks = list(range(int(max(0, xmin)) - (int(max(0, xmin)) % 10), int(xmax) + 11, 10))
  yticks = list(range(int(max(0, ymin)) - (int(max(0, ymin)) % 10), int(ymax) + 11, 10))
  plt.xticks(xticks)
  plt.yticks(yticks)
  plt.grid(True)
  path = os.path.join(FIG_DIR, fname)
  plt.tight_layout()
  plt.savefig(path)
  print(f"Saved: {path}")
  if SHOW_PLOTS:
    try:
      plt.show()
    except Exception:
      # If interactive display fails, continue without raising
      pass
  plt.close()


# Prepare simple data for plotting (drop rows with missing values)
plot_df = df[['city', 'distance_km', 'ns_fare_eur', 'travel_time_min']].dropna()
cities = plot_df['city'].astype(str).values
distance = plot_df['distance_km'].values
cost = plot_df['ns_fare_eur'].values
time = plot_df['travel_time_min'].values

# Print functions (very simple) so output appears in the terminal
'''
def print_cost_vs_distance(df):
  print('\nCost vs Distance (city, distance_km, ns_fare_eur):')
  for city, dist, c in df[['city', 'distance_km', 'ns_fare_eur']].itertuples(index=False):
    print(f"{city:20} {dist:6.0f} km   {c:5.2f} EUR")
'''
'''
def print_cost_vs_time(df):
  print('\nCost vs Time (city, travel_time_min, ns_fare_eur):')
  for city, t, c in df[['city', 'travel_time_min', 'ns_fare_eur']].itertuples(index=False):
    print(f"{city:20} {t:6.0f} min  {c:5.2f} EUR")
'''
'''
def print_distance_vs_time(df):
  print('\nDistance vs Time (city, travel_time_min, distance_km):')
  for city, t, dist in df[['city', 'travel_time_min', 'distance_km']].itertuples(index=False):
    print(f"{city:20} {t:6.0f} min  {dist:6.0f} km")
'''


# 1) Cost vs Distance (cost on y-axis, distance on x-axis)
simple_scatter(distance, cost, cities, 'Distance (km)', 'Train cost (EUR)',
               'Cost vs Distance (per city)', 'cost_vs_distance.png')

# 2) Cost vs Time (cost on y-axis, time on x-axis)
simple_scatter(time, cost, cities, 'Train travel time (min)', 'Train cost (EUR)',
               'Cost vs Time (per city)', 'cost_vs_time.png')

# 3) Distance vs Time (distance on y-axis, time on x-axis)
simple_scatter(time, distance, cities, 'Train travel time (min)', 'Distance (km)',
               'Distance vs Time (per city)', 'distance_vs_time.png')

# Create bar charts with cities on x-axis
def simple_bar(values, city_names, ylabel, title, fname):
    """Create a simple bar chart with cities on x-axis."""
    plt.figure(figsize=(10, 5))
    plt.bar(city_names, values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y')
    path = os.path.join(FIG_DIR, fname)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved: {path}")
    if SHOW_PLOTS:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


print("\nGenerating bar charts...")

# 1. Time bar chart
simple_bar(time, cities, 'Travel time (minutes)',
          'Train Travel Time by City', 'time_bars.png')

# 2. Distance bar chart
simple_bar(distance, cities, 'Distance (km)',
          'Distance by City', 'distance_bars.png')

# 3. Cost bar chart
simple_bar(cost, cities, 'Cost (EUR)',
          'Train Cost by City', 'cost_bars.png')



