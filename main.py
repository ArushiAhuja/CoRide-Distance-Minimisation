import numpy as np
import pandas as pd
from itertools import combinations


ProximRadius = 3.0     # I have set a proximity radius to ensure ALL pickups must be within 3 km to be eligible for carpool

"""
Cosine Similarity is cosine of the angle between two vectors (in this case our displacement vectors).
If cosine similarity is 1, the vectors point in the same direction. (pooling is most beneficial)
If cosine similarity is 0, the vectors are orthogonal (90 degrees apart). (pooling is not the best approach)
If cosine similarity is -1, the vectors point in opposite directions. (pooling is not beneficial)
"""

CosSim = 0.6     # cosine similarity must exceed this

# I have converted the entire dataset into a numpy array for easy manipulation

data = [
    ("CP001", (0,0), (4,3)),
    ("CP002", (1,1), (3,3)),
    ("CP003", (0,0), (3,1)),
    ("CP004", (0,0), (5,2)),
    ("CP005", (2,0), (3,3)),
    ("CP006", (1,1), (3,5)),
    ("CP007", (0,0), (3,2)),
    ("CP008", (2,1), (6,-1)),
    ("CP009", (1,2), (2,-1)),
    ("CP010", (0,0), (2,-1)),
]

# I converted the array into a pandas dataframe where indexing can be through the TripID instead of default 0,1,2..)
df = pd.DataFrame(data, columns=["TripID", "Pickup", "Dropoff"]).set_index("TripID")

# I have defined some functions to compute distances between any two points (pickup/dropoff)

def distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def displacement(pickup, dropoff):
    return np.array(dropoff) - np.array(pickup)

def trip_distance(pickup, dropoff):
    return distance(pickup, dropoff)

def cosine_similarity(u, v, eps=1e-12):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + eps))

# added the displacement and distance columns to the dataframe
# we have written axis=1 to apply the function row-wise (horizontally)
df["Displacement"] = df.apply(lambda r: displacement(r["Pickup"], r["Dropoff"]), axis=1)
df["Distance"]     = df.apply(lambda r: trip_distance(r["Pickup"], r["Dropoff"]), axis=1)

# to decide on the best way to save distance, I have created a dataframe of all possible pairs of trips
# I have used combinations from itertools to avoid duplicate pairs and self-pairing

pairs = []
for t1, t2 in combinations(df.index, 2):
    d1, d2 = df.loc[t1, "Displacement"], df.loc[t2, "Displacement"]
    cos_sim = cosine_similarity(d1, d2)
    pickup_prox = distance(df.loc[t1, "Pickup"], df.loc[t2, "Pickup"])
    pairs.append((t1, t2, cos_sim, pickup_prox))

pairDF = pd.DataFrame(pairs, columns=["Trip1", "Trip2", "DirectionSimilarity", "PickupProximity"])

# creating another dataframe to filter out only those pairs that meet our criteria
candidate_pairs = pairDF[
    (pairDF["DirectionSimilarity"] > CosSim) &
    (pairDF["PickupProximity"]   <= ProximRadius)
].copy()

# I have created a function to compute the pooled distance for two trips
# it tries all valid pickup/dropoff sequences and returns the minimal distance and the best order
def pooled_distance(p1, d1, p2, d2):
    points = {"A_P": p1, "A_D": d1, "B_P": p2, "B_D": d2}
    best_dist = np.inf
    best_order = None

    valid_orders = [
        ["A_P","A_D","B_P","B_D"],
        ["A_P","B_P","A_D","B_D"],
        ["A_P","B_P","B_D","A_D"],
        ["B_P","B_D","A_P","A_D"],
        ["B_P","A_P","B_D","A_D"],
        ["B_P","A_P","A_D","B_D"],
    ]

    for order in valid_orders:
        total = 0
        for k in range(len(order) - 1):
            total += distance(points[order[k]], points[order[k + 1]])

        if total < best_dist:
            best_dist = total
            best_order = order

    return best_dist, best_order

# Compute separate vs pooled distances only for feasible pairs
results = []
for _, row in candidate_pairs.iterrows():
    a, b = row["Trip1"], row["Trip2"]
    r1, r2 = df.loc[a], df.loc[b]

    separate = r1["Distance"] + r2["Distance"]
    pooled, order = pooled_distance(r1["Pickup"], r1["Dropoff"], r2["Pickup"], r2["Dropoff"])
    savings = separate - pooled

    results.append((
        a, b,
        separate, pooled, savings, order,
        row["DirectionSimilarity"], row["PickupProximity"]
    ))

results_df = pd.DataFrame(
    results,
    columns=[
        "Trip1","Trip2","SeparateDist","PooledDist","Savings","BestOrder",
        "DirectionSimilarity","PickupProximity"
    ]
)

# dataframe containing only those pairs that actually save distance
best_pools = results_df[results_df["Savings"] > 0].sort_values("Savings", ascending=False)

# traditional approach without cosine similarity and proximity radius
RandomResults = []
for t1, t2 in combinations(df.index, 2):
    r1, r2 = df.loc[t1], df.loc[t2]
    separate = r1["Distance"] + r2["Distance"]
    pooled, order = pooled_distance(r1["Pickup"], r1["Dropoff"], r2["Pickup"], r2["Dropoff"])
    savings = separate - pooled
    RandomResults.append((t1, t2, separate, pooled, savings, order))

random_df = pd.DataFrame(
    RandomResults,
    columns=["Trip1","Trip2","SeparateDist","PooledDist","Savings","BestOrder"]
)

AvgRandomSavings = random_df["Savings"].mean()
AvgFilteredSavings = results_df["Savings"].mean() if not results_df.empty else 0

#.2f = round off upto 2 decimals
#.1f = round off upto 1 decimals

print("Average pooling savings (random pairing): {:.2f} km".format(AvgRandomSavings))
print("Average pooling savings (cosine+proximity filtered): {:.2f} km".format(AvgFilteredSavings))

if AvgRandomSavings > 0:
    improvement_pct = ((AvgFilteredSavings - AvgRandomSavings) / AvgRandomSavings) * 100
    print("Our approach improves pooling efficiency by {:.1f}% over random pairing".format(improvement_pct))
else:
    print("Random pooling produces no savings; our approach is {:.2f} km better on average".format(AvgFilteredSavings))
