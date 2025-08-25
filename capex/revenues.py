import numpy as np

def sample(dist_obj):
    if dist_obj["dist"] == "Normale":
        return np.random.normal(dist_obj["p1"], dist_obj["p2"])
    elif dist_obj["dist"] == "Triangolare":
        return np.random.triangular(dist_obj["p1"], dist_obj["p2"], dist_obj.get("p3", dist_obj["p1"] + dist_obj["p2"]))
    elif dist_obj["dist"] == "Lognormale":
        return np.random.lognormal(dist_obj["p1"], dist_obj["p2"])
    elif dist_obj["dist"] == "Uniforme":
        return np.random.uniform(dist_obj["p1"], dist_obj["p2"])
    else:
        return dist_obj["p1"]
