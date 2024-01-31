import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sql_encoder.encoder import SQLEncoder
from sql_encoder.optimizer.canonicalizer import rewrite_query
from config.config import workload
import json


def encode(sql):
    sql = rewrite_query(sql, json.load(open(f"./schema/{workload}.json", "r")))
    sql_encoder = SQLEncoder()
    if not sql_encoder.load_encoder():
        print("Encoder checkpoint file not found, can't encode the sql...")
        return
    sql_embedding = sql_encoder.encode([sql])[0].tolist()
    return sql_embedding

