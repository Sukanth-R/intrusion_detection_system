# src/consumer/feature_extractor.py
import numpy as np

N_FEATURES = 78

# Names for the first few demo features; remaining slots are placeholders.
FEATURE_NAMES = [
    "total_bytes",
    "packet_count",
    "avg_bytes_per_packet",
    "syn_count",
    "unique_destination_ports",
] + [f"f_{i}" for i in range(5, N_FEATURES)]

def extract_window_features(flow_list):
    """
    Minimal demo extractor. Replace with full 78-feature extraction for best accuracy.
    """
    total_bytes = sum(f.get('bytes',0) for f in flow_list)
    pkt_count = len(flow_list)
    avg_pkt = total_bytes / pkt_count if pkt_count else 0
    syn_count = sum(1 for f in flow_list if f.get('flags') and (int(f['flags']) & 0x02))
    unique_dports = len(set(f.get('dport') for f in flow_list))
    arr = [total_bytes, pkt_count, avg_pkt, syn_count, unique_dports]
    arr += [0] * (N_FEATURES - len(arr))
    return np.array(arr, dtype=float)
