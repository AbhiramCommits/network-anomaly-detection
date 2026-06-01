"""
Attempt to download the CICIDS2017 dataset. Falls back to generating a
synthetic dataset with 80+ realistic network flow features if the download
fails or is not available.
"""

import os
import urllib.request
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "cicids_sample.csv")
CICIDS_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"

# ---------------------------------------------------------------------------
# Feature definitions (82 features in total)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Flow identifiers / meta
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    # Flow duration and basic IAT stats
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    # Forward inter-arrival time stats
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    # Backward inter-arrival time stats
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    # Packet length stats (forward)
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    # Packet length stats (backward)
    "Total Backward Packets",
    "Total Length of Bwd Packets",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    # Flow bytes / packet counts
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    # TCP flag counts
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    # Header length
    "Fwd Header Length",
    "Bwd Header Length",
    # Packet counts for small sizes
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    # Subflow stats
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    # Active / idle stats
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    # Bulk stats
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    # Label
    "Label",
]

LABELS = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration", "Web Attack",
          "Brute Force", "Heartbleed"]


def _generate_ip():
    return ".".join(str(np.random.randint(1, 224)) for _ in range(4))


def _generate_mac():
    return ":".join(f"{np.random.randint(0, 256):02x}" for _ in range(6))


def generate_synthetic(n_samples: int = 50000) -> pd.DataFrame:
    """Generate a synthetic dataset that mimics CICIDS2017 network flows."""
    rng = np.random.default_rng(42)
    n = n_samples

    # --- Label distribution (skewed toward BENIGN) ---
    benign_idx = int(n * 0.65)
    attack_idx = n - benign_idx
    labels = np.full(n, "BENIGN", dtype=object)
    attack_types = ["DDoS", "PortScan", "Bot", "Infiltration",
                    "Web Attack", "Brute Force", "Heartbleed"]
    if attack_idx > 0:
        labels[benign_idx:] = rng.choice(attack_types, size=attack_idx, p=[0.3, 0.3, 0.15, 0.05, 0.1, 0.05, 0.05])

    # --- Protocol ---
    protocol = rng.choice(["TCP", "UDP", "ICMP"], size=n, p=[0.85, 0.14, 0.01])

    # --- Ports ---
    common_ports = np.array([80, 443, 22, 21, 25, 53, 8080, 3306, 3389, 5900])
    src_port = rng.integers(1024, 65535, size=n, dtype=np.int64)
    dst_port = rng.choice(common_ports, size=n)

    # --- Flow duration (microseconds, heavily right-skewed) ---
    flow_duration = rng.lognormal(mean=8, sigma=2, size=n) * 1_000  # ~milliseconds

    # --- Packet counts ---
    total_fwd_packets = rng.poisson(lam=6, size=n) + 1
    total_bwd_packets = rng.poisson(lam=4, size=n) + 1

    # --- Packet lengths ---
    fwd_pkt_len_mean = rng.exponential(scale=500, size=n).astype(np.float64) + 40
    fwd_pkt_len_std  = fwd_pkt_len_mean * rng.uniform(0.1, 0.6, size=n)
    fwd_pkt_len_max  = fwd_pkt_len_mean + fwd_pkt_len_std * rng.uniform(1.0, 3.0, size=n)
    fwd_pkt_len_min  = np.maximum(fwd_pkt_len_mean - fwd_pkt_len_std * rng.uniform(0.5, 1.5, size=n), 20)

    bwd_pkt_len_mean = rng.exponential(scale=300, size=n).astype(np.float64) + 40
    bwd_pkt_len_std  = bwd_pkt_len_mean * rng.uniform(0.1, 0.5, size=n)
    bwd_pkt_len_max  = bwd_pkt_len_mean + bwd_pkt_len_std * rng.uniform(1.0, 3.0, size=n)
    bwd_pkt_len_min  = np.maximum(bwd_pkt_len_mean - bwd_pkt_len_std * rng.uniform(0.5, 1.5, size=n), 20)

    total_fwd_len = fwd_pkt_len_mean * total_fwd_packets
    total_bwd_len = bwd_pkt_len_mean * total_bwd_packets

    # --- Flow bytes / packets per second ---
    duration_sec = np.maximum(flow_duration / 1_000_000, 0.001)
    flow_bytes_s  = (total_fwd_len + total_bwd_len) / duration_sec
    flow_pkts_s   = (total_fwd_packets + total_bwd_packets) / duration_sec

    # --- IAT stats (forward / backward / overall) ---
    iat_lam = 10_000 / (total_fwd_packets + total_bwd_packets + 1)
    def _iat_stats(lam):
        iat_mean = lam + rng.normal(0, lam * 0.3, size=n)
        iat_mean = np.maximum(iat_mean, 1)
        iat_std  = iat_mean * rng.uniform(0.2, 0.8, size=n)
        iat_max  = iat_mean + iat_std * rng.uniform(2, 5, size=n)
        iat_min  = np.maximum(iat_mean - iat_std * rng.uniform(0.3, 0.7, size=n), 0)
        return iat_mean, iat_std, iat_max, iat_min

    fwd_iat_total = flow_duration * 0.6
    fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _iat_stats(fwd_iat_total / np.maximum(total_fwd_packets - 1, 1))
    bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _iat_stats(
        flow_duration * 0.4 / np.maximum(total_bwd_packets - 1, 1))
    flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _iat_stats(
        flow_duration / np.maximum(total_fwd_packets + total_bwd_packets - 1, 1))

    # --- TCP flag counts ---
    syn_count = rng.poisson(lam=1.5, size=n).astype(np.int64)
    ack_count = rng.poisson(lam=8, size=n).astype(np.int64)
    fin_count = rng.binomial(n=2, p=0.3, size=n)
    rst_count = rng.binomial(n=2, p=0.05, size=n)
    psh_count = rng.binomial(n=5, p=0.4, size=n)
    urg_count = rng.binomial(n=1, p=0.01, size=n)
    cwe_count = np.zeros(n, dtype=np.int64)
    ece_count = rng.binomial(n=2, p=0.02, size=n)

    fwd_psh = rng.binomial(n=psh_count.astype(int), p=0.5, size=n).astype(np.int64)
    bwd_psh = (psh_count - fwd_psh).astype(np.int64)
    fwd_urg = rng.binomial(n=urg_count.astype(int), p=0.5, size=n).astype(np.int64)
    bwd_urg = (urg_count - fwd_urg).astype(np.int64)

    # --- Header lengths ---
    fwd_header_len = total_fwd_packets * rng.choice([20, 40, 60], size=n, p=[0.1, 0.7, 0.2])
    bwd_header_len = total_bwd_packets * rng.choice([20, 40, 60], size=n, p=[0.1, 0.7, 0.2])

    # --- Packet-level global stats ---
    min_pkt_len = np.minimum(fwd_pkt_len_min, bwd_pkt_len_min)
    max_pkt_len = np.maximum(fwd_pkt_len_max, bwd_pkt_len_max)
    pkt_len_mean = (total_fwd_len + total_bwd_len) / (total_fwd_packets + total_bwd_packets)
    pkt_len_std  = np.sqrt((fwd_pkt_len_std**2 + bwd_pkt_len_std**2) / 2)
    pkt_len_var  = pkt_len_std ** 2

    # --- Subflow stats ---
    subflow_fwd_pkts  = rng.poisson(lam=2, size=n).astype(np.int64) + 1
    subflow_fwd_bytes = subflow_fwd_pkts * fwd_pkt_len_mean * 0.3
    subflow_bwd_pkts  = rng.poisson(lam=2, size=n).astype(np.int64) + 1
    subflow_bwd_bytes = subflow_bwd_pkts * bwd_pkt_len_mean * 0.3

    # --- Init window bytes ---
    init_win_fwd = rng.integers(-1, 65535, size=n, dtype=np.int64)
    init_win_bwd = rng.integers(-1, 65535, size=n, dtype=np.int64)

    # --- Active data packets / min segment size ---
    act_data_pkt_fwd = rng.integers(0, 10, size=n, dtype=np.int64)
    min_seg_size_fwd = rng.integers(20, 1500, size=n, dtype=np.int64)

    # --- Active / idle stats ---
    active_mean = rng.exponential(scale=500, size=n).astype(np.float64)
    active_std  = active_mean * rng.uniform(0.1, 0.5, size=n)
    active_max  = active_mean + active_std * rng.uniform(2, 4, size=n)
    active_min  = np.maximum(active_mean - active_std * 0.5, 0)
    active_cnt  = rng.poisson(lam=3, size=n).astype(np.int64) + 1

    idle_mean = rng.exponential(scale=5000, size=n).astype(np.float64)
    idle_std  = idle_mean * rng.uniform(0.1, 0.5, size=n)
    idle_max  = idle_mean + idle_std * rng.uniform(2, 4, size=n)
    idle_min  = np.maximum(idle_mean - idle_std * 0.5, 0)
    idle_cnt  = rng.poisson(lam=2, size=n).astype(np.int64) + 1

    # --- Bulk stats (aggregated across bulk transfers) ---
    bulk_fwd_pkts  = rng.poisson(lam=3, size=n).astype(np.int64) + 1
    bulk_fwd_bytes = bulk_fwd_pkts * fwd_pkt_len_mean * rng.uniform(0.5, 1.5, size=n)
    bulk_bwd_pkts  = rng.poisson(lam=2, size=n).astype(np.int64) + 1
    bulk_bwd_bytes = bulk_bwd_pkts * bwd_pkt_len_mean * rng.uniform(0.5, 1.5, size=n)

    fwd_bulk_rate = np.where(bulk_fwd_pkts > 0,
                             bulk_fwd_bytes / (bulk_fwd_pkts * (flow_duration / 1_000_000) + 1e-6), 0.0)
    bwd_bulk_rate = np.where(bulk_bwd_pkts > 0,
                             bulk_bwd_bytes / (bulk_bwd_pkts * (flow_duration / 1_000_000) + 1e-6), 0.0)

    # --- Segment size ---
    fwd_seg_size_min = rng.integers(20, 600, size=n, dtype=np.int64)
    bwd_seg_size_min = rng.integers(20, 600, size=n, dtype=np.int64)

    # --- Down/Up ratio ---
    down_up_ratio = np.where(total_fwd_len > 0,
                             total_bwd_len.astype(np.float64) / total_fwd_len.astype(np.float64), 0.0)

    # --- IPs and identifiers ---
    flow_ids = [f"172.16.0.{i}" for i in range(n)]  # simplified
    src_ips = np.array([_generate_ip() for _ in range(n)])
    dst_ips = np.array([_generate_ip() for _ in range(n)])
    timestamps = pd.date_range("2017-07-03 09:00:00", freq="10ms", periods=n)

    # -----------------------------------------------------------------------
    # Assemble DataFrame
    # -----------------------------------------------------------------------
    data = {
        "Flow ID": flow_ids,
        "Src IP": src_ips,
        "Src Port": src_port,
        "Dst IP": dst_ips,
        "Dst Port": dst_port,
        "Protocol": protocol,
        "Timestamp": timestamps,
        # Duration / IAT
        "Flow Duration": flow_duration.astype(np.int64),
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": fwd_iat_total.astype(np.int64),
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": fwd_iat_std,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Min": fwd_iat_min,
        "Bwd IAT Total": (flow_duration * 0.4).astype(np.int64),
        "Bwd IAT Mean": bwd_iat_mean,
        "Bwd IAT Std": bwd_iat_std,
        "Bwd IAT Max": bwd_iat_max,
        "Bwd IAT Min": bwd_iat_min,
        # Fwd packet / length stats
        "Total Fwd Packets": total_fwd_packets,
        "Total Length of Fwd Packets": total_fwd_len.astype(np.int64),
        "Fwd Packet Length Max": fwd_pkt_len_max,
        "Fwd Packet Length Min": fwd_pkt_len_min,
        "Fwd Packet Length Mean": fwd_pkt_len_mean,
        "Fwd Packet Length Std": fwd_pkt_len_std,
        # Bwd packet / length stats
        "Total Backward Packets": total_bwd_packets,
        "Total Length of Bwd Packets": total_bwd_len.astype(np.int64),
        "Bwd Packet Length Max": bwd_pkt_len_max,
        "Bwd Packet Length Min": bwd_pkt_len_min,
        "Bwd Packet Length Mean": bwd_pkt_len_mean,
        "Bwd Packet Length Std": bwd_pkt_len_std,
        # Flow rates
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        # TCP flags
        "Fwd PSH Flags": fwd_psh,
        "Bwd PSH Flags": bwd_psh,
        "Fwd URG Flags": fwd_urg,
        "Bwd URG Flags": bwd_urg,
        "FIN Flag Count": fin_count,
        "SYN Flag Count": syn_count,
        "RST Flag Count": rst_count,
        "PSH Flag Count": psh_count,
        "ACK Flag Count": ack_count,
        "URG Flag Count": urg_count,
        "CWE Flag Count": cwe_count,
        "ECE Flag Count": ece_count,
        # Header length
        "Fwd Header Length": fwd_header_len,
        "Bwd Header Length": bwd_header_len,
        # Per-second rates
        "Fwd Packets/s": total_fwd_packets / duration_sec,
        "Bwd Packets/s": total_bwd_packets / duration_sec,
        # Global packet length stats
        "Min Packet Length": min_pkt_len,
        "Max Packet Length": max_pkt_len,
        "Packet Length Mean": pkt_len_mean,
        "Packet Length Std": pkt_len_std,
        "Packet Length Variance": pkt_len_var,
        # Subflow
        "Subflow Fwd Packets": subflow_fwd_pkts,
        "Subflow Fwd Bytes": subflow_fwd_bytes.astype(np.int64),
        "Subflow Bwd Packets": subflow_bwd_pkts,
        "Subflow Bwd Bytes": subflow_bwd_bytes.astype(np.int64),
        # Init win / act data
        "Init_Win_bytes_forward": init_win_fwd,
        "Init_Win_bytes_backward": init_win_bwd,
        "act_data_pkt_fwd": act_data_pkt_fwd,
        "min_seg_size_forward": min_seg_size_fwd,
        # Active / idle
        "Active Mean": active_mean,
        "Active Std": active_std,
        "Active Max": active_max,
        "Active Min": active_min,
        "Idle Mean": idle_mean,
        "Idle Std": idle_std,
        "Idle Max": idle_max,
        "Idle Min": idle_min,
        # Active / idle counts
        "Active Count": active_cnt,
        "Idle Count": idle_cnt,
        # Bulk stats
        "Fwd Avg Bytes/Bulk": bulk_fwd_bytes / np.maximum(bulk_fwd_pkts, 1),
        "Fwd Avg Packets/Bulk": bulk_fwd_pkts.astype(np.float64),
        "Fwd Avg Bulk Rate": fwd_bulk_rate,
        "Bwd Avg Bytes/Bulk": bulk_bwd_bytes / np.maximum(bulk_bwd_pkts, 1),
        "Bwd Avg Packets/Bulk": bulk_bwd_pkts.astype(np.float64),
        "Bwd Avg Bulk Rate": bwd_bulk_rate,
        # Segment size mins
        "Fwd Seg Size Min": fwd_seg_size_min,
        "Bwd Seg Size Min": bwd_seg_size_min,
        # Ratio
        "Down/Up Ratio": down_up_ratio,
        "Label": labels,
    }

    df = pd.DataFrame(data)
    return df


def main():
    print("Attempting to download CICIDS2017 dataset...")
    downloaded = False

    try:
        # The CICIDS2017 dataset is hosted as a set of CSV files; the
        # canonical URL is https://www.unb.ca/cic/datasets/ids-2017.html
        # Those files are fairly large (~2.5 GB total) and require manual
        # acceptance.  We try one common mirror as a lightweight check.
        print(f"  Fetching {CICIDS_URL} ...")
        try:
            urllib.request.urlopen(CICIDS_URL, timeout=10)
            print("  CICIDS2017 page is reachable, but the dataset requires manual download.")
            print("  Please visit https://www.unb.ca/cic/datasets/ids-2017.html to obtain the CSVs.")
        except Exception:
            print("  Cannot reach CICIDS2017 page. Will generate synthetic data.")
    except Exception:
        pass

    if not downloaded:
        print("\nGenerating synthetic network flow dataset (mimicking CICIDS2017)...")
        df = generate_synthetic(n_samples=50000)
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved {len(df)} rows × {len(df.columns)} columns to {DATA_PATH}")
        print(f"Feature count: {len(df.columns) - 1} features + 1 label column")
        print(f"Label distribution:\n{df['Label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
