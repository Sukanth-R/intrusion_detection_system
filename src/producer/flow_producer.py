# src/producer/flow_producer.py
import argparse, time, json, random
from kafka import KafkaProducer
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

KAFKA = os.getenv("KAFKA_BOOTSTRAP", "localhost:29092")
TOPIC = "flows"

producer = KafkaProducer(bootstrap_servers=KAFKA, value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def produce_synthetic(rate=200, mode="benign"):
    print("Starting synthetic producer", "mode:", mode, "rate:", rate)
    while True:
        for _ in range(rate):
            src = f"10.0.0.{random.randint(2,200)}"
            dst = f"10.0.0.{random.randint(2,10)}"

            if mode == "attack":
                dport = random.choice([23, 445, 3389, 80, 8080])
                flags = random.choices([2, 18, 16, 0], weights=[0.65, 0.15, 0.15, 0.05], k=1)[0]
                byte_count = random.randint(40, 400)
            elif mode == "scan":
                dport = random.randint(1, 65535)
                flags = random.choices([2, 18, 16, 0], weights=[0.9, 0.02, 0.06, 0.02], k=1)[0]
                byte_count = random.randint(40, 120)
            else:
                dport = random.choice([80, 443, 22, 8080])
                flags = random.choices([16, 18, 24, 2, 0], weights=[0.45, 0.25, 0.15, 0.1, 0.05], k=1)[0]
                byte_count = random.randint(60, 1500)

            rec = {
                "ts": time.time(), "src": src, "dst": dst,
                "sport": random.randint(1024,65535), "dport": dport,
                "proto": 6, "flags": flags, "bytes": byte_count,
                "mode": mode,
            }
            producer.send(TOPIC, rec)
        time.sleep(1)


def _try_import_scapy():
    try:
        from scapy.all import sniff, get_if_list, IFACES, conf  # type: ignore
        from scapy.layers.inet import IP, TCP, UDP  # type: ignore
        try:
            from scapy.layers.inet6 import IPv6  # type: ignore
        except Exception:
            IPv6 = None
        return sniff, get_if_list, IFACES, conf, IP, TCP, UDP, IPv6
    except Exception as e:
        raise RuntimeError(
            "Live capture requires scapy and a packet capture driver.\n\n"
            "Install scapy into your active venv:\n"
            "  python -m pip install scapy\n\n"
            "On Windows you also typically need Npcap installed (WinPcap-compatible mode helps).\n"
            "Then rerun with --mode live.\n\n"
            f"Import error: {e}"
        )


def produce_live(interface: str | None = None, bpf_filter: str = "arp or tcp or udp"):
    sniff, get_if_list, IFACES, conf, IP, TCP, UDP, IPv6 = _try_import_scapy()

    # Best-effort promiscuous capture (still limited by Wi-Fi/switch behavior).
    # Set LIVE_PROMISC=0 to disable.
    try:
        conf.sniff_promisc = os.getenv("LIVE_PROMISC", "1").strip() not in ("0", "false", "no", "off")
    except Exception:
        pass

    def _iface_label(network_name: str, iface_obj) -> str:
        name = str(getattr(iface_obj, "name", "") or "")
        desc = str(getattr(iface_obj, "description", "") or "")
        ip = str(getattr(iface_obj, "ip", "") or "")
        return f"{network_name} | {name} | {desc} | ip={ip}"

    def resolve_interface(user_value: str) -> str:
        available = set(get_if_list())
        if user_value in available:
            return user_value

        raw = (user_value or "").strip()
        if not raw:
            raise ValueError("Empty interface")

        # Accept passing just the GUID portion like {A-B-C}.
        if raw.startswith("{") and raw.endswith("}"):
            candidate = f"\\Device\\NPF_{raw}"
            if candidate in available:
                return candidate

        # Accept passing an IP address to choose the adapter that owns it.
        # Common when users think '--interface' means 'Wi-Fi IP'.
        def _looks_like_ipv4(s: str) -> bool:
            parts = s.split(".")
            if len(parts) != 4:
                return False
            try:
                return all(0 <= int(p) <= 255 for p in parts)
            except Exception:
                return False

        if _looks_like_ipv4(raw):
            matches: list[str] = []
            for _, iface in IFACES.items():
                network_name = getattr(iface, "network_name", None)
                if not network_name or network_name not in available:
                    continue
                ips = getattr(iface, "ips", None)
                if ips and isinstance(ips, dict) and raw in (ips.get(4) or []):
                    matches.append(network_name)
                elif str(getattr(iface, "ip", "") or "") == raw:
                    matches.append(network_name)
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise RuntimeError(
                    "The provided IP matched multiple adapters. Please pass the exact \\Device\\NPF_{...} value.\n"
                    + "\n".join(matches)
                )
            raise RuntimeError(
                "No capture adapter matched that IP. Re-run with --list-interfaces and choose the Wi-Fi adapter's \\Device\\NPF_{...} value."
            )

        # Accept friendly names/descriptions by substring match.
        needle = raw.lower()

        matches: list[str] = []
        match_labels: list[str] = []
        for _, iface in IFACES.items():
            # On Windows, network_name is typically the \Device\NPF_{GUID} string.
            network_name = getattr(iface, "network_name", None)
            if not network_name or network_name not in available:
                continue
            name = str(getattr(iface, "name", "") or "")
            desc = str(getattr(iface, "description", "") or "")
            if needle in name.lower() or needle in desc.lower():
                matches.append(network_name)
                match_labels.append(_iface_label(network_name, iface))

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "Ambiguous --interface value. It matched multiple adapters.\n"
                "Pass the exact \\Device\\NPF_{...} string for the one you want. Matches:\n"
                + "\n".join(match_labels)
            )
        raise RuntimeError(
            "Interface not found. Re-run with --list-interfaces and pass one of the listed adapters."
        )

    if interface is None:
        raise RuntimeError("--interface is required for --mode live. Use --list-interfaces to see options.")

    interface = resolve_interface(interface)

    # Allow disabling BPF filter for troubleshooting.
    bpf = bpf_filter
    if bpf_filter is None:
        bpf = None
    else:
        bpf_text = str(bpf_filter).strip().lower()
        if bpf_text in ("none", "no", "off", "disabled"):
            bpf = None

    print("Starting live producer", "interface:", interface, "filter:", (bpf if bpf is not None else "<none>"))
    stats_every = float(os.getenv("LIVE_STATS_SECS", "5"))
    last_stats_ts = time.time()
    pkt_count = 0
    sent_count = 0
    flush_every = int(os.getenv("LIVE_FLUSH_EVERY", "200"))

    def on_pkt(pkt):
        nonlocal last_stats_ts, pkt_count, sent_count
        try:
            is_ipv4 = IP in pkt
            is_ipv6 = (IPv6 is not None) and (IPv6 in pkt)
            if not is_ipv4 and not is_ipv6:
                return
            pkt_count += 1
            ip = pkt[IP] if is_ipv4 else pkt[IPv6]
            # IPv4 uses .proto, IPv6 uses .nh (next header)
            proto = int(getattr(ip, "proto", getattr(ip, "nh", 0)) or 0)
            ts = time.time()

            if TCP in pkt:
                l4 = pkt[TCP]
                sport = int(getattr(l4, "sport", 0) or 0)
                dport = int(getattr(l4, "dport", 0) or 0)
                flags = int(getattr(l4, "flags", 0) or 0)
            elif UDP in pkt:
                l4 = pkt[UDP]
                sport = int(getattr(l4, "sport", 0) or 0)
                dport = int(getattr(l4, "dport", 0) or 0)
                flags = 0
            else:
                return

            byte_count = int(len(pkt))
            rec = {
                "ts": ts,
                "src": str(getattr(ip, "src", "")),
                "dst": str(getattr(ip, "dst", "")),
                "sport": sport,
                "dport": dport,
                "proto": proto,
                "flags": flags,
                "bytes": byte_count,
                "mode": "live",
            }
            producer.send(TOPIC, rec)
            sent_count += 1

            if flush_every > 0 and (sent_count % flush_every) == 0:
                producer.flush(timeout=2)

            now = time.time()
            if stats_every > 0 and (now - last_stats_ts) >= stats_every:
                print(f"LIVE stats: captured={pkt_count} sent={sent_count} last_src={rec.get('src')} last_dport={rec.get('dport')}")
                last_stats_ts = now
        except Exception:
            # Best-effort: don't crash the sniffer loop.
            return

    sniff(iface=interface, filter=bpf, prn=on_pkt, store=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benign", "attack", "scan", "live"], default="benign")
    parser.add_argument("--rate", type=int, default=300, help="Synthetic only")
    parser.add_argument("--interface", default=None, help="Live only (example: 'Wi-Fi')")
    parser.add_argument("--bpf", default="tcp or udp", help="Live only (BPF filter)")
    parser.add_argument("--list-interfaces", action="store_true", help="Print capture interfaces and exit")
    args = parser.parse_args()
    if args.list_interfaces:
        _, get_if_list, IFACES, *_ = _try_import_scapy()
        avail = set(get_if_list())
        for _, iface in IFACES.items():
            network_name = getattr(iface, "network_name", None)
            if not network_name or network_name not in avail:
                continue
            name = str(getattr(iface, "name", "") or "")
            desc = str(getattr(iface, "description", "") or "")
            print(f"{network_name} | {name} | {desc}")
        raise SystemExit(0)
    if args.mode == "live":
        produce_live(interface=args.interface, bpf_filter=args.bpf)
    else:
        produce_synthetic(rate=args.rate, mode=args.mode)
