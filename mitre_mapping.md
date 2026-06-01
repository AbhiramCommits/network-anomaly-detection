# MITRE ATT&CK — Feature-to-Technique Mapping

Top 15 engineered network-flow features and their corresponding
[MITRE ATT&CK](https://attack.mitre.org/) technique IDs.

| # | Feature | Technique ID | Technique Name | Rationale |
|---|---------|-------------|----------------|-----------|
| 1 | `Dst Port` | [T1046](https://attack.mitre.org/techniques/T1046) | Network Service Scanning | Attackers probe multiple ports to discover open services; anomalous port spread is a strong scanning indicator. |
| 2 | `Flow Duration` | [T1071.001](https://attack.mitre.org/techniques/T1071/001) | Web Protocols | C2 sessions often exhibit very short (beacon) or excessively long (tunnelling) flow durations compared to normal traffic. |
| 3 | `Flow Bytes/s` | [T1048](https://attack.mitre.org/techniques/T1048) | Exfiltration Over Alternative Protocol | Sustained high byte rates on non-standard ports often indicate data exfiltration. |
| 4 | `Total Fwd Packets` | [T1048](https://attack.mitre.org/techniques/T1048) | Exfiltration Over Alternative Protocol | Large forward packet counts can reflect data staging or exfiltration bursts. |
| 5 | `Total Length of Fwd Packets` | [T1041](https://attack.mitre.org/techniques/T1041) | Exfiltration Over C2 Channel | High forward byte volume is characteristic of data being pushed out of the network. |
| 6 | `Fwd Packet Length Mean` | [T1030](https://attack.mitre.org/techniques/T1030) | Data Transfer Size Limits | Attackers may use fixed, small, or unusually large payload sizes to blend in or maximise throughput. |
| 7 | `Bwd Packet Length Mean` | [T1071](https://attack.mitre.org/techniques/T1071) | Application Layer Protocol | C2 server responses often have consistent small sizes (commands) vs. normal application traffic. |
| 8 | `Fwd IAT Mean` | [T1071.004](https://attack.mitre.org/techniques/T1071/004) | DNS | Regular inter-arrival times (low IAT variance) are a hallmark of beaconing — the attacker's callback heartbeat. |
| 9 | `Bwd IAT Mean` | [T1071](https://attack.mitre.org/techniques/T1071) | Application Layer Protocol | Backward IAT patterns distinguish server-driven responses from interactive user traffic. |
| 10 | `SYN Flag Count` | [T1046](https://attack.mitre.org/techniques/T1046) | Network Service Scanning | An abnormally high SYN count per flow (or across flows) signals TCP SYN scanning. |
| 11 | `PSH Flag Count` | [T1048](https://attack.mitre.org/techniques/T1048) | Exfiltration Over Alternative Protocol | Push flags indicate data being forced through the connection quickly — common during exfiltration. |
| 12 | `Flow Packets/s` | [T1498](https://attack.mitre.org/techniques/T1498) | Network Denial of Service | Extremely high packet rates are the defining characteristic of volumetric DDoS and DoS attacks. |
| 13 | `Active Mean` | [T1071.001](https://attack.mitre.org/techniques/T1071/001) | Web Protocols | Short, periodic active windows suggest automated C2 polling rather than human browsing behaviour. |
| 14 | `Idle Mean` | [T1071](https://attack.mitre.org/techniques/T1071) | Application Layer Protocol | The silence between C2 beacons is a critical signal; regular idle intervals map to beacon sleep times. |
| 15 | `Init_Win_bytes_forward` | [T1071](https://attack.mitre.org/techniques/T1071) | Application Layer Protocol | Unusual TCP window sizes can indicate custom C2 tooling that differs from standard OS defaults. |
