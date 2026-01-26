import pandas as pd 
import numpy as np
import os, sys
import re
from tqdm import tqdm

log_pattern = re.compile(
    r'(?P<ip>\S+)\s+'
    r'(?P<identd>\S+)\s+'
    r'(?P<user>\S+)\s+'
    r'\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+'
    r'(?P<status>\d{3})\s+'
    r'(?P<size>\S+)'
)

class Processor():

    def __init__(self):
        pass
        
    def extract_info(self, line, return_df = False):
        '''
        Extract information from a log line.

        Input: 
        line (str): log line from request
        return_df (bool): return DataFrame if True

        Output:

        d (dict): dictionary of extracted information
        or
        d (pd.DataFrame): DataFrame of extracted information
        '''
        match = log_pattern.match(line)
        if not match:
            return None

        d = match.groupdict()

        if d["request"]:
            parts = d["request"].split()
            d["resource"] = parts[1] if len(parts) > 1 else None
            d["protocol"] = parts[2] if len(parts) > 2 else None
        else:
            d["resource"] = d["protocol"] = None

        d["size"] = None if d["size"] == "-" else int(d["size"])
        d["status"] = int(d["status"])

        d['utc'] = f'UTC - {d['time'].split('-')[1][:2]}'

        d["time"] = pd.to_datetime(
            d["time"].split('-')[0].strip(), format="%d/%b/%Y:%H:%M:%S"
        )

        if return_df:
            return pd.DataFrame([d])
            
        return d

    def batch_streaming_ingestion(self, original_path, batch_size = 50000, output_path = 'output.csv'):

        fieldnames = list(self.extract_info(lines[0]).keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total = len(lines)

        for start in tqdm(range(0, total, BATCH_SIZE), desc="Batches"):
            end = min(start + BATCH_SIZE, total)
            batch = lines[start:end]

            buffer = []
            bad = 0

            for line in batch:
                try:
                    d = self.extract_info(line)
                    if d is not None:      
                        buffer.append(d)
                    else:
                        bad += 1
                except Exception:
                    bad += 1

            if buffer:
                writer.writerows(buffer)
                f.flush()
                os.fsync(f.fileno())

            print(f"Batch {start//batch_size + 1}: "
                f"written {len(buffer)}, bad {bad}")

