import pandas as pd 
import numpy as np
import os, sys
import re
from tqdm import tqdm
import csv

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

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

    def batch_streaming_ingestion(self, lines, batch_size = 50000, output_path = 'output.csv'):
        '''
        Batch processing a lot of logs into structured dataframe and save into csv

        Input:
        lines: list of log lines
        batch_size (int): number of lines to process at once
        output_path (str): path to save the output csv file

        Output
        '''

        fieldnames = list(self.extract_info(lines[0]).keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            total = len(lines)

            for start in tqdm(range(0, total, batch_size), desc="Batches"):
                end = min(start + batch_size, total)
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

        return pd.read_csv(output_path)

    def extract_additional_info(self, df):
        '''
        Extract Additional information from the dataframe

        Input:
        df: pandas dataframe
        '''

        df['time'] = pd.to_datetime(df['time'])

        # hits/sec
        hits_per_sec = df.set_index("time").resample("1s").size()

        # error rate (also per sec)
        df["is_error"] = df["status"] >= 400 # if Not Found is also counted as Error

        error_rate = (
            df.set_index("time")
              .resample("1s")
              .apply(lambda x: x["is_error"].mean())
        )

        # spikes
        window = 60  # 1 minute window

        rolling_mean = hits_per_sec.rolling(window).mean()
        rolling_std = hits_per_sec.rolling(window).std()

        zscore = (hits_per_sec - rolling_mean) / rolling_std

        spikes = zscore > 3

        metrics = pd.DataFrame({
            "hits_per_sec": hits_per_sec,
            "error_rate": error_rate,
            "is_spike": spikes
        })

        metrics = metrics.fillna(0)
        metrics = metrics.reset_index()

        return metrics

# if __name__ == '__main__':

#     processor = Processor()

#     with open(os.path.join(project_root, 'data', 'original', 'train.txt'), 'r', encoding='utf-8') as f:
#         train = f.readlines()
    
#     print(f'Read {len(train)} lines from train.txt')

#     with open(os.path.join(project_root, 'data', 'original', 'test.txt'), 'r', encoding='utf-8') as f:
#         test = f.readlines()
    
#     print(f'Read {len(test)} lines from test.txt')
    
#     print('Processing train')
#     processor.batch_streaming_ingestion(train, output_path = os.path.join(project_root, 'data', 'processed', 'train.csv'))

#     print('Processing test')
#     processor.batch_streaming_ingestion(test, output_path = os.path.join(project_root, 'data', 'processed', 'test.csv'))

#     train_df = pd.read_csv(os.path.join(project_root, 'data', 'processed', 'train.csv'))
#     test_df = pd.read_csv(os.path.join(project_root, 'data', 'processed', 'test.csv'))

#     print('Extracting additional info from train')
#     train_metrics = processor.extract_additional_info(train_df)
#     print('Extracting additional info from test')
#     test_metrics = processor.extract_additional_info(test_df)

#     metrics = pd.concat([train_metrics, test_metrics], axis = 0)

#     metrics.to_csv(os.path.join(project_root, 'data', 'processed', 'for_viz.csv'), index=False)

#     print('Done!')

