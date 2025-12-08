import os


DATA_LIST = ["data/test_data_8.csv"] # 101.57 seconds
DATA_LIST = ["data/test_data.csv"] # 3.56 seconds
DATA_LIST = ["data/test_data_9_300.csv"] # 3.56 seconds



MAX_TOP_THREADS = 20
MAX_SUB_THREADS = 5
MAX_PROCESS_THREADS = os.cpu_count()