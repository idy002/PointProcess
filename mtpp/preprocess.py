import pandas as pd
import numpy as np
import json

#
# the function is modified based on [github.com/Receiling/DeepPointProcess/utils/preprocess.py/generate_sequence]
#

def generate_sequence(csv_file, odir, split_ratio=(0.8, 0.1, 0.1), length=7, min_event_interval=0.01):
    dataset = pd.read_csv(csv_file).sort_values(by=['id', 'time'], ascending=[True, True]).reset_index(drop=True)
    last_id = None
    time_sequence = None
    time_sequences = []
    event_sequence = None
    event_sequences = []
    event_count = {str(i): 0 for i in range(7)}

    def add_sequence():
        if event_sequence is not None and length <= len(event_sequence):
            for ed in range(length, len(event_sequence)):
                if float(time_sequence[ed]) - float(time_sequence[ed - 1]) < min_event_interval:
                    continue
                time_sequences.append(time_sequence[ed - length:ed])
                event_count[event_sequence[ed]] += 1
                event_sequences.append(event_sequence[ed - length:ed])

    for index, row in dataset.iterrows():
        if index * 100 // dataset.shape[0] != (index-1) * 100 // dataset.shape[0]:
            print(f"{index * 100 // dataset.shape[0]}")
        id = str(row['id'])
        timestamp = str(row['time'])
        event = str(row['event'])
        if last_id is None or last_id != id:
            add_sequence()
            last_id = id
            time_sequence = [timestamp]
            event_sequence = [event]
        else:
            time_sequence.append(timestamp)
            event_sequence.append(event)
    add_sequence()

    json.dump(event_count, open(f'{odir}/event_count.json', 'w'))

    n_train_data = int(split_ratio[0] * len(event_sequences))
    n_valid_data = int(split_ratio[1] * len(event_sequences))
    perm = np.random.permutation(len(event_sequences))

    time_sequences = np.array(time_sequences)
    event_sequences = np.array(event_sequences)

    time_sequences[perm] = time_sequences
    event_sequences[perm] = event_sequences


    with open(f'{odir}/train_time.txt', 'w') as train_time_fout:
        train_time_fout.writelines([','.join(time_sequence) + '\n' for time_sequence in time_sequences[:n_train_data]])
    with open(f'{odir}/valid_time.txt', 'w') as valid_time_fout:
        valid_time_fout.writelines([','.join(time_sequence) + '\n' for time_sequence in time_sequences[n_train_data:n_train_data + n_valid_data]])
    with open(f'{odir}/test_time.txt', 'w') as test_time_fout:
        test_time_fout.writelines([','.join(time_sequence) + '\n' for time_sequence in time_sequences[n_train_data + n_valid_data:]])

    with open(f'{odir}/train_event.txt', 'w') as train_event_fout:
        train_event_fout.writelines([','.join(event_sequence) + '\n' for event_sequence in event_sequences[:n_train_data]])
    with open(f'{odir}/valid_event.txt', 'w') as valid_event_fout:
        valid_event_fout.writelines([','.join(event_sequence) + '\n' for event_sequence in event_sequences[n_train_data:n_train_data + n_valid_data]])
    with open(f'{odir}/test_event.txt', 'w') as test_event_fout:
        test_event_fout.writelines([','.join(event_sequence) + '\n' for event_sequence in event_sequences[n_train_data + n_valid_data:]])


if __name__ == '__main__':
    generate_sequence("data/ATM_day.csv", "data")
