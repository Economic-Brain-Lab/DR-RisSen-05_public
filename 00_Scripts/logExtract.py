#%%
import pandas as pd
#%%

def process_log_file(file_path):
    trials = []
    trig_numbers = []
    blckN = []
    counter = 1

    with open(logfile, 'r') as file:
        for line in file:
            if "Where: RespSub_Trial:" in line:
                parts = line.split("Where: RespSub_Trial:")[1]
                trial_number = parts.split('/')[0].strip()
                block_number = parts.split('/')[1].split(':')[1].strip()
                trials.append(int(trial_number))
                blckN.append(int(block_number))
                trig_numbers.append(counter)
                counter += 1
            
    data = {'trial': trials, "block": blckN,'respTrigN': trig_numbers}
    df = pd.DataFrame(data)

    #Sort the values so that the align with the behavioural data
    df2 = df.sort_values(by=['block', 'trial']).reset_index(drop=True)
    return df2

# Example usage:
file_path = 'path_to_your_log_file.log'
df = process_log_file(file_path)
print(df)


#%% Testing

logfile = 'F:\\DR-RisSen-05\\LOG\\sub-15_MAIN.log'
trials = []
trig_numbers = []
blckN = []
counter = 1

with open(logfile, 'r') as file:
    for line in file:
        if "Where: RespSub_Trial:" in line:
            parts = line.split("Where: RespSub_Trial:")[1]
            trial_number = parts.split('/')[0].strip()
            block_number = parts.split('/')[1].split(':')[1].strip()
            trials.append(int(trial_number))
            blckN.append(int(block_number))
            trig_numbers.append(counter)
            counter += 1
        
data = {'trial': trials, "block": blckN,'respTrigN': trig_numbers}
df = pd.DataFrame(data)

# Sorting by block and trial number
df2 = df.sort_values(by=['block', 'trial']).reset_index(drop=True)
# %%
