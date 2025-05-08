# ###############! Add ECW to clean_train_data.csv
# # This script adds ECW values to the clean_train_data.csv file based on the ESW_data_1.csv file.

# import pandas as pd

# # Read the ESW_data_1.csv file
# esw_data = pd.read_csv('ESW_data_3.csv')

# # Read the clean_train_data_ECW.csv file
# train_data = pd.read_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full.csv')
# print(train_data.head())

# esw_data['smiles'] = '[Cu]' + esw_data['smiles'] + '[Au]'
# # Skip the first 29 rows of data in esw_data while keeping the header
# esw_data = esw_data.iloc[29:].reset_index(drop=True)
# print(esw_data.head())

# # Merge the data based on the 'smiles' column without changing the rows of train_data
# merged_data = train_data.merge(esw_data[['smiles', 'ECW']], right_on='smiles' ,how='left', validate='many_to_one')

# # Fill missing Bandgap chain values with 5.0
# merged_data['ECW'] = merged_data['ECW'].fillna(5.0)

# # Save the updated data back to a CSV file
# merged_data.to_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full_4.csv', index=False)


# import numpy as np

# # Add a random ECW value between 4 and 7 for each 'smiles'
# esw_data['ECW'] = np.random.uniform(4, 8, size=len(esw_data))

# # Save the updated data back to a CSV file
# esw_data.to_csv('ESW_data_3_updated.csv', index=False)





# ############################################
# #! Add ECW to s_full.csv
# import pandas as pd
# import numpy as np

# # df = pd.read_csv('/home/naibing/work/chemarr_ECW/data/cross_val_data/s_full.csv')
# new_df = pd.read_csv('/home/naibing/work/chemarr_ECW/data/clean_train_data_ECW.csv')
# # df = pd.read_csv('/home/naibing/work/chemarr_ECW/data/cross_val_data/s_screen.csv')
# # df = df[["smiles"]]
# # new_df = new_df[["ECW", "temperature"]]
# # new_df = pd.concat([df, new_df], axis=1)
# # # new_df = df.append(new_df, ignore_index=True)
# # print(new_df.head())

# # new_df.to_csv('/home/naibing/work/chemarr_ECW/data/cross_val_data/s_full.csv', index=False)



# new_df = new_df[["smiles","conductivity", "ECW", "temperature"]]
# new_df.to_csv('/home/naibing/work/chemarr_ECW/data/cross_val_data/s_full.csv', index=False)



############################################
# #! This code generates a new s_full_3 file with 10115 lines. 
# import pandas as pd
# import numpy as np

# df = pd.read_csv('/home/naibing/work/chemarr_ECW/data/cross_val_data/s_full_2.csv')
# print(f'df.columns is {df.columns}')
# print(f'df.shape is {df.shape}')

# # Randomly pick 10114 rows (including the header)
# sampled_df = df.sample(n=10114, random_state=42)

# # Save the sampled data to a new CSV file
# sampled_df.to_csv('/home/naibing/work/chemarr_ECW/data/cross_val_data/s_full_3.csv', index=False)


# ##############################
# ##! this code is to replace the 0 in mw to 3000
# import pandas as pd
# df = pd.read_csv('/home/naibing/work/chemarr_fork/Chem-prop-pred/data/cross_val_data/f_screen.csv')
# # df["mw"].replace(0, 3000, inplace=True)
# df["mw"].fillna(3000, inplace=True)
# df["molality"].fillna(0, inplace=True)
# df.to_csv('/home/naibing/work/chemarr_fork/Chem-prop-pred/data/cross_val_data/f_screen_2.csv', index=False)
# print(df.head())

##################################
# #! this code is to clean the data in s_full_3.csv
# import pandas as pd

# def clean_data(input_path, output_path):
#     df = pd.read_csv(input_path)
    
#     # Replace error strings with NaN
#     df.replace('#VALUE!', float(0), inplace=True)
#     # Drop the 'conductivity' column
#     df.drop(columns=['conductivity'], inplace=True)
    
#     # Save the cleaned data to the output path
#     df.to_csv(output_path, index=False)

# clean_data('data/cross_val_data/s_full_3.csv', 'data/cross_val_data/s_full_3_clean.csv')




###############################################################
# #! This code is to add ECW to the s_screen.csv file.
# import pandas as pd

# # Read the data files
# esw_data = pd.read_csv('ESW_data_3.csv')
# train_data = pd.read_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full.csv')

# # Preprocess SMILES in ESW data (if needed)
# esw_data['smiles'] = '[Cu]' + esw_data['smiles'] + '[Au]'
# esw_data = esw_data.iloc[29:].reset_index(drop=True)  # Skip first 29 rows

# # Perform a left join to preserve all original training data rows
# merged_data = pd.merge(
#     left=train_data,
#     right=esw_data[['smiles', 'ECW']],  # Only bring in ECW column
#     on='smiles',
#     how='left',  # Keep all original rows
#     validate='many_to_one'  # Ensure one-to-one mapping
# )

# # Fill missing ECW values (for SMILES not found in ESW data)
# merged_data['ECW'] = merged_data['ECW'].fillna(5.0)  # Default value

# # Verify no original columns were lost
# assert all(col in merged_data.columns for col in train_data.columns)

# # Save the result
# merged_data.to_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full_4.csv', index=False)
# print("Merge completed successfully. Original columns preserved:")
# print(merged_data.head())

#################################################################

# #! modify the s_full_4.csv file to a easier to read format.
# import pandas as pd

# df = pd.read_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full_4.csv')
# df = df[["smiles", "ECW", "temperature"]]
# df['ECW'].rename('conductivity', inplace=True)

# df.to_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full_4_2.csv', index=False)


######################################################### 
# # #!
# import pandas as pd
# df = pd.read_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full_4_2.csv')
# # print(df["conductivity"].values)
# # if pd.api.types.is_float_dtype(df["conductivity"]):
# #     print("The 'conductivity' column is of type float.")
# # else:
# #     print("The 'conductivity' column is not of type float.")
# # Convert the 'conductivity' column to float
# df["conductivity"] = df["conductivity"].astype(float)
# print("Converted 'conductivity' column to float.")
# print(df["conductivity"].values)

######################################################
# #! to change the feature file & smiles file
# import pandas as pd
# s_df = pd.read_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full_4_2.csv')
# s_df = s_df[["smiles", "conductivity"]]
# s_df.to_csv('/home/naibing/work/chemarr_ECW_new/data/cross_val_data/s_full_4_3.csv', index=False)

###################
#! to make a identical prediction on training set (different from the s_screen, only without salts)
import pandas as pd
df = pd.read_csv("/home/naibing/work/chemarr_fork/Chem-prop-pred/data/cross_val_data/s_full_4_2.csv")
df = df[["smiles","temperature"]]
df.to_csv("/home/naibing/work/chemarr_fork/Chem-prop-pred/data/cross_val_data/s_full_4_4.csv", index=False)