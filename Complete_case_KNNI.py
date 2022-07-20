#Author Name : Rakesh Swaminathan
#Email_id : rakesh.swaminaathan@outlook.com
#Data Created : 07/19/2022
#Content: Python program for missing value prediction/imputation using K - Nearest Neighbor

#Import the required library packages
import numpy as np
import pandas as pd
import math
import os
import glob
import time
from sklearn.impute import KNNImputer

#Start

FILE_PATH = 'F:\Canada\Meng\Sem 4\Data Mining\Project\KNN'
IMP_FILE = FILE_PATH + "\Impute_Files.txt"
ORG_FILE = FILE_PATH + "\Original_Files.txt"

#Read the given NRMS file to write the results
result_file = pd.read_excel(FILE_PATH + '\Table-NRMS.xlsx')
result_file_array = result_file.to_numpy()  #Converting to array for assigning the results for the corresponding datasets

impute_files = open(IMP_FILE,"r")
for IMP in impute_files:
    IMP_DATA = IMP.rstrip('\n')
    ORG_DATA = IMP_DATA.replace(" ","_")

    incomplete_datasets_path = "{0}\\Dataset\\Incomplete datasets\\{1}".format(FILE_PATH,IMP_DATA)
    files_to_be_imputed = glob.glob(os.path.join(incomplete_datasets_path, "*csv"))
    for file in files_to_be_imputed:
        start = time.time()
        original_dataset = pd.read_csv("{0}\\Dataset\\Complete datasets\\{1}.csv".format(FILE_PATH,ORG_DATA),skiprows=1,header=None)
        df = pd.read_csv(file,header=None)
        df1 = pd.DataFrame(df.dropna(axis=0,how='any')) #Complete Data
        df2 = pd.DataFrame(df[df.isnull().any(axis=1)]) #Incomplete Data
        #impute missing values
        for i in range(len(df2)):
            df1 = pd.concat([df2.head(1),df1],axis=0)
            k = int(math.sqrt(len(df1)))
            impute = KNNImputer(n_neighbors=k)
            aft_imputation = impute.fit_transform(df1.sort_index(axis=0))
            df2=df2.iloc[1:,:]
        #Compute NRMS
        nrms_num = np.linalg.norm(aft_imputation - original_dataset)
        nrms_den = np.linalg.norm(original_dataset)
        nrms = nrms_num/nrms_den
        end = time.time()
        run_time = end - start
        imputed_file_name = os.path.split(file)[1].split(".")[0]
        temp = np.where(result_file_array==imputed_file_name)
        row = temp[0][0]
        col = temp[1][0]
        result_file_array[row][col + 1] = nrms
        result_file_array[row][col + 2] = run_time
        result_file_array[row][col + 3] = k
        print(file + " - completed")
        
    pd.DataFrame(result_file_array,columns=['File Name','NRMS Value','Run Time','K parameter']).to_excel(FILE_PATH + '\Table-NRMS-CompleteCase.xlsx',index=False)
#End
