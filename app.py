from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import numpy as np
import ast
import pyarrow as pa
import pyarrow.parquet as pq
import os


def ollama_qa(context,keys):
    template='''
    Answer the question below.
    Here is the conversation history: {context}
    Question: {question}
    Answer:
    '''
    model = OllamaLLM(model='llama3.2')
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    question='''
    Return only the answer.
    Create a dictionary without nested elements.
    The dictionary should contain key-value pairs from the keys I give you.
    Here are the keys for dictionary :     
    '''+keys
    result = chain.invoke({"context": context,"question": question})
    
    return result

def print_data_info(df):
    # number of rows and columns
    rows = len(df.axes[0])
    cols = len(df.axes[1])
    print("Number of Rows : ", rows)
    print("Number of Columns : ", cols)
    print("Columns :",list(df.columns))
    print(df['manufacturer'].value_counts())


def csv_to_parquet(filename):
    # create a parquet file and load it instead of csv
    save_name = filename.split('.')[0]
    parquet_file = save_name+".parquet"

    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)

    else:
        # extract datetime as date format
        df=pd.read_csv(filename,parse_dates=['posting_date'])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_file)
        df = pd.read_parquet(parquet_file)

    # print('---Original Data Information---')
    # print_data_info(df)

    return df

def manage_DF(filename):   
    # df=csv_to_parquet(filename)
    # df['posting_date'] = pd.to_datetime(df['posting_date']).dt.date
    df=pd.read_csv(filename,parse_dates=['posting_date'])


    # df['posting_date'] = pd.to_datetime(df['posting_date']).dt.year
    # drop columns
    df=df.drop(columns=['county','state','lat','long'])

    # df=df[['id','price','year','manufacturer','model',]]
    # filters
    # df = df[df['posting_date'] > 2019]
    df = df[df['year'] > 1999]
    df = df[df['odometer'] < 150000]
    df = df[(df['condition'] == 'excellent') | (df['condition'] == 'good')]
    df = df[(df['transmission'] == 'manual') | (df['transmission'] == 'automatic')]

    manufacturer_list=['subaru','bmw','toyota','ford','lexus','gmc']
    df = df[df['manufacturer'].isin(manufacturer_list)]
    
    # drop rows where 'description' is missing
    df=df.dropna(subset=['description'])
    df = df.replace({np.nan: None})

    # print('---Filtered Data Information---')
    # print_data_info(df)

    return df


def print_dict(dict):
    for key, value in dict.items():
        print(f"{key} : {value}")


def description_to_dict(description,keys):
    res=ollama_qa(description,keys)
    res=ast.literal_eval(res)

    keys=res.keys()
    values=res.values() 
    # dictionary comprehension to remove None values
    res = { k:v for (k,v) in zip(keys, values) if res[k]}  

    return res
   

def run_all():
    filename='vehicles_100.csv'
    df_main=manage_DF(filename)

    df_main=pd.read_csv(filename)
    df_main=df_main.head(2)


    # get columns of DF, remove description and convert to a string
    columns=df_main.columns.tolist()
    columns.remove("description")
    keys_str = ", ".join(columns)

   


    # change all columns data type to string
    # for col in columns:
    #     df_main[col]=df_main[col].astype('string')

    # iterate through all rows and create dict from description for each row
    for index, row in df_main.iterrows():
        res=df_main.iloc[index]

        # text description of each car 
        description=res['description']

        # remove 'description' from df_dict Pandas Series
        df_dict=res.drop(labels=['description'])
        # turn df_dict from Series -> Dictionary 
        df_dict=df_dict.to_dict()

        # description of each car by columns
        car_desc=description_to_dict(description,keys_str)

        keys=list(car_desc.keys())

        # # print(car_desc)

        for key in keys:
            df_val=df_main.at[index, key]
            desc_val=car_desc[key]
            # compare values or update with description
            # df_main.at[index, key] = df_val +', Desc : ' + str(desc_val)
            df_main.at[index, key] = desc_val


    # print(df_main)
    # csv_name='result_csv'+filename
    csv_name='result_csv.csv'
    df_main.to_csv(csv_name,index=False,encoding="utf-8")


def main():
    run_all()


if __name__=="__main__":
    main()
