from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import numpy as np
import ast


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


def manage_DF(filename):   
    df=pd.read_csv(filename)    
    # drop columns
    df.drop(columns=['county','state','lat','long','posting_date'])
    # drop rows where 'description' is missing
    df=df.dropna(subset=['description'])
    df = df.replace({np.nan: None})

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
    filename='vehicles_small.csv'
    df_main=manage_DF(filename)

    # get columns of DF, remove description and convert to a string
    columns=df_main.columns.tolist()
    columns.remove("description")
    keys_str = ", ".join(columns)

    # change all columns data type to string
    for col in columns:
        df_main[col]=df_main[col].astype('string')

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

        # print(car_desc)

        for key in keys:
            df_val=df_main.at[index, key]
            desc_val=car_desc[key]
            df_main.at[index, key] = df_val +', Desc : ' + str(desc_val)

    # print(df_main)
    csv_name=filename
    df_main.to_csv(csv_name,index=False,encoding="utf-8")


def main():
    run_all()


if __name__=="__main__":
    main()
