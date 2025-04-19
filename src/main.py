import os
from groq import Groq
import pandas as pd
from dotenv import load_dotenv
from enum import Enum
from Agents import *
from settings import *
from tqdm import tqdm

# Load environment variables (Groq API key)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AgentType(Enum):
    PROSECUTOR = {'role': "prosecution_lawyer", 'focus': "evidence of guilt, witness statements against the defendant, and legal arguments for conviction"}
    DEFENDANT = {'role': "defendant", 'focus': "personal account, explanations, mitigating circumstances, and character evidence"}
    PLAINTIFF = {'role': "plaintiff", 'focus': "details of the incident, harm suffered, and evidence implicating the defendant"}
    DEFENSE = {'role': "defense_lawyer", 'focus': "evidence of innocence, alibi, character witnesses, and legal arguments for acquittal"}

# Preprocess CSV to Summarize Case Descriptions
def summarize_case(description, summary_type , max_words=300):
    """Summarize a case description to ~300 words."""    

    # Get the focus for the specified role
    role = summary_type['role']
    focus = summary_type['focus']
    # Create a role-specific prompt
    prompt = f"""Provide only the concise summary of the following case description in 300 words, 
    written from the perspective of the {role}, focusing on {focus}. 
    The summary should be designed for the {role} to carry with them to the trial for quick reference, 
    and no introductory or explanatory text should be included: {description}"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.5,
        max_completion_tokens=int(max_words * 1.33) # ~300 words
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    dataset_path = "resources/cases.csv"
    log_file_path = "logs.txt"
    df = pd.read_csv(dataset_path)

    case_IDs  = []
    vers = []

    

    for case_no in tqdm(range(14,50)):

        fullCase = df.loc[case_no,'text']
        case_ID = df.loc[case_no,'id']

        #print(case_ID)
        #print(fullCase)

        print("Generating Summaries")

        case_descriptions = {"defendant": summarize_case(fullCase,AgentType.DEFENDANT.value),
                             "plaintiff": summarize_case(fullCase,AgentType.PLAINTIFF.value),
                             "defense_lawyer": summarize_case(fullCase,AgentType.DEFENSE.value),
                             "prosecution_lawyer": summarize_case(fullCase,AgentType.PROSECUTOR.value)}

        print("Summaries Generated\n")

        personalities = {}

        personalities["prosecution_lawyer"] = prosecution_personality
        personalities["defence_lawyer"] = defence_personality
        personalities["judge"] = judge_personality


        #for r,m in case_descriptions.items():
            #print()
            #print(f'Summary of --{r.capitalize()}--')
            #print(m)

        prosecution_laywer = Lawyer("Prosecution Lawyer",case_descriptions["prosecution_lawyer"],personalities["prosecution_lawyer"])
        defence_laywer = Lawyer("Defence Lawyer",case_descriptions["defense_lawyer"],personalities["defence_lawyer"])
        defendant = Defendant("Prosecution Lawyer",case_descriptions["defendant"])
        plantiff = Plaintiff("Prosecution Lawyer",case_descriptions["plaintiff"])
        judge = Judge("Judge",personality=personalities["judge"])

        court = CourtroomSimulation(case_descriptions,log_file_path,
                                    plantiff,prosecution_laywer,defendant,defence_laywer,judge)

        f = open(log_file_path,"a")

        ver = court.simulate_trial()

        f.close()

        if(ver != 0 and ver != 1):
            print(f"Error {ver}")
            ver = 0


        case_IDs.append(case_ID)
        vers.append(ver)

        new_row = {'id':case_ID,'Label':ver}
        new_row_df = pd.DataFrame([new_row])

        # Append to CSV
        new_row_df.to_csv('Result.csv', mode='a', header=False, index=False,index_label=None) 

    for case_no in range(10,100):
        #fullCase = df.loc[case_no,'text']
        case_ID = df.loc[case_no,'id']

        case_IDs.append(case_ID)
        vers.append(0)

    df = pd.DataFrame({
    "Id" : case_IDs,
    "Label": vers
    })
    df.reset_index(drop=True, inplace=True)

    #df.to_csv("Result.csv",index=False,index_label=None)    

    
    