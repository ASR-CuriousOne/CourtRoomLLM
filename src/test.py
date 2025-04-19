import os
import pandas as pd
from enum import Enum
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define the Hugging Face model to use
MODEL_NAME = "gpt2"  # Replace with your preferred model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional: Use 4-bit quantization for large models
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=quantization_config if DEVICE == "cuda" else None,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully.")

# Base Agent Class
class Agent:
    def __init__(self, role, summary_case='', personality=None, model_name=MODEL_NAME):
        self.role = role
        self.model_name = model_name
        self.case_summary = summary_case
        self.personality = personality

    def generate_response(self, prompt, temperature=0.7, max_tokens=200):
        """Generate a response using the local Hugging Face model."""
        if self.personality:
            prompt += (f" Personality and style of response should be of following: {self.personality}. "
                       "Only text as roleplay should be included. Response should be only 100 words")

        try:
            # Prepare input for the model
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            # Decode and clean the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response if included
            response = response[len(prompt):].strip()
            # Truncate to 100 words if personality is specified
            if self.personality:
                words = response.split()
                response = " ".join(words[:100])
            return response
        except Exception as e:
            return f"Error generating response for {self.role}: {str(e)}"

# Role-Specific Agent Classes
class Defendant(Agent):
    def testify(self):
        prompt = f"As the Defendant in the case '{self.case_summary}', provide your testimony defending yourself."
        return self.generate_response(prompt)

class Plaintiff(Agent):
    def testify(self):
        prompt = f"As the Plaintiff in the case '{self.case_summary}', provide your testimony supporting your accusation."
        return self.generate_response(prompt)

class Lawyer(Agent):
    def make_statement(self, phase):
        prompt = f"As the {self.role} in the case '{self.case_summary}', provide a {phase} statement."
        return self.generate_response(prompt)

    def interrogate_witness(self, witness_statement):
        prompt = f"As the {self.role}, interrogate the witness based on their statement: '{witness_statement}'."
        return self.generate_response(prompt)

    def call_witness(self, summary, trial_log):
        prompt = (f"As the {self.role}, based on the case summary: {summary} and trial log: {trial_log}, "
                  "suggest a witness to call (e.g., bystander, expert) and provide a brief reason (50 words).")
        return self.generate_response(prompt, max_tokens=70)

class Judge(Agent):
    def deliberate(self, trial_log):
        prompt = (
            f"As the Judge, review the following trial log: {trial_log}. "
            "Provide a clear verdict with only a single word between GRANTED and DENIED in the last line only. "
            "The reasoning must be based solely on the evidence, witness statements, and applicable legal principles presented in the trial log."
        )
        return self.generate_response(prompt)

class Witness(Agent):
    def testify(self, context=""):
        prompt = f"As a Witness in the case '{self.case_summary}', provide testimony based on this context: '{context}'."
        return self.generate_response(prompt)

# Central Controller
class CourtroomSimulation:
    def __init__(self, case_descriptions, log_file_path, plaintiff, prosecution_lawyer, defendant, defence_lawyer, judge):
        self.case_descriptions = case_descriptions
        self.trial_log = []
        self.agents = {
            "defendant": defendant,
            "plaintiff": plaintiff,
            "defense_lawyer": defence_lawyer,
            "prosecution_lawyer": prosecution_lawyer,
            "judge": judge
        }
        self.witnesses = []
        self.max_witnesses_per_lawyer = 2
        self.log_file_path = log_file_path

    def log(self, entry):
        """Add an entry to the trial log and print it."""
        self.trial_log.append(entry)
        self.trial_log.append("\n-----\n")

    def run_opening_statements(self):
        self.log("\n--- Opening Statements ---")
        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            statement = self.agents[lawyer].make_statement("opening")
            self.log(f"{lawyer.replace('_', ' ').title()}: {statement}")

    def run_interrogation(self):
        self.log("\n--- Witness Interrogation & Argumentation ---")
        plaintiff_testimony = self.agents["plaintiff"].testify()
        self.log(f"Plaintiff Testimony: {plaintiff_testimony}")
        defendant_testimony = self.agents["defendant"].testify()
        self.log(f"Defendant Testimony: {defendant_testimony}")

        witness = Witness("Witness", self.case_descriptions["defendant"])
        witness_statement = witness.testify("I saw the incident occur.")
        self.log(f"Witness Testimony: {witness_statement}")
        self.witnesses.append(witness)

        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            interrogation = self.agents[lawyer].interrogate_witness(witness_statement)
            self.log(f"{lawyer.replace('_', ' ').title()} Interrogation: {interrogation}")

    def run_closing_statements(self):
        self.log("\n--- Closing Statements ---")
        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            statement = self.agents[lawyer].make_statement("closing")
            self.log(f"{lawyer.replace('_', ' ').title()}: {statement}")

    def run_ruling(self):
        self.log("\n--- Judge’s Ruling ---")
        verdict = self.agents["judge"].deliberate("\n".join(self.trial_log))
        self.log(f"Judge: {verdict}")
        return verdict

    def simulate_trial(self):
        """Run the full trial simulation."""
        self.run_opening_statements()
        self.run_interrogation()
        self.run_closing_statements()
        ver = self.run_ruling()
        if "GRANTED" in ver:
            return 1
        elif "DENIED" in ver:
            return 0
        else:
            return "Neither GRANTED nor DENIED found"

# Enum for Agent Types
class AgentType(Enum):
    PROSECUTOR = {'role': "prosecution_lawyer", 'focus': "evidence of guilt, witness statements against the defendant, and legal arguments for conviction"}
    DEFENDANT = {'role': "defendant", 'focus': "personal account, explanations, mitigating circumstances, and character evidence"}
    PLAINTIFF = {'role': "plaintiff", 'focus': "details of the incident, harm suffered, and evidence implicating the defendant"}
    DEFENSE = {'role': "defense_lawyer", 'focus': "evidence of innocence, alibi, character witnesses, and legal arguments for acquittal"}

# Preprocess CSV to Summarize Case Descriptions
def summarize_case(description, summary_type, max_words=300):
    """Summarize a case description to ~300 words."""
    role = summary_type['role']
    focus = summary_type['focus']
    prompt = (f"Provide only the concise summary of the following case description in 100 words, "
              f"written from the perspective of the {role}, focusing on {focus}. "
              f"The summary should be designed for the {role} to carry with them to the trial for quick reference, "
              f"and no introductory or explanatory text should be included: {description}")

    # Use the local model to generate the summary
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_words * 1.33),
        temperature=0.5,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

if __name__ == "__main__":
    dataset_path = "resources/cases.csv"
    log_file_path = "logs.txt"
    df = pd.read_csv(dataset_path)

    case_IDs = []
    vers = []

    # Placeholder for personalities (define as needed)
    prosecution_personality = "Confident, assertive, and prosecutorial"
    defence_personality = "Empathetic, strategic, and defensive"
    judge_personality = "Neutral, authoritative, and judicious"

    for case_no in tqdm(range(10, 50)):
        fullCase = df.loc[case_no, 'text']
        case_ID = df.loc[case_no, 'id']

        print("Generating Summaries")
        case_descriptions = {
            "defendant": summarize_case(fullCase, AgentType.DEFENDANT.value),
            "plaintiff": summarize_case(fullCase, AgentType.PLAINTIFF.value),
            "defense_lawyer": summarize_case(fullCase, AgentType.DEFENSE.value),
            "prosecution_lawyer": summarize_case(fullCase, AgentType.PROSECUTOR.value)
        }
        print("Summaries Generated\n")

        personalities = {
            "prosecution_lawyer": prosecution_personality,
            "defence_lawyer": defence_personality,
            "judge": judge_personality
        }

        # Initialize agents
        prosecution_lawyer = Lawyer("Prosecution Lawyer", case_descriptions["prosecution_lawyer"], personalities["prosecution_lawyer"])
        defence_lawyer = Lawyer("Defence Lawyer", case_descriptions["defense_lawyer"], personalities["defence_lawyer"])
        defendant = Defendant("Defendant", case_descriptions["defendant"])
        plaintiff = Plaintiff("Plaintiff", case_descriptions["plaintiff"])
        judge = Judge("Judge", personality=personalities["judge"])

        court = CourtroomSimulation(case_descriptions, log_file_path, plaintiff, prosecution_lawyer, defendant, defence_lawyer, judge)

        with open(log_file_path, "a") as f:
            ver = court.simulate_trial()

        if ver not in [0, 1]:
            print(f"Error {ver}")
            ver = 0

        case_IDs.append(case_ID)
        vers.append(ver)

        new_row = {'id': case_ID, 'Label': ver}
        new_row_df = pd.DataFrame([new_row])
        new_row_df.to_csv('ResultTest.csv', mode='a', header=False, index=False, index_label=None)

    for case_no in range(50, 100):
        case_ID = df.loc[case_no, 'id']
        case_IDs.append(case_ID)
        vers.append(0)

    