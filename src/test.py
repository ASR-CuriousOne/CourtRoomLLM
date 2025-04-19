import os
from groq import Groq
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (Groq API key)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Preprocess CSV to Summarize Case Descriptions
def summarize_case(description, max_words=300):
    """Summarize a case description to ~300 words."""
    prompt = f"Summarize the following case description in 200–300 words, focusing on the parties, allegations, evidence, and key events: {description}"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.5,
        max_completion_tokens=int(max_words * 1.33) # ~300 words
    )
    return response.choices[0].message.content


# Base Agent Class
class Agent:
    def __init__(self, role, personality = None ,model="meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.role = role
        self.model = model
        self.personality = personality

    def generate_response(self, prompt, temperature=0.7, max_tokens=500):
        """Generate a response using the Groq API."""

        if(self.personality != None):
            prompt += f" Personality and style of response should be of following: {self.personality}. \
                Only text as roleplay should be included."

        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response for {self.role}: {str(e)}"

# Role-Specific Agent Classes
class Defendant(Agent):
    def testify(self, case_description):
        prompt = f"As the Defendant in the case '{case_description}', provide your testimony defending yourself."
        return self.generate_response(prompt)

class Plaintiff(Agent):
    def testify(self, case_description):
        prompt = f"As the Plaintiff in the case '{case_description}', provide your testimony supporting your accusation."
        return self.generate_response(prompt)

class Lawyer(Agent):
    def make_statement(self, case_description, phase):
        prompt = f"As the {self.role} in the case '{case_description}', provide a {phase} statement."
        return self.generate_response(prompt)

    def interrogate_witness(self, witness_statement):
        prompt = f"As the {self.role}, interrogate the witness based on their statement: '{witness_statement}'."
        return self.generate_response(prompt)

class Judge(Agent):
    def deliberate(self, trial_log):
        prompt = f"As the Judge, deliberate on the case based on this trial log: \n{trial_log}\nProvide a verdict and reasoning."
        return self.generate_response(prompt)

class Witness(Agent):
    def testify(self, case_description, context=""):
        prompt = f"As a Witness in the case '{case_description}', provide testimony based on this context: '{context}'."
        return self.generate_response(prompt)

# Central Controller
class CourtroomSimulation:
    def __init__(self, case_description, 
                 plantiff: Plaintiff,
                 prosecution_lawyer: Lawyer,
                 defendant: Defendant,
                 defence_lawyer: Lawyer,
                 judge: Judge):
        self.case_description = case_description
        self.trial_log = []
        self.agents = {
            "defendant": defendant,
            "plaintiff": plantiff,
            "defense_lawyer": defence_lawyer,
            "prosecution_lawyer": prosecution_lawyer,
            "judge": judge
        }
        self.witnesses = []

    def log(self, entry):
        """Add an entry to the trial log and print it."""
        print(entry)
        self.trial_log.append(entry)

    def run_opening_statements(self):
        self.log("\n--- Opening Statements ---")
        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            statement = self.agents[lawyer].make_statement(self.case_description, "opening")
            self.log(f"{lawyer.replace('_', ' ').title()}: {statement}")

    def run_interrogation(self):
        self.log("\n--- Witness Interrogation & Argumentation ---")
        # Plaintiff and Defendant testify
        plaintiff_testimony = self.agents["plaintiff"].testify(self.case_description)
        self.log(f"Plaintiff Testimony: {plaintiff_testimony}")
        defendant_testimony = self.agents["defendant"].testify(self.case_description)
        self.log(f"Defendant Testimony: {defendant_testimony}")

        # Lawyers interrogate (example with static witness for simplicity)
        witness = Witness("Witness")
        witness_statement = witness.testify(self.case_description, "I saw the incident occur.")
        self.log(f"Witness Testimony: {witness_statement}")
        self.witnesses.append(witness)

        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            interrogation = self.agents[lawyer].interrogate_witness(witness_statement)
            self.log(f"{lawyer.replace('_', ' ').title()} Interrogation: {interrogation}")

    def run_closing_statements(self):
        self.log("\n--- Closing Statements ---")
        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            statement = self.agents[lawyer].make_statement(self.case_description, "closing")
            self.log(f"{lawyer.replace('_', ' ').title()}: {statement}")

    def run_ruling(self):
        self.log("\n--- Judge’s Ruling ---")
        verdict = self.agents["judge"].deliberate("\n".join(self.trial_log))
        self.log(f"Judge: {verdict}")

    def simulate_trial(self):
        """Run the full trial simulation."""
        self.run_opening_statements()
        self.run_interrogation()
        self.run_closing_statements()
        self.run_ruling()
        return "\n".join(self.trial_log)

# Demo
if __name__ == "__main__":
    datasetPath = "resources/data.csv"
    df = pd.read_csv(datasetPath)

    CASE_NO = 1

    fullCase = df.loc[CASE_NO,'text']

    summarizedCase = summarize_case(fullCase)
    
    sim = CourtroomSimulation(summarizedCase)
    sim.simulate_trial()