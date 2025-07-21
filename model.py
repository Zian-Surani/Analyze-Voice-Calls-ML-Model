#INSTALL THE DEPENDENCIES
'''!pip install pyttsx3
!pip install transformers
!pip install textblob
!pip install numpy'''

import pyttsx3
import random
import time
from transformers import pipeline
import numpy as np
import pickle
import os

# Component 1: Voice Agent Setup (Call Simulation)

# Setup TTS Engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    tts_engine = None

# Agent Prompts (Actions)
AGENT_PROMPTS = {
    'base_prompt': "Hello! I'm calling to inform you about PM-KUSUM Yojana — a government scheme for solar pumps that can significantly reduce your electricity bills and help the environment. Would you like more details?",
    'simplified_prompt': "Hello! The PM-KUSUM Yojana helps farmers install solar pumps with government help. Want to hear more?",
    'document_info_prompt': "Hello! PM-KUSUM gives subsidies for solar pumps. I can guide you on required documents too.",
    'apology_prompt': "Sorry to bother you! Just a quick word about solar pump subsidies from the government. Want info?",
    'cost_objection_handler': "I understand cost is a concern. This scheme offers significant government subsidies, reducing your out-of-pocket expense. Would you like to know about the subsidy details?",
    'authenticity_handler': "This is a genuine government scheme, the PM-KUSUM Yojana, implemented by the Ministry of New and Renewable Energy. You can verify it on official government websites. Would you like me to tell you how to find more information?",
    'clarification_prompt': "I apologize if that was unclear. Let me explain it more simply. The PM-KUSUM Yojana provides support for farmers to install solar-powered water pumps. This helps save on electricity bills. Is that clearer?",
    'follow_up_request': "Okay, I understand you're busy. Can I note down your request to call you back later? Or would you prefer I share a link via SMS?",
    'closing_success': "That's great! I can help you with the next steps to apply. Would you like to proceed?",
    'closing_failure': "I understand. Thank you for your time. Have a good day.",
    'closing_follow_up': "Okay, I'll make a note to follow up as you requested. Thank you!"
}

# Simulated Farmer Responses
SIMULATED_RESPONSES = {
    'interested': ["I'm very interested. Please tell me how to apply.", "Sounds good. Can you help me apply?", "Tell me more.", "That sounds like a good idea.", "Batao."],
    'confused': ["This is confusing. I don’t understand the process.", "I'm not sure.", "How does it work?", "Is this a real scheme?", "Samajh nahi aaya.", "Kya bol rahe ho?", "What documents do I need?", "Is there any government support?"], # Added some questions here as they often come from confusion or lack of info
    'disinterested': ["Not interested, don’t call again.", "I already have a solar pump.", "Leave me alone.", "Nahi chahiye."],
    'inquisitive': ["Will this cost me anything?", "What documents do I need?", "Is there a subsidy?", "How long does it take?", "Where can I get more information?", "Kitne ka hai?", "Mujhe free chahiye", "Main eligible hoon kya?"], # More focused questions
    'busy': ["Sorry, I'm busy right now.", "Call me back later.", "Thank you, I will think about it.", "Later."], # Responses indicating need for follow-up
    'neutral': ["Okay.", "Hmm.", "Alright.", "Yes."], # Ambiguous or minimal responses
}

# Map response keywords to types for simulation
RESPONSE_TYPE_MAP = {}
for response_type, responses in SIMULATED_RESPONSES.items():
    for resp in responses:
        RESPONSE_TYPE_MAP[resp] = response_type

def get_simulated_response(farmer_state):
    """Selects a simulated response based on the farmer's current state."""
    # Simple mapping for now: if farmer is in a specific state, pick a response from that category
    # Fallback to neutral if state doesn't have a direct response category
    responses = SIMULATED_RESPONSES.get(farmer_state, SIMULATED_RESPONSES['neutral'])
    return random.choice(responses)


def speak(text):
    if tts_engine:
        print(f"Speaking: {text}")
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Error during TTS speak: {e}")
    else:
        print(f"TTS Engine not initialized. Cannot speak: {text}")
        print(f"Speaking (text only): {text}")

# Component 2: Call Analysis Layer

def analyze_farmer_response(farmer_text):
    text_lower = farmer_text.lower()

    # Sentiment Analysis
    sentiment_label = 'neutral'
    sentiment_score = 0.0
    try:
        sentiment_result = sentiment_pipeline(farmer_text)[0]
        sentiment_label = sentiment_result['label'].lower()
        sentiment_score = sentiment_result['score']
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")

    # Farmer Interest Level
    interest_level = 'neutral'
    if any(kw in text_lower for kw in ['interested', 'apply', 'help', 'good', 'tell me more', 'sounds good', 'batao', 'yes']):
        interest_level = 'interested'
    elif any(kw in text_lower for kw in ['confused', 'don’t understand', 'not sure', 'how does it work', 'samajh nahi aaya', 'kya bol rahe ho', 'what']):
        interest_level = 'confused'
    elif any(kw in text_lower for kw in ['not interested', 'don’t call', 'busy', 'sorry', 'already have', 'leave me alone', 'nahi chahiye', 'mat batana', 'no']):
        interest_level = 'disinterested'
    elif any(kw in text_lower for kw in ['what', 'how', 'documents', 'scheme', 'real', 'cost', 'subsidy', 'eligible', 'sarkari', 'government support', 'time', 'long', 'where', 'kitne ka hai', 'mujhe free chahiye', 'main eligible hoon kya']):
         interest_level = 'inquisitive'

    # Intro Clarity (Heuristic)
    intro_clarity_issue = any(kw in text_lower for kw in ['confused', 'don’t understand', 'not sure', 'samajh nahi aaya', 'kya bol rahe ho', 'is this real scheme', 'what', 'how does it work']) or (interest_level == 'disinterested' and len(text_lower.split()) < 5)

    # Objections / Concerns
    objections = {
        'cost': any(kw in text_lower for kw in ['cost', 'kitne ka hai', 'free chahiye', 'subsidy']),
        'eligibility': any(kw in text_lower for kw in ['eligible', 'eligible hoon kya', 'sarkari support']),
        'time': any(kw in text_lower for kw in ['busy', 'time lagta hai', 'how long', 'later', 'dobara call']),
        'authenticity': any(kw in text_lower for kw in ['real scheme', 'sarkari', 'government support'])
    }
    has_objection = any(objections.values())

    # Call Outcome (Based on final state/response in a simplified single-turn)
    # In multi-turn, this would be determined at the end of the conversation
    call_outcome = 'uncertain'
    if interest_level == 'interested' or any(kw in text_lower for kw in ['apply', 'documents', 'tell me more', 'where can i get more information', 'yes', 'help me apply']):
        call_outcome = 'success'
    elif interest_level == 'disinterested' or any(kw in text_lower for kw in ['don’t call', 'leave me alone', 'not interested', 'hang up', 'nahi chahiye', 'no']):
         call_outcome = 'failure'
    elif any(kw in text_lower for kw in ['busy', 'call back later', 'will think about it', 'later', 'dobara call karo']):
        call_outcome = 'follow_up'
    elif has_objection and call_outcome == 'uncertain':
         call_outcome = 'objection_raised'

    return {
        'sentiment_label': sentiment_label,
        'sentiment_score': sentiment_score,
        'interest_level': interest_level,
        'intro_clear': not intro_clarity_issue,
        'objections': objections,
        'has_objection': has_objection,
        'call_outcome': call_outcome,
        'raw_text': farmer_text # Keep raw text for specific keyword checks if needed
    }

# Component 3: Reinforcement Loop

# States: Combine interest_level, intro_clear, has_objection, and sentiment
STATES = []
interest_levels = ['interested', 'confused', 'disinterested', 'inquisitive', 'neutral']
clarity_issues = [True, False]
has_objections = [True, False]
sentiment_labels = ['positive', 'neutral', 'negative']

for interest in interest_levels:
    for clarity_issue in clarity_issues:
        for objection in has_objections:
            for sentiment in sentiment_labels:
                STATES.append(f'interest:{interest}_clarity_issue:{clarity_issue}_objection:{objection}_sentiment:{sentiment}')

# Add start state for the beginning of a call
START_CALL_STATE = 'start_call'
STATES.append(START_CALL_STATE)

ACTIONS = list(AGENT_PROMPTS.keys())

# Q-table initialized with zeros
Q_table = np.zeros((len(STATES), len(ACTIONS)))

# Reward Mapping based on Call Outcome and potentially state transitions
REWARD_MAP = {
    'success': 1,
    'follow_up': 0.3,
    'objection_raised': 0.1,
    'uncertain': -0.1,
    'failure': -1
}

def get_state_index(analysis_results):
    """Maps analysis results to a defined state index."""
    if analysis_results is None: # Handle initial state
        return STATES.index(START_CALL_STATE)

    interest = analysis_results['interest_level']
    clarity_issue = not analysis_results['intro_clear']
    objection = analysis_results['has_objection']
    sentiment = analysis_results['sentiment_label']

    state_str = f'interest:{interest}_clarity_issue:{clarity_issue}_objection:{objection}_sentiment:{sentiment}'
    try:
        return STATES.index(state_str)
    except ValueError:
        print(f"Warning: Unknown state combination during indexing: {state_str}")
        # Fallback to a neutral state if combination is unexpected
        return STATES.index('interest:neutral_clarity_issue:False_objection:False_sentiment:neutral')

# RL Training Loop (Multi-Turn Simulation)
def train_agent(episodes=200, max_turns_per_episode=3, alpha=0.1, gamma=0.9, epsilon=0.2, epsilon_decay=0.995, epsilon_min=0.01):
    global Q_table # Use global Q_table
    epsilon_current = epsilon

    print(f"Starting training for {episodes} episodes with max {max_turns_per_episode} turns...")

    for episode in range(episodes):
        print(f"\n--- 📞 Episode {episode + 1}/{episodes} ---")

        # Start each episode in the initial call state
        current_state_idx = STATES.index(START_CALL_STATE)
        current_state = STATES[current_state_idx]
        print(f"Initial State: {current_state}")

        episode_reward = 0 # Track total reward for the episode

        for turn in range(max_turns_per_episode):
            print(f"-- Turn {turn + 1} --")
            print(f"Current State: {current_state}")

            # Epsilon-greedy action selection from the current state
            if random.uniform(0, 1) < epsilon_current:
                action_idx = random.randint(0, len(ACTIONS) - 1) # Explore
                print("Action: Explore")
            else:
                action_idx = np.argmax(Q_table[current_state_idx]) # Exploit
                print("Action: Exploit")

            action = ACTIONS[action_idx]
            prompt = AGENT_PROMPTS[action]

            print(f"🤖 Voice Agent Prompt: {prompt}")
            speak(prompt)

            # Simulate farmer response based on the *current state* (more realistic)
            # Instead of random, try to pick a response type that matches the state's interest level
            simulated_response_type = current_state.split('_')[0].split(':')[-1] # Extract interest level from state string
            farmer_response = get_simulated_response(simulated_response_type)


            print(f"👨‍🌾 Farmer Response: {farmer_response}")
            speak(f"Farmer says: {farmer_response}")

            # Analyze the farmer's response to determine the *next* state and immediate reward
            analysis_results = analyze_farmer_response(farmer_response)
            print(f"🧠 Analysis Results: {analysis_results}")

            next_state_idx = get_state_index(analysis_results)
            next_state = STATES[next_state_idx]

            # Determine reward based on the outcome of this turn
            # For multi-turn, immediate reward might be small, with a larger reward at the end
            # Let's use a simple immediate reward based on moving towards 'interested' or away from 'disinterested'
            # and penalize confusion/irritation. Also include outcome reward at the end.

            immediate_reward = 0
            if analysis_results['interest_level'] == 'interested':
                 immediate_reward += 0.2
            elif analysis_results['interest_level'] == 'disinterested':
                 immediate_reward -= 0.2
            if analysis_results['intro_clear'] is False:
                 immediate_reward -= 0.1 # Penalty for lack of clarity
            if analysis_results['has_objection']:
                 immediate_reward += 0.1 # Small reward for engagement, even if it's an objection

            print(f"Transitioned to State: {next_state} | Immediate Reward: {immediate_reward:.2f}")

            # Q-value update using the Q-learning formula
            old_value = Q_table[current_state_idx, action_idx]
            next_max = np.max(Q_table[next_state_idx]) if Q_table[next_state_idx].size > 0 else 0.0
            new_value = (1 - alpha) * old_value + alpha * (immediate_reward + gamma * next_max)
            Q_table[current_state_idx, action_idx] = new_value

            print(f"📈 Updated Q-value for ('{current_state}', '{action}'): {Q_table[current_state_idx, action_idx]:.2f}")

            episode_reward += immediate_reward # Accumulate immediate reward

            # Check for terminal state (Success, Failure, or max turns reached)
            if analysis_results['call_outcome'] in ['success', 'failure', 'follow_up']:
                final_reward = REWARD_MAP.get(analysis_results['call_outcome'], 0)
                episode_reward += final_reward # Add final outcome reward
                print(f"Call ended with outcome: {analysis_results['call_outcome']} | Final Reward: {final_reward}")
                # Update Q-value one last time with the final reward and next_max=0 (terminal state)
                old_value = Q_table[current_state_idx, action_idx]
                Q_table[current_state_idx, action_idx] = (1 - alpha) * old_value + alpha * (immediate_reward + final_reward + gamma * 0)
                break # End episode if terminal state reached

            # Move to the next state for the next turn
            current_state_idx = next_state_idx
            current_state = next_state

            time.sleep(0.5) # Reduced sleep for faster training simulation

        print(f"Episode {episode + 1} finished. Total Episode Reward: {episode_reward:.2f}")

        # Decay epsilon after each episode
        epsilon_current = max(epsilon_min, epsilon_current * epsilon_decay)
        print(f"Epsilon after episode {episode + 1}: {epsilon_current:.2f}")


    print("\nTraining finished.")
    save_q_table(Q_table)


# Show final policy
def show_policy():
    """Displays the learned policy from the Q-table."""
    q_table = load_q_table()
    if q_table is None or len(q_table) == 0:
        print("Q-table not loaded or empty. Cannot show policy.")
        return

    print("\n🎯 Final Learned Policy:")
    print("-" * 60)
    # Show policy for the 'start_call' state
    if START_CALL_STATE in STATES:
        start_state_idx = STATES.index(START_CALL_STATE)
        if np.sum(q_table[start_state_idx]) != 0:
            best_action_idx = np.argmax(q_table[start_state_idx])
            best_action = ACTIONS[best_action_idx]
            print(f"Policy for '{START_CALL_STATE}':")
            print(f"  - Best initial prompt: '{AGENT_PROMPTS[best_action]}'")
            print(f"  - Q-value: {q_table[start_state_idx, best_action_idx]:.2f}")
        else:
             print(f"Policy for '{START_CALL_STATE}': No policy learned yet (state not visited).")

    print("-" * 60)
    # Show policy for other relevant states
    print("Learned Policies for other States (if visited and learned):")
    relevant_states = [state for state in STATES if state != START_CALL_STATE]

    if not relevant_states:
         print("  - No other states defined or visited.")
         return

    for state in relevant_states:
        state_idx = STATES.index(state)
        if np.any(q_table[state_idx] != 0):
            best_action_idx = np.argmax(q_table[state_idx])
            best_action = ACTIONS[best_action_idx]
            print(f"Policy for '{state}':")
            print(f"  - Best action: '{ACTIONS[best_action_idx]}' (Prompt: '{AGENT_PROMPTS[best_action]}')")
            print(f"  - Q-value: {q_table[state_idx, best_action_idx]:.2f}")


# File for saving/loading Q-table
Q_TABLE_FILE = "q_table.pkl"

def load_q_table():
    """Loads the Q-table from a file if it exists, otherwise initializes a new one."""
    global STATES, ACTIONS, Q_table
    # Ensure STATES and ACTIONS are consistent with the current script definitions
    STATES = []
    interest_levels = ['interested', 'confused', 'disinterested', 'inquisitive', 'neutral']
    clarity_issues = [True, False]
    has_objections = [True, False]
    sentiment_labels = ['positive', 'neutral', 'negative']

    for interest in interest_levels:
        for clarity_issue in clarity_issues:
            for objection in has_objections:
                for sentiment in sentiment_labels:
                    STATES.append(f'interest:{interest}_clarity_issue:{clarity_issue}_objection:{objection}_sentiment:{sentiment}')

    START_CALL_STATE = 'start_call'
    STATES.append(START_CALL_STATE)
    ACTIONS = list(AGENT_PROMPTS.keys())

    if os.path.exists(Q_TABLE_FILE):
        print(f"Loading Q-table from {Q_TABLE_FILE}")
        try:
            with open(Q_TABLE_FILE, "rb") as f:
                loaded_q_table = pickle.load(f)
            if loaded_q_table.shape == (len(STATES), len(ACTIONS)):
                 Q_table = loaded_q_table
                 return Q_table
            else:
                 print("Loaded Q-table dimensions do not match current state/action space. Initializing new Q-table.")
                 Q_table = np.zeros((len(STATES), len(ACTIONS)))
                 return Q_table
        except Exception as e:
            print(f"Error loading Q-table: {e}. Initializing new Q-table.")
            Q_table = np.zeros((len(STATES), len(ACTIONS)))
            return Q_table

    else:
        print("Q-table file not found, initializing new Q-table.")
        Q_table = np.zeros((len(STATES), len(ACTIONS)))
        return Q_table


def save_q_table(q_table):
    """Saves the Q-table to a file."""
    try:
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(q_table, f)
        print(f"Q-table saved to {Q_TABLE_FILE}")
    except Exception as e:
        print(f"Error saving Q-table: {e}")


# Main Execution
Q_table = load_q_table()

# Ensure STATES and ACTIONS are consistent before training
STATES = []
interest_levels = ['interested', 'confused', 'disinterested', 'inquisitive', 'neutral']
clarity_issues = [True, False]
has_objections = [True, False]
sentiment_labels = ['positive', 'neutral', 'negative']

for interest in interest_levels:
    for clarity_issue in clarity_issues:
        for objection in has_objections:
            for sentiment in sentiment_labels:
                STATES.append(f'interest:{interest}_clarity_issue:{clarity_issue}_objection:{objection}_sentiment:{sentiment}')

START_CALL_STATE = 'start_call'
STATES.append(START_CALL_STATE)
ACTIONS = list(AGENT_PROMPTS.keys())


train_agent(episodes=200, max_turns_per_episode=5)

show_policy()
