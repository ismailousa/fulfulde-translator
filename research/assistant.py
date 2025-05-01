import json
from ollama import chat

# Load JSON data (templates, default inputs, and decorators)
with open("prompts.json", "r") as f:
    data = json.load(f)

TEMPLATES = data["templates"]
DEFAULT_INPUTS = data["default_inputs"]
DECORATORS = data["decorators"]

def display_inputs(task_type, inputs):
    """Display current inputs for a given task in JSON format."""
    print(f"\n=== Current Inputs for {task_type.replace('_', ' ').title()} ===")
    print(json.dumps(inputs, indent=4))
    print("\nModify any values as needed, or press Enter to keep them.")

def modify_inputs(inputs):
    """Allow user to modify input fields interactively."""
    for key, value in inputs.items():
        new_value = input(f"{key} ({value}): ").strip()
        if new_value:
            inputs[key] = new_value
    return inputs

def choose_decorator():
    """Allow the user to choose a decorator."""
    print("\nAvailable Decorators:")
    for i, (name, desc) in enumerate(DECORATORS.items(), 1):
        print(f"{i}. {desc}")

    choice = input("\nChoose a decorator (Enter number, or press Enter for default): ").strip()
    if choice.isdigit() and int(choice) in range(1, len(DECORATORS) + 1):
        return list(DECORATORS.values())[int(choice) - 1]
    return None  # Keep the default

def generate_response(task_type):
    """Generate a response from the model using the specified task type and inputs."""
    if task_type not in TEMPLATES:
        raise ValueError("Invalid task type.")

    # Show current inputs and allow modification
    inputs = DEFAULT_INPUTS[task_type].copy()
    display_inputs(task_type, inputs)
    modified_inputs = modify_inputs(inputs)

    # Allow the user to modify the decorator
    print("\nCurrent Decorator:", modified_inputs.get("decorator", "None"))
    new_decorator = choose_decorator()
    if new_decorator:
        modified_inputs["decorator"] = new_decorator

    # Format the prompt
    formatted_prompt = TEMPLATES[task_type].format(**modified_inputs)

    # Call the model
    response = chat(
        model="llama3.2",  # Replace with your preferred local model
        messages=[{'role': 'user', 'content': formatted_prompt}],
        options={'temperature': 0.7}  
    )

    return response['message']['content']
