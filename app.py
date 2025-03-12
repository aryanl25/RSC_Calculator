import tkinter as tk
import torch
from rsc_module.preprocessing import preprocess_query
from rsc_module.embedding_classifier import RSCSystem

# 1. Determine the device (GPU if available, else CPU).
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialize the RSC system with the chosen device.
rsc_system = RSCSystem(device=device)

# 3. Load the trained weights from your checkpoint file.
checkpoint = torch.load("rsc_system_weights.pth", map_location=device)
rsc_system.fusion.load_state_dict(checkpoint["fusion_state_dict"])
rsc_system.dim_reducer.load_state_dict(checkpoint["dim_reducer_state_dict"])
rsc_system.classifier.load_state_dict(checkpoint["classifier_state_dict"])

def process_input():
    # Retrieve user inputs from the UI.
    raw_query = query_input.get("1.0", tk.END)
    cleaned_query = preprocess_query(raw_query)
    simulated_result = query_result_input.get("1.0", tk.END).strip()
    user_role = role_entry.get().strip()
    
    # Compute the relevance score using the loaded model.
    score = rsc_system.score_query(cleaned_query, simulated_result, user_role)
    
    # Build and display the output text.
    output_text = (
        f"User Role: {user_role}\n"
        f"Cleaned Query: {cleaned_query}\n"
        f"Simulated Query Result: {simulated_result}\n\n"
        f"Relevance Score: {score:.4f}\n\n"
        "(A lower score may indicate a suspicious query based on content and user role.)"
    )
    result_label.config(text=output_text)

# -------------------- UI Setup --------------------

app = tk.Tk()
app.title("RSC Module Demo - Enhanced Embedding & Fusion")

# Open-ended user role input
role_label = tk.Label(app, text="Enter User Role:")
role_label.pack(pady=(10, 0))
role_entry = tk.Entry(app, width=50)
role_entry.pack(pady=(0, 10))

# SQL query input
query_label = tk.Label(app, text="Enter SQL Query:")
query_label.pack()
query_input = tk.Text(app, height=5, width=80)
query_input.pack(pady=(0, 10))

# Simulated query result input
query_result_label = tk.Label(app, text="Enter Simulated Query Result:")
query_result_label.pack()
query_result_input = tk.Text(app, height=5, width=80)
query_result_input.pack(pady=(0, 10))

# Button to process input
process_button = tk.Button(app, text="Process Input", command=process_input)
process_button.pack(pady=(0, 10))

# Label to display results
result_label = tk.Label(app, text="", justify="left")
result_label.pack(pady=(10, 10))

# Run the main UI loop
app.mainloop()
