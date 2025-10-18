# app.py
import streamlit as st
import torch
import torch.nn as nn
import pickle
import json
import os
import pandas as pd
import numpy as np

# --- Helper Functions and Classes (Copy from your notebook) ---

# Special token indices (must match training)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Model Definitions (Encoder, Decoder, Seq2Seq) - Copy from your notebook
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions must match!"
        assert encoder.n_layers == decoder.n_layers, "Number of layers must match!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=self.device)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs

# Inference Functions (Greedy, Beam) - Copy from your notebook
def greedy_search(model, src_indices, max_len=50):
    model.eval()
    with torch.no_grad():
        src_tensor = src_indices.unsqueeze(0).to(DEVICE)
        hidden, cell = model.encoder(src_tensor)
        input_token = torch.full((1,), SOS_IDX, dtype=torch.long, device=DEVICE)
        outputs = []

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            predicted_token_idx = output.argmax(1).item()
            outputs.append(predicted_token_idx)

            if predicted_token_idx == EOS_IDX:
                break

            input_token = torch.full((1,), predicted_token_idx, dtype=torch.long, device=DEVICE)

    eos_idx = next((i for i, x in enumerate(outputs) if x == EOS_IDX), len(outputs))
    return outputs[:eos_idx]

def beam_search(model, src_indices, beam_width=3, max_len=50):
    model.eval()
    with torch.no_grad():
        import heapq
        src_tensor = src_indices.unsqueeze(0).to(DEVICE)
        hidden, cell = model.encoder(src_tensor)

        initial_input = torch.full((1,), SOS_IDX, dtype=torch.long, device=DEVICE)
        output, hidden, cell = model.decoder(initial_input, hidden, cell)
        log_probs = torch.log_softmax(output, dim=1)

        top_k_log_probs, top_k_indices = torch.topk(log_probs.squeeze(0), beam_width)
        beam = []
        for log_prob, token_idx in zip(top_k_log_probs, top_k_indices):
            heapq.heappush(beam, (-log_prob.item(), [SOS_IDX, token_idx.item()], hidden, cell))

        finished_sequences = []

        for step in range(1, max_len):
            candidates = []
            for _ in range(len(beam)):
                log_prob, seq, hidden_beam, cell_beam = heapq.heappop(beam)

                if seq[-1] == EOS_IDX:
                    finished_sequences.append((log_prob, seq))
                    continue

                last_token = torch.full((1,), seq[-1], dtype=torch.long, device=DEVICE)
                output, hidden_next, cell_next = model.decoder(last_token, hidden_beam, cell_beam)
                log_probs_next = torch.log_softmax(output, dim=1).squeeze(0)
                top_k_log_probs_next, top_k_indices_next = torch.topk(log_probs_next, beam_width)

                for log_prob_add, token_idx_add in zip(top_k_log_probs_next, top_k_indices_next):
                    new_log_prob = log_prob - log_prob_add.item()
                    new_seq = seq + [token_idx_add.item()]
                    candidates.append((new_log_prob, new_seq, hidden_next, cell_next))

            for c in candidates:
                heapq.heappush(beam, c)

            if len(beam) > beam_width:
                top_beam = heapq.nsmallest(beam_width, beam)
                beam = top_beam
                heapq.heapify(beam)

            if len(finished_sequences) >= beam_width:
                finished_sequences.sort(key=lambda x: x[0])
                if beam and finished_sequences[-1][0] < beam[0][0]:
                    break

        while beam:
            finished_sequences.append(heapq.heappop(beam))

        finished_sequences.sort(key=lambda x: x[0])

        if finished_sequences:
            best_seq = finished_sequences[0][1]
            eos_idx = next((i for i, x in enumerate(best_seq) if x == EOS_IDX), len(best_seq))
            return best_seq[1:eos_idx]
        else:
            best_seq = beam[0][1] if beam else [SOS_IDX]
            return best_seq[1:]

def string_to_indices(text, char_to_idx, unk_idx=UNK_IDX):
    indices = [char_to_idx.get(ch, unk_idx) for ch in text]
    return torch.tensor(indices, dtype=torch.long)

def indices_to_string(indices, idx_to_char):
    tokens = [idx_to_char.get(idx, '<UNK>') for idx in indices]
    filtered_tokens = [t for t in tokens if t not in ['<SOS>', '<EOS>', '<PAD>']]
    return "".join(filtered_tokens)

# --- Streamlit App Code ---

st.title("Hindi to English Transliteration")

# 1. Load Model and Vocabularies
@st.cache_resource
def load_resources():
    # Load model config
    with open('data/model_config.json', 'r') as f:
        model_config = json.load(f)

    # Load vocabularies (Assuming you saved them using pickle in the notebook)
    with open('src_char_to_idx.pkl', 'rb') as f:
        src_char_to_idx = pickle.load(f)
    with open('tgt_char_to_idx.pkl', 'rb') as f:
        tgt_char_to_idx = pickle.load(f)
    with open('tgt_idx_to_char.pkl', 'rb') as f:
        tgt_idx_to_char = pickle.load(f)

    # Determine device (CPU for local Streamlit)
    DEVICE = torch.device("cpu") # Streamlit usually runs on CPU unless configured otherwise
    print(f"Using device for inference: {DEVICE}")

    # Reconstruct model
    enc = Encoder(model_config["src_vocab_size"], model_config["emb_dim"], model_config["hid_dim"], model_config["num_layers"], model_config["dropout"])
    dec = Decoder(model_config["tgt_vocab_size"], model_config["emb_dim"], model_config["hid_dim"], model_config["num_layers"], model_config["dropout"])
    model = Seq2Seq(enc, dec, DEVICE)

    # Load trained weights
    model_path = "models/best_lstm_model.pt" # Adjust path if needed
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run the training notebook first.")
        st.stop() # Stop execution if model is missing
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode

    return model, src_char_to_idx, tgt_char_to_idx, tgt_idx_to_char, DEVICE

model, src_char_to_idx, tgt_char_to_idx, tgt_idx_to_char, DEVICE = load_resources()

st.success("Model and vocabularies loaded successfully!")

# 2. Display Evaluation Results (Hardcoded or from CSV)
st.header("Model Evaluation Results (Test Set)")

# You can either hardcode the results from your notebook output here:
# Or load them from a saved CSV if you created one.
# Example hardcoded results (replace with your actual results):
eval_results_data = {
    "Metric": ["Word-Level Exact Accuracy (ACC)", "Character-Level F1 Score (Mean)"],
    "Greedy Search": [0.0, 0.0], # Replace with actual values
    "Beam Search (beam=5)": [0.0, 0.0] # Replace with actual values
}

eval_results_df = pd.DataFrame(eval_results_data)
st.table(eval_results_df)

# 3. Interactive Transliteration
st.header("Transliterate Hindi to English")
input_text = st.text_input("Enter Hindi text:", value="नमस्ते")

if st.button("Transliterate"):
    if input_text:
        # Convert input string to indices tensor
        src_tensor = string_to_indices(input_text, src_char_to_idx)

        with st.spinner("Transliterating..."):
            # --- Greedy Search ---
            greedy_indices = greedy_search(model, src_tensor, max_len=50)
            greedy_output = indices_to_string(greedy_indices, tgt_idx_to_char)

            # --- Beam Search ---
            beam_width = 5
            beam_indices = beam_search(model, src_tensor, beam_width=beam_width, max_len=50)
            beam_output = indices_to_string(beam_indices, tgt_idx_to_char)

        # Display results
        st.subheader("Results:")
        st.write(f"**Input:** {input_text}")
        st.write(f"**Greedy Search Output:** {greedy_output}")
        st.write(f"**Beam Search Output (beam={beam_width}):** {beam_output}")
    else:
        st.warning("Please enter some Hindi text to transliterate.")

# Optional: Add a section for potential errors or notes
# st.sidebar.header("Notes")
# st.sidebar.info("This model was trained on the Aksharantar Hindi-English dataset.")
# st.sidebar.info("Ensure the input text is in Devanagari script.")