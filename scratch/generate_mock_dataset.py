import json
import os

mock_dataset = [
    {
        "question": "What is the liability cap in the Master Services Agreement?",
        "ground_truth": "The total liability shall not exceed $1,000,000.",
        "source": "Section 3 of sample_msa.docx",
        "category": "Single-doc factual"
    },
    {
        "question": "What is the project completion date for SOW-003?",
        "ground_truth": "The project completion date is December 31, 2024.",
        "source": "Section 2 of sample_sow.docx",
        "category": "Single-doc factual"
    },
    {
        "question": "Which agreement governs SOW-003 and what is its liability limit?",
        "ground_truth": "SOW-003 is governed by the Master Services Agreement, which has a liability limit of $1,000,000.",
        "source": "SOW-003 and MSA Section 3",
        "category": "Multi-hop reasoning"
    },
    {
        "question": "Is there a confidentiality clause in the NDA?",
        "ground_truth": "Yes, Section 1 covers Confidential Information.",
        "source": "Section 1 of sample_nda.docx",
        "category": "Single-doc factual"
    }
]

output_path = "data/evaluation/golden_dataset.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(mock_dataset, f, indent=2)

print(f"Mock golden dataset created at {output_path}")
