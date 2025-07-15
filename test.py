import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Set up device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Define the same model class used during training
class BugPredictor(nn.Module):
    def __init__(self, n_classes=3):  # Set to 3 since model was trained with 3 classes
        super(BugPredictor, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert_model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

# 🔁 Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 🧠 Load the trained model weights
model = BugPredictor(n_classes=3)
model_path = r"D:\Company_Data\Bug_prediction\Flask-Bug-Tracker\Models\best_model.bin"
# 🧠 Load the trained model weights (with strict=False to avoid unexpected key errors)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

model.to(device)
model.eval()

# 👨‍💻 Your custom input code snippet (change this as needed)
custom_code = "for(int i = 0; i < 10; i++) { cout << i; }"

# ✏️ Tokenize the input
inputs = tokenizer.encode_plus(
    custom_code,
    max_length=512,
    truncation=True,
    add_special_tokens=True,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt'
)

input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

# 🔍 Predict
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    prediction = torch.argmax(outputs, dim=1).item()

# 🏷️ Label map (change this to match your actual training labels)
labels = ["clean", "buggy", "needs review"]
print(f"🧪 Predicted Class: {labels[prediction]}")
