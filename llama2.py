# Import libraries
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AdamW,
    get_scheduler,
    BitsAndBytesConfig
)
from datasets import Dataset
import bitsandbytes as bnb

# Define model ID and load the tokenizer and model

# LLaMA model ID (ensure you have access or choose another model if it's gated)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # You can use any compatible LLaMA model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token='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')
tokenizer.pad_token = tokenizer.eos_token



# Using 4-bit quantization for efficient model loading (optional for large models)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


# Load the model with sequence classification head for binary classification (0 or 1)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,  # Binary classification (0 or 1)
    device_map="auto",  # This ensures the model is loaded to the appropriate device
    quantization_config=bnb_config,
    use_cache=False, 
    use_auth_token='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'
)

# Resize token embeddings (to accommodate special tokens if any)
#model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id

# Applying LoRA adapters to the quantized model
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                # LoRA rank
    lora_alpha=16,      # LoRA scaling factor
    lora_dropout=0.1    # LoRA dropout
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model) ## Preparing the model for training with k-bit(as specified in the bnb config)
model = get_peft_model(model, lora_config) ## Applying the low-rank adaptation


# Load dataset (make sure you update paths as per your dataset)
train= pd.read_csv('/home/walleed/CERC/formatted_dialog_LLAMA_commenting.csv')
test = pd.read_csv('/home/walleed/CERC/formatted_dialog_LLAMA_commenting_test.csv')
val = pd.read_csv('/home/walleed/CERC/formatted_dialog_LLAMA_commenting_validation.csv')

# Tokenization without padding (first step)
def tokenize_conversation(examples):
    return tokenizer(examples['formatted_dialog_LLAMA'], truncation=False)

# Find the max token length across the datasets
def find_max_token_length(dataframe):
    dataset = Dataset.from_pandas(dataframe)
    tokenized_dataset = dataset.map(tokenize_conversation, batched=True)
    input_ids_list = tokenized_dataset['input_ids']
    max_len = max(len(input_ids) for input_ids in input_ids_list)
    return max_len

# Finding max token lengths for train, val, and test datasets
max_train_len = find_max_token_length(train)
max_val_len = find_max_token_length(val)
max_test_len = find_max_token_length(test)

# Tokenization with padding to the max token length
def tokenize_conversation_maxlen(examples, max_len):
    return tokenizer(examples['formatted_dialog_LLAMA'], max_length=max_len, padding='max_length', truncation=True)

def apply_tokenization_with_padding(dataframe, max_len):
    dataset = Dataset.from_pandas(dataframe)
    tokenized_dataset = dataset.map(lambda x: tokenize_conversation_maxlen(x, max_len), batched=True)

    input_ids = torch.tensor(tokenized_dataset['input_ids'])
    attention_masks = torch.tensor(tokenized_dataset['attention_mask'])
    # Binary classification output (Category is either 0 or 1)
    labels = torch.tensor(tokenized_dataset['numeric_label'])

    return input_ids, attention_masks, labels

# Apply tokenization with padding to max length for train, test, and val datasets
train_input_ids, train_attention_masks, train_labels = apply_tokenization_with_padding(train, max_train_len)
test_input_ids, test_attention_masks, test_labels = apply_tokenization_with_padding(test, max_test_len)
val_input_ids, val_attention_masks, val_labels = apply_tokenization_with_padding(val, max_val_len)

# Create dataset dictionaries for trainer compatibility
train_dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} 
                 for input_ids, attention_mask, labels in zip(train_input_ids, train_attention_masks, train_labels)]
val_dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} 
               for input_ids, attention_mask, labels in zip(val_input_ids, val_attention_masks, val_labels)]
test_dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} 
                for input_ids, attention_mask, labels in zip(test_input_ids, test_attention_masks, test_labels)]

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluate every epoch
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,
)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define the learning rate scheduler
num_training_steps = len(train_dataset) * 3 // training_args.per_device_train_batch_size
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Function to compute evaluation metrics (binary classification)
def compute_metrics(p):
    predictions, labels = p
    preds = torch.argmax(torch.tensor(predictions), dim=-1)  # Use argmax for classification
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, lr_scheduler),
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(test_dataset)
print(results)

model_path = "/home/walleed/CERC/llama_models1/commenting_new"

# Save the model
trainer.save_model(model_path)

# Optionally save the tokenizer as well
tokenizer.save_pretrained(model_path)

