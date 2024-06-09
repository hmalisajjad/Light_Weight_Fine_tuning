from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

# Loading the Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  

model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id  

# Load the dataset
dataset = load_dataset('imdb')

def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

encoded_dataset = dataset.map(preprocess_data, batched=True)

# Evaluation metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# Define training arguments for evaluation
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
)

# Initialize the Trainer with the foundation model
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Initial Evaluation Results: {eval_results}")

# Perform Parameter-Efficient Fine-Tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=["attn.c_attn"]
)

# Create the PEFT model
peft_model = get_peft_model(model, peft_config)

# Define training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Initialize the Trainer with the PEFT model
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

# Train the PEFT model
trainer.train()

# Save the trained PEFT model
peft_model.save_pretrained('./trained_peft_model')

# Load the trained PEFT model
peft_model = GPT2ForSequenceClassification.from_pretrained('./trained_peft_model', num_labels=2)
peft_model.config.pad_token_id = tokenizer.pad_token_id 

# Ensure the model is in evaluation mode
peft_model.eval()

# Initialize the Trainer with the PEFT model for evaluation
trainer = Trainer(
    model=peft_model,
    args=training_args,
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

# Evaluate the PEFT model
peft_eval_results = trainer.evaluate()
print(f"Evaluation Results with PEFT Model: {peft_eval_results}")
