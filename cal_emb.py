from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import evaluate
import numpy as np
import os
from tqdm import tqdm
import pickle
import random
import pandas as pd
from typing import Tuple
from collections import defaultdict
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
)

from trl import RewardTrainer
from trl.trainer.utils import (
    decode_and_strip_padding,
    print_rich_table,
)
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from accelerate.utils import gather_object
from transformers.utils import PaddingStrategy
from transformers.trainer_pt_utils import nested_detach
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["HF_TOKEN"] = 'xxxxxxxxxxxxxxxxxx'

from PIL import Image
import io
from transformers import Qwen2VLForConditionalGeneration  # Use specific for Qwen VL

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=5e-6)
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler"},)
    max_length: Optional[int] = field(default=1024)  # 1024, 512, 600
    use_lora: Optional[bool] = field(default=False)
    base_model: Optional[str] = field(default='Qwen/Qwen2.5-VL-7B-Instruct')
    wandb_name: Optional[str] = field(default="qwen_vl_reward_sb_bench")
    log_dir: Optional[str] = field(default='./reward_models_sb_bench')
    loss_type: Optional[str] = field(default='origin')
    use_smallset: Optional[bool] = field(default=False)
    freeze_pretrained: Optional[bool] = field(default=False)
    data_path: Optional[str] = field(default='/content/drive/MyDrive/Debias_VLMs/Sb-Bench_Dataset/Real')  # Path to directory with Parquet files
    save_steps: Optional[int] = field(default=100)
    cls_embs_path: Optional[str] = field(default='/content/drive/MyDrive/Debias_VLMs/Embeddings')
    debug: Optional[bool] = field(default=False)
    batch_size: Optional[int] = field(default=4)  # New: Batch size for DataLoader

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_name = script_args.base_model
processor_name = model_name
data_path = script_args.data_path

token_patterns = {
    "qwen": [151644, 46593, 198],  # <|im_start|>assistant\n - Adjust based on actual tokens if needed
}

def find_token_for_gating(lst, model_family):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")

def build_dataset_mix(data_path, processor, split='train', size=None, specific_file='0000.parquet'):
    import glob
    if specific_file:
        parquet_file = os.path.join(data_path, specific_file)
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Specified file {parquet_file} not found.")
        dfs = [pd.read_parquet(parquet_file, engine="fastparquet")]
    else:
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {data_path}")
        dfs = [pd.read_parquet(parquet_files[0], engine="fastparquet")]  # Default to first file if no specific file
    full_df = pd.concat(dfs, ignore_index=True)

    ds = Dataset.from_pandas(full_df)

    if size is not None:
        ds = ds.select(range(0, size))

    # Add num_index for dataset
    new_column = list(range(len(ds)))
    ds = ds.add_column("data_index", new_column)
    print("length of dataset:", len(ds))
    
    def formatting_func(example):
        import ast
        additional_metadata = ast.literal_eval(example['additional_metadata'])

        # Image from binary
        image_data = example['file_name.bytes']
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preferred answer
        label = example['label']
        chosen = example[f'ans{label}']

        # Rejected: Pick a stereotypical one, e.g., ans0 if not label, else ans1 (simplify)
        rejected_idx = 0 if label != 0 else 1
        rejected = example[f'ans{rejected_idx}']

        # Prompt text
        prompt_text = example['context'] + " " + example['question']

        # Messages for chosen and rejected
        chosen_messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text + " Answer: " + chosen}]}
        ]
        rejected_messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text + " Answer: " + rejected}]}
        ]

        # Prompt only for length
        prompt_messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
        ]

        # Apply chat template
        prompt_plus_chosen = processor.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected = processor.apply_chat_template(rejected_messages, tokenize=False)
        prompt_template = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        # Process with processor components
        kwargs = {"padding": "max_length", "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}

        # Tokenize text using processor.tokenizer
        text_inputs_chosen = processor.tokenizer(prompt_plus_chosen, **kwargs)
        text_inputs_rejected = processor.tokenizer(prompt_plus_rejected, **kwargs)

        # Process image using processor.image_processor
        image_inputs = processor.image_processor.preprocess(image, return_tensors="pt")

        # Combine (ensure tensor shapes are correct)
        inputs_chosen = {
            "input_ids": text_inputs_chosen["input_ids"][0],
            "attention_mask": text_inputs_chosen["attention_mask"][0],
            "pixel_values": image_inputs["pixel_values"][0],
        }
        inputs_rejected = {
            "input_ids": text_inputs_rejected["input_ids"][0],
            "attention_mask": text_inputs_rejected["attention_mask"][0],
            "pixel_values": image_inputs["pixel_values"][0],  # Same image for both
        }

        # For prompt length (tokenize text only)
        tokens_prompt = processor.tokenizer.encode(prompt_template)
        model_type = "qwen"
        try:
            prompt_len = find_token_for_gating(tokens_prompt, model_type)
        except:
            prompt_len = len(tokens_prompt) - 1  # Fallback

        return {
            "pixel_values_chosen": inputs_chosen["pixel_values"],
            "input_ids_chosen": inputs_chosen["input_ids"],
            "attention_mask_chosen": inputs_chosen["attention_mask"],
            "pixel_values_rejected": inputs_rejected["pixel_values"],
            "input_ids_rejected": inputs_rejected["input_ids"],
            "attention_mask_rejected": inputs_rejected["attention_mask"],
            "data_index": example['data_index'],
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
            "prompt_plus_chosen_response": prompt_plus_chosen,
            "prompt_plus_rejected_response": prompt_plus_rejected,
            'prompt_length': prompt_len
        }

    ds = ds.map(formatting_func, batched=False, num_proc=1)  # Reduced to 1 for stability
    ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=1)
    
    len_before_filter = len(ds)
    ds = ds.filter(lambda x: x["prompt_length"] < script_args.max_length, num_proc=1)
    len_after_filter = len(ds)
    print(f"{len_after_filter - len_before_filter} prompts' lengths are over the prompt_response")
    
    ds.set_format(type="torch")
    
    return ds

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = model_name.split("/")[-1]
output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_{script_args.learning_rate}"

training_args = TrainingArguments(
    output_dir=os.path.join(output_name, 'logs'),
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=script_args.save_steps, #100
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True, 
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=10,
    warmup_ratio=0.05,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    run_name=script_args.wandb_name,
    max_grad_norm=5.0,
    report_to='none',
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
)

# Load the model and processor.
processor = AutoProcessor.from_pretrained(processor_name)
print("Processor tokenizer type:", type(processor.tokenizer))  # Debug print
print("use build_dataset_mix")
dataset = build_dataset_mix(data_path, processor, split='train')
eval_dataset = dataset

# Create DataLoader for batched loading
train_dataloader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=script_args.batch_size, shuffle=False)

#######################################################
print("Length of train dataset:", len(dataset))
print("Length of eval dataset:", len(eval_dataset))
print("Batch size:", script_args.batch_size)
print("Number of train batches:", len(train_dataloader))
print("Number of eval batches:", len(eval_dataloader))

device = Accelerator().local_process_index 
print("device:", device)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, num_labels=1, device_map=device, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# For classification, add a score head if needed, but since we extract hidden, perhaps no
# But to match, add a linear head for 'reward'
model.score = nn.Linear(model.config.hidden_size, 1, bias=False)  # Add manually

def custom_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,  # Added for VL
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompt_length: Optional[torch.Tensor] = None
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(  # Note: For Qwen VL, it's self.model or self depending on class
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,  # Added
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always True for extraction
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs.hidden_states[-1]  # Last layer hidden states
        logits = self.score(hidden_states)  # [bs, seq, 1]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]  # [bs, 1]

        loss = None
        
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
     
        chose_emb = hidden_states[0, sequence_lengths[0], :]
        rej_emb = hidden_states[1, sequence_lengths[1], :]
        prompt_emb = hidden_states[0, (prompt_length-1):(prompt_length+1), :] 
        
        emb = torch.cat([chose_emb[None,...], rej_emb[None,...], prompt_emb], 0)
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=emb,
            attentions=transformer_outputs.attentions,
        )

# hack model's forward
model.original_forward = model.generate  # Or whatever, but for inference
model.forward = custom_forward.__get__(model, type(model))

# Define the metric that we'll use for validation. (Not used for extraction)
accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

@dataclass
class RewardDataCollatorWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    data_path: str = ""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features_input_ids = []
        merged_features_attention_mask = []
        merged_features_pixel_values = []

        for feature in features:
            # Chosen
            merged_features_input_ids.append(feature["input_ids_chosen"])
            merged_features_attention_mask.append(feature["attention_mask_chosen"])
            merged_features_pixel_values.append(feature["pixel_values_chosen"])
            # Rejected
            merged_features_input_ids.append(feature["input_ids_rejected"])
            merged_features_attention_mask.append(feature["attention_mask_rejected"])
            merged_features_pixel_values.append(feature["pixel_values_rejected"])

        # Stack tensors
        batch = {}
        batch["input_ids"] = torch.stack(merged_features_input_ids)
        batch["attention_mask"] = torch.stack(merged_features_attention_mask)
        batch["pixel_values"] = torch.stack(merged_features_pixel_values)

        batch["prompt"] = features[0]["prompt"]  # Assuming single per batch or adjust
        batch["chosen"] = features[0]["chosen"]
        batch["rejected"] = features[0]["rejected"]
        batch["prompt_plus_chosen_response"] = features[0]["prompt_plus_chosen_response"]
        batch["prompt_plus_rejected_response"] = features[0]["prompt_plus_rejected_response"]
        batch["prompt_length"] = features[0]["prompt_length"]
        batch["data_index"] = features[0]["data_index"]

        return batch

class RewardVisualizer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        res = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"], prompt_length=inputs["prompt_length"])
        rewards = res['logits']
        emb = res['hidden_states']
        
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        ###################################
        if script_args.loss_type == 'origin':
            loss = - nn.functional.logsigmoid(rewards_j - rewards_k).mean() 
        elif script_args.loss_type == 'margin':
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k - torch.tensor(inputs["margin"], device=inputs["margin"][0].device).view(-1,1)).mean()
        elif script_args.loss_type == 'labelsmooth':
            loss = - 0.9 * nn.functional.logsigmoid(rewards_j - rewards_k).mean() - 0.1 * nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
        else:
            raise NotImplementedError

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}, emb
        return loss, emb
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict, emb = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)

        # Stack accepted against rejected, 
        logits = torch.stack(logits).mean(dim=2).T 
        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels, emb

    def visualize_samples(self, num_print_samples: int, cls_embs_path: str, data_path: str):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        if not os.path.exists(cls_embs_path):
            os.makedirs(cls_embs_path)

        print("Length of eval dataset:", len(self.eval_dataset))
        print("Length of eval dataloader:", len(eval_dataloader))
       
        for idx, inputs in tqdm.tqdm(enumerate(eval_dataloader)):
            data_index = inputs["data_index"]
            fn = os.path.join(cls_embs_path, f"emb_{data_index}.npy")
            if os.path.exists(fn):
                continue
            
            _, logits, _, emb = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            
            chosen_text = inputs["chosen"]
            rejected_text = inputs["rejected"]
            prompt = inputs["prompt"]
            source = "sb_bench"  # Fixed for SB-Bench
            
            table["prompt"].append(prompt)
            table["chosen_text"].append(chosen_text)
            table["rejected_text"].append(rejected_text)
            table["prompt_plus_chosen_response"].append(chosen_text)
            table["prompt_plus_rejected_response"].append(rejected_text)
            table["source"].append(source)
            table["data_index"].append(data_index.item())

            if num_print_samples >= 0 and len(table["chosen_text"]) >= num_print_samples:
                break
            
            if logits[0][0] > logits[0][1]:
                table["flag"].extend([1])
            else:
                table["flag"].extend([0])

            # save emb (3, hidden_size) - chose, rej, prompt
            cls_emb = emb.float().cpu().numpy() 
            cls_emb = cls_emb[None,...]
            np.save(fn, cls_emb)

            if Accelerator().num_processes == 1:
                table["cls_emb"].extend(gather_object([fn]))
                table["logits"].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()]))
            else:
                table["cls_emb"].append(fn)
                table["logits"].extend(
                [[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])

            if len(table['chosen_text']) % 2000 == 1:
                df = pd.DataFrame(table)
                df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}.csv")

        df = pd.DataFrame(table)
        df.to_csv(f"data_{os.path.basename(cls_embs_path)}_{Accelerator().local_process_index}.csv")

        return df

if __name__ == "__main__":
    trainer = RewardVisualizer(
    model=model,
    args=training_args,
    tokenizer=processor.tokenizer,  # Use processor's tokenizer
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(processor=processor, max_length=script_args.max_length, data_path=data_path),
    )

    cls_embs_path=script_args.cls_embs_path
    df = trainer.visualize_samples(1e8, cls_embs_path, data_path)