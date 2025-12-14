import pandas as pd
import click
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_parameters, log_info, log_warn, log_error
from utils.parameter_handling import compute_secondary_parameters
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.nn import CrossEntropyLoss


loaded_parameters = load_parameters()


def safe_mean(stuff):
    if len(stuff) == 0:
        return None
    return sum(stuff) / len(stuff)

class Perplexity:
    def __init__(self, model_name="gpt2", batch_size=8):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()    
        self.device = self.model.device

    def compute(
        self, texts, batch_size: int = 16, add_start_token: bool = True,  max_length=None
    ):
        if max_length is None:
            max_tokenized_len = self.tokenizer.model_max_length
        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}    
    
    def add_perplexity_column(self, df, text_column="output", batch_size=8, perplexity_column="perplexity"):
        texts = df[text_column].apply(lambda x: x[0] if isinstance(x, list) else x)
        assert texts.isnull().sum() == 0, "Some text columns have null values"
        df[perplexity_column] = None
        perplexities = self.compute(texts.tolist(), batch_size=batch_size)["perplexities"]
        df[perplexity_column] = perplexities
        return df
    
@click.command()
@click.option("--input_csvs", required=True, multiple=True, help="Input CSV files to evaluate")
@click.option("--perplexity_model", required=True, help="Model name for perplexity computation")
@click.option("--perplexity_column", default="perplexity", help="Column name to store perplexity scores")
@click.option("--overwrite", is_flag=True, help="Whether to overwrite existing output files")
@click.option("--batch_size", default=8, help="Batch size for perplexity computation")
@click.pass_obj
def evaluate_perplexity(parameters, input_csvs, perplexity_model, perplexity_column, overwrite, batch_size):
    parameters = load_parameters(parameters)
    log_info(f"Evaluating perplexity using model {perplexity_model}", parameters)
    perplexity_evaluator = Perplexity(model_name=perplexity_model, batch_size=batch_size)
    for input_csv in input_csvs:
        log_info(f"Processing file {input_csv}", parameters)
        if "_perp" not in input_csv:
            output_csv = input_csv.replace(".csv", f"_perp.csv")
        else:
            output_csv = input_csv
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            if perplexity_column in existing_df.columns and not overwrite:
                log_warn(f"Output file {output_csv} already exists. Skipping.", parameters)
                continue
        df = pd.read_csv(input_csv)
        df = perplexity_evaluator.add_perplexity_column(df, perplexity_column=perplexity_column)
        df.to_csv(output_csv, index=False)
        log_info(f"Saved results to {output_csv}", parameters)


@click.group()
@click.option("--random_seed", default=loaded_parameters["random_seed"], help="The random seed for the project")
@click.pass_context
def main(ctx, **input_parameters):
    loaded_parameters.update(input_parameters)
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = loaded_parameters

main.add_command(evaluate_perplexity, name="evaluate_perplexity")


if __name__ == "__main__":
    main()