
import torch



def get_loss_scores(model, tokenizer, dataset):
    """
    Get the loss scores for the model on the dataset.
    """
    # Set the model to evaluation.
    model.eval()
    
    loss_scores = []
    
    # For each sample in the dataset, get the scores.
    for sample in dataset:
        input_batch = tokenizer.prepare_seq2seq_batch(
            src_texts=sample,
            tgt_texts=gold_labels,
            max_length=self.args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = input_batch["input_ids"]
        attention_mask = input_batch["attention_mask"]
        labels = input_batch["labels"]

        batch = (input_ids, attention_mask, labels)
        inputs = get_inputs_dict(batch)

            with torch.no_grad():  # Do not compute gradients.
                outputs = self.model(**inputs)
                loss = outputs[0]
            return loss.item()