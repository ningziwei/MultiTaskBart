

def evaluate(model,loader):
    tag_dict = loader.dataset.id_to_tag_dict
    tokenizer = loader.dataset.tokenizer
    with torch.no_grad():
        model.eval()
        predicts, labels, texts, masks, word_ids = [], [], [], [], []
        for i, batch in enumerate(loader):
            inputs = batch["input_ids"]
            mask = batch["mask"]
            label = batch["tag_ids"]
            predicts += model(inputs, mask, label=None).tolist()
            labels += label.tolist()
            masks += mask.tolist()
            word_ids += batch["word_ids"]
            texts += [tokenizer.convert_ids_to_tokens(sen) for sen in batch["input_ids"].tolist()]
        ep, er, ef, wp, wr, wf = micro_metrics(predicts, labels, word_ids, texts, masks, tag_dict)
        model.train()
        return ep, er, ef, wp, wr, wf