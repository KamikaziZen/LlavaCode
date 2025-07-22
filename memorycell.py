import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import lightning.pytorch as pl

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import load_from_disk

import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from nltk import sent_tokenize
from dotenv import load_dotenv
import pickle

device = torch.device("cuda:0")
load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments with text compression using memory tokens')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-160m', help='Name of the model to use')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for computations')
    parser.add_argument('--use_flash_attention_2', action='store_true', help='Whether to use flash attention 2')
    parser.add_argument('--N_mem_tokens', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help='List of memory token numbers to experiment with')
    parser.add_argument('--max_length', type=int, nargs='+',
                        default=[8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1568],
                        help='List of max lengths to experiment with')
    parser.add_argument('--num_iterations', type=int, default=5000, help='Number of iterations for each experiment')
    parser.add_argument('--lr', type=float, default=1e-02, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='adam beta_1')
    parser.add_argument('--beta_2', type=float, default=0.9, help='adam beta_2')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=2000, help='Early stopping patience')
    parser.add_argument('--shuffled', action='store_true', help='Whether to use random text sampled from GloVe vocab.')
    parser.add_argument('--save_path', type=str, default='./memruns', help='path to save experiments')
    parser.add_argument('--texts_path', type=str, help='path to texts to compress')
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)
    parser.add_argument('--clearml', action='store_true', help='report metrics to clearml, credentials are read from the .env file')
    return parser.parse_args()


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, memory_dim):
        super().__init__()
        self.model = base_model
        self.memory_dim = memory_dim
        self.num_mem_tokens = num_mem_tokens
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        self.create_memory()

    def create_memory(self):
        embeddings = self.model.get_input_embeddings()
        memory_params = torch.randn((self.num_mem_tokens, self.memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_params, requires_grad=True))
        self.read_memory_position = range(self.num_mem_tokens)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, **kwargs)

        # todo: allow labels to be passed, could be used for masking
        labels = input_ids
        logits = out.logits
        labels = labels.to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        out.loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'],
                                  attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        mem_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        mem_kwargs['input_ids'] = None
        mem_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            mem_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        mem_kwargs['output_hidden_states'] = True
        return mem_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens:] = attention_mask
            return mask

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithPast()
            # take read memory here
            memory_state = model_outputs.hidden_states[-1][:, self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:]

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


def calculate_accuracy(logits, labels):
    # bs = 1
    shift_logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    predictions = torch.argmax(shift_logits, dim=-1)
    correct = (predictions == labels).float()
    return correct.mean().item()


def run_single_experiment(N_mem_tokens, text_sample, max_length, num_iterations, sample_idx, run_idx,
                          model_name, dtype, use_flash_attention_2, device, tokenizer, lr, beta_1, beta_2,
                          weight_decay, early_stopping_patience=2000, shuffled=False, logger=None):
    # split text sample on two parts: prefix and main text
    # sentences = sent_tokenize(text_sample)
    # prefix can be used lately for compression analysis
    # prefix_text = ' '.join(sentences[:len(sentences)//2])
    # suffix is compressed
    # suffix_text = ' '.join(sentences[len(sentences)//2:])
    suffix_text = text_sample

    if shuffled:
        vocab = []
        with open('./data/vocab_100k.txt') as fin:
            for line in fin:
                vocab += [line.strip()]
        max_length = np.random.randint(2, max_length+1)
        suffix_text = ' '.join(np.random.choice(vocab, size=max_length * 5))
        inp = tokenizer(suffix_text, max_length=max_length, truncation=True, return_tensors='pt').to(device)
    else:
        inp = tokenizer(suffix_text, max_length=max_length, truncation=True, return_tensors='pt').to(device)

    model = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2=use_flash_attention_2)
    model.to(device)

    with torch.amp.autocast(device_type='cuda', dtype=dtype):
        with torch.no_grad():
            orig_output = model(**inp, labels=inp['input_ids'])
            orig_loss = orig_output.loss.item()
            orig_accuracy = calculate_accuracy(orig_output.logits, inp['input_ids'])

    model = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2=use_flash_attention_2)
    memory_dim = getattr(model.config, 'word_embed_proj_dim', getattr(model.config, 'hidden_size'))
    model_with_memory = MemoryCell(model, N_mem_tokens, memory_dim)
    model_with_memory.to(device)

    opt = AdamW(model_with_memory.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta_1, beta_2))

    desc = (f"Training (m={N_mem_tokens}, l={max_length}, i={sample_idx}), "
            f"no_mem_loss={orig_loss:.4f}, no_mem_acc={orig_accuracy:.4f}")
    progress_bar = tqdm(range(num_iterations), desc=desc, leave=False)

    losses, accuracies = [], []
    best_loss, best_accuracy, = float('inf'), 0
    best_memory_params = None
    early_stopping_counter = 0

    for step in progress_bar:
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            out, mem = model_with_memory(**inp)
            loss = out.loss
            accuracy = calculate_accuracy(out.logits, inp['input_ids'])

        loss.backward()
        opt.step()
        opt.zero_grad()
        current_loss = loss.item()
        losses.append(current_loss)
        accuracies.append(accuracy)

        if logger and step % 20 == 0:
            logger.log_metrics({'loss': current_loss, 'accuracy': accuracy}, step=step) 

        if best_accuracy < accuracy:
            best_loss = current_loss
            best_accuracy = accuracy
            best_memory_params = model_with_memory.memory.data.cpu().numpy()
            if logger:
                logger.log_metrics({'best_loss': best_loss, 'best_accuracy': best_accuracy}, step=step)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}", best_loss=f"{best_loss:.4f}", best_acc=f"{best_accuracy:.4f}")

        if best_accuracy == 1.0:
            break

        if early_stopping_counter >= early_stopping_patience:
            break

    mem_embedding = model_with_memory.memory.data.detach().cpu()

    return mem_embedding, {
        'losses': losses,
        'accuracies': accuracies,
        'original_loss': orig_loss,
        'original_accuracy': orig_accuracy,
        'best_memory_params': best_memory_params,
        'best_loss': best_loss,
        'best_accuracy': best_accuracy,
        'max_length': max_length,
        'n_mem_tokens': N_mem_tokens,
        'suffix_text': suffix_text,
        'args': {
            'N_mem_tokens': N_mem_tokens,
            'max_length': max_length,
            'num_iterations': num_iterations,
            'sample_idx': sample_idx,
            'run_idx': run_idx,
            'model_name': model_name,
            'dtype': dtype,
            'use_flash_attention_2': use_flash_attention_2,
            'device': device,
            'lr': lr,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'weight_decay': weight_decay,
            'shuffled': shuffled},
    }


def main():
    args = parse_arguments()
    print(args)

    train_orig_data = load_from_disk(args.texts_path)
    texts = [
        {"text": "\n".join(text.splitlines()[1:])}  # removing path to file
        for item in train_orig_data
        for text in item["content"]["crossfile_array"][:10]  # taking only top-10 retrieved chunks
    ]
    df = pd.DataFrame(texts)

    if args.clearml:
        from pl_logger import ClearMLLogger
        clearml_logger = ClearMLLogger(project_name='LlavaCode', task_name=f'mem_cell_training_l{args.max_length[0]}_s{args.start_index}-{args.end_index}', tags=[args.model_name.split('/')[-1]])

    samples = df['text'][args.start_index:args.end_index]
    num_runs = 1
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    total_experiments = len(args.max_length) * len(args.N_mem_tokens) * len(samples) * num_runs
    overall_progress = tqdm(total=total_experiments, desc="Overall Progress", position=0)
    for max_length in args.max_length:
        for N_mem_tokens in args.N_mem_tokens:

            if N_mem_tokens > max_length:
                continue

            aggregated_results = []

            save_path = Path(f"./{args.save_path}/{args.model_name.split('/')[-1]}")
            if not args.shuffled:
                save_path = save_path / f'mem_{N_mem_tokens}_len_{max_length}_s{args.start_index}-s{args.end_index}.pkl'
            else:
                save_path = save_path / f'mem_{N_mem_tokens}_len_{max_length}_s{args.start_index}-s{args.end_index}_rnd_vocab_100k.pkl'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f'save_path: {save_path}')

            if save_path.exists():
                print(f'loading previous results from {save_path}')
                aggregated_results = pickle.load(open(save_path, 'rb'))

            mem_embeddings = []
            for sample_idx, sample in enumerate(samples):
                for run in range(num_runs):
                    mem_embedding, result = run_single_experiment(
                        N_mem_tokens, sample, max_length, args.num_iterations, sample_idx,
                        run, args.model_name, dtype, args.use_flash_attention_2, device,
                        tokenizer, args.lr, args.beta_1, args.beta_2, args.weight_decay,
                        args.early_stopping_patience, args.shuffled, logger=clearml_logger if args.clearml else None)
                    aggregated_results.append(result)
                    overall_progress.update(1)
                    pickle.dump(aggregated_results, open(save_path, 'wb'))
                    mem_embeddings.append(mem_embedding)

        overall_progress.close()
        torch.save(torch.vstack(mem_embeddings), f"./{args.save_path}/{args.model_name.split('/')[-1]}/mem_{N_mem_tokens}_len_{max_length}_s{args.start_index}-s{args.end_index}.pt")

    if args.clearml:
        clearml_logger.finalize(status='ok')


if __name__ == "__main__":
    main()
