
import logging
import transformers
import warnings
import torch

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

def main():
    # Load Huggingface Model
    from utils.import_model import model_from_hf_path
    model = model_from_hf_path(args.model_path,
                args.use_cuda_graph,
                device_map='auto',
            ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)                

    # Quantization

    # Analysis Tool

    # Inference (Chatbot, Perplexity, LM-Eval)


if __name__ == '__main__':
    import argparse
    from utils.common import str2list, str2bool, str2int
    parser = argparse.ArgumentParser()
    # Model and Tasks
    parser.add_argument('--model_path', type=str, default=None)   
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--tasks', type=str2list, default=[])
    parser.add_argument('--num_fewshot', type=str2int, default='none')
    parser.add_argument('--limit', type=str2int, default='none')
    parser.add_argument('--eval_ppl', type=str2bool, default=False)
    parser.add_argument('--eval_ppl_seqlen', type=int, default=2048)
    parser.add_argument('--use_cuda_graph', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    # Quantization Configs
    parser.add_argument('--bits_a', type=int, default=16)
    parser.add_argument('--sym_a', type=str2bool, default=False)
    parser.add_argument('--groupsize_a', type=int, default=-1)
    parser.add_argument('--bits_w', type=int, default=4)
    parser.add_argument('--sym_w', type=str2bool, default=False)
    parser.add_argument('--groupsize_w', type=int, default=-1)    
    # GPTQ Configs
    
    # SpQR Configs
    
    