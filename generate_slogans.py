import argparse
import torch
import pandas as pd
from phonetic_similarity.sim_words import get_phon_and_sem_similar_words
from kobart import get_kobart_tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
from model.SynSemBartForConditionalGeneration import SynSemBartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = get_kobart_tokenizer()
tokenizer.add_tokens('<name>')

pos_tokenizer = BartTokenizer(vocab_file='.model/POSvocab.json', merges_file='.model/merges.txt')

# load model
model_path = "pretrained_LexPOS_ko"
model = SynSemBartForConditionalGeneration.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)


def generate_slogans(input_words, model_path, pos_inputs=None, num_beams=1, temperature=1.0):

    # search for phonetically & semantically similar words to each input word
    sim_words = []
    for word in input_words:
        sim_words += get_phon_and_sem_similar_words(word)[:3]
    print(sim_words)

    # in case no particular POS constraints are given
    if not pos_inputs:
        lexical_inputs = []
        for word in sim_words:
            lexical_inputs += ['<mask> ' +  ' <mask> '.join(input_words + [word]) + ' <mask>'] * 3
        # popular list of POS tags in slogan data
        # ['NNG', 'JK', 'MM', 'NNG']
        # ['NNG', 'JK', 'NNG', 'JK', 'VV', 'EC']
        # ['NNG', 'JK', 'VV', 'EC']
        # ['NNG', 'NNG']
        # ['NNG', 'JK', 'NNG']
        pos_inputs = [['NNG', 'JK', 'MM', 'NNG']*2, ['NNG', 'JK', 'NNG', 'JK', 'VV', 'EC']*2, ['NNG', 'JK', 'VV', 'EC']*2] * (len(lexical_inputs) // 3)

    # in case particular POS constraints are given
    else:
        lexical_inputs = []
        for word in sim_words:
            lexical_inputs += ['<mask> ' +  ' <mask> '.join(input_words + [word]) + ' <mask>']
        pos_inputs = [pos_inputs] * len(lexical_inputs)

    # generate slogans
    preds = []
    for lexical_input, pos_input in zip(lexical_inputs, pos_inputs):
        inputs = tokenizer(lexical_input, return_tensors="pt").to(device)
        pos_inputs = pos_tokenizer.encode_plus(pos_input, is_pretokenized=True, return_tensors='pt').to(device)
        outputs = model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask,
                                 pos_input_ids = pos_inputs.input_ids, \
                                 pos_attention_mask = pos_inputs.attention_mask, num_beams=3, temperature=1.2)

        preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return preds



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', dest='keywords', help='input keywords to generate slogan with, delimited by comma')
    parser.add_argument('--pos_inputs', help='list of POS tags delimited by comma', type=str)
    parser.add_argument('--num_beams', help='Number of beams for beam search. Default to 1.', type=int, default=1)
    parser.add_argument('--temperature', help=' The value used to module the next token probabilities. Default to 1.0.', type=float, default=1.0)
    parser.add_argument('--model_path', type=str, default="pretrained_LexPOS_ko")
    args = parser.parse_args()

    if args.keywords:
        pos_inputs = None
        if args.pos_inputs:
            pos_inputs = [str(tag) for tag in args.pos_inputs.split(',')]
        preds = generate_slogans(input_words=str(args.keywords).split(','), model_path=args.model_path, pos_inputs=pos_inputs, num_beams=args.num_beams, temperature=args.temperature)
        print(preds)

