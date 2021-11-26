# LexPOS_ko
Generating Korean Slogans with phonetic and structural repetition

## Generating Slogans with Linguistic Features
LexPOS is a sequence-to-sequence transformer model that <b> generates slogans with phonetic and structural repetition</b>. For phonetic repetition, it searches for phonetically similar words with user keywords. Both the sound-alike words and user keywords become the <b>lexical constraints</b> while generating slogans. It also adjusts the logits distribution to implement further <b>phonetic constraints</b>. For structural repetition, LexPOS uses <b>POS constraints</b>. Users can specify any repeated phrase structure by POS tags.

### Generating slogans with lexical, POS constraints 
#### 1. Code 
* Need to download pretrained Korean word2vec model from [here](https://github.com/Kyubyong/wordvectors) and put it below `phonetic_similarity/KoG2P`
```python3
# clone this repo
git clone https://github.com/yeounyi/LexPOS_ko
cd LexPOS
# generate slogans 
python3 generate_slogans.py --keywords 카드,혜택 --num_beams 3 --temperature 1.2
```
- `-keywords`: Keywords that you want to be included in slogans. You can enter multiple keywords, delimited by comma
-  `-pos_inputs`: You can either specify the particular list of POS tags delimited by comma, or the model will generate slogans with the most frequent syntax used in corpus. POS tags generally follow the format of [Konlpy Mecab POS tags](https://konlpy.org/en/latest/api/konlpy.tag/#mecab-class).  
- `-num_beams`: Number of beams for beam search. Default to 1, meaning no beam search.
- `-temperature`: The value used to module the next token probabilities. Default to 1.0.
- `-model_path`: Path to the pretrained model

#### 2. Examples 

<b>Keyword</b>: `카드, 혜택` <br>
<b>POS</b>: `[NNG, JK, VV, EC, SF, NNG, JK, VV, EF]`	 <br>
<b>Output</b>: 카드를 택하면, 혜택이 바뀐다 <br>

<b>Keyword</b>: `안전, 항공` <br>
<b>POS</b>: `[MM, NNG, SF, MM, NNG, SF]` <br>
<b>Output</b>: 새로운 공항, 안전한 항공 <br>

<b>Keywords</b>: `추석, 선물` <br>
<b>POS</b>: `[NNG, JK, MM, NNG, SF, NNG, JK, MM, NNG]` <br>
<b>Output</b>: 추석을 앞둔 추억, 당신을 위한 선물 <br>

### Model Architecture
<br>
<img src="https://github.com/yeounyi/LexPOS/blob/main/assets/adj_graph.png" width=400>

### Pretrained Model
https://drive.google.com/drive/folders/1opkhDApURnjibVYmmhj5bqLTWy4miNe4?usp=sharing

### References
https://github.com/scarletcho/KoG2P

### Citation
```
@misc{yi2021lexpos,
  author = {Yi, Yeoun},
  title = {Generating Korean Slogans with Linguistic Constraints using Sequence-to-Sequence Transformer},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yeounyi/LexPOS_ko}}
}
```
