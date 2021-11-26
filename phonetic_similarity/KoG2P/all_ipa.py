from gensim.models import Word2Vec
from g2p import runKoG2P
import pandas as pd

model = Word2Vec.load(model_name)


words = [w for w in model.wv.vocab]
ipas = [runKoG2P(w, '/home/yeoun/KO_BART/rulebook.txt') for w in words]

assert len(words) == len(ipas)

df = pd.DataFrame(columns=['word', 'ipa'])
df['word'] = words
df['ipa'] = ipas

df.to_csv('ipa_dict.csv', index=False, encoding='utf-8')