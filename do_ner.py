from pyltp import Segmentor, Postagger, NamedEntityRecognizer
from os.path import join
import pycorenlp


LTP_DATA = "/home/hanzhao/ltp_data_v3.4.0/"
text = "中国国家主席习近平和美国总统特朗普"
seg_model_f = join(LTP_DATA, 'cws.model')
pos_model_f = join(LTP_DATA, 'pos.model')
ner_model_f = join(LTP_DATA, 'ner.model')
seg = Segmentor()
pos = Postagger()
ner = NamedEntityRecognizer()
seg.load(seg_model_f)
pos.load(pos_model_f)
ner.load(ner_model_f)

words = seg.segment(text)
tag = pos.postag(words)
nne = ner.recognize(words, tag)
print(list(words), list(tag))
print('\t'.join(nne))

seg.release()
ner.release()
