import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class mT5:
    def __init__(self):
        source = "csebuetnlp/mT5_multilingual_XLSum"
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source)
    
    @staticmethod
    def whitespace_handler(k):
        return re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))

    def run(self, text):
        input_ids = self.tokenizer([self.whitespace_handler(text)], return_tensors="pt", padding="max_length", truncation=True, max_length=512)["input_ids"]
        output_ids = self.model.generate(input_ids=input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return summary

XLSUM_LANGS = [
    ["English", 301444],
    ["Ukrainian", 57952],
    ["Russian", 52712],
    ["Hindi", 51715],
    ["Spanish", 44413],
    ["Indonesian", 44170],
    ["Urdu", 40714],
    ["Arabic", 40327],
    ["Chinese", 39810],
    ["Turkish", 29510],
    ["Persian", 25783],
    ["Portuguese", 23521],
    ["Vietnamese", 23468],
    ["Tamil", 17846],
    ["Pashto", 15274],
    ["Welsh", 11596],
    ["Telugu", 11308],
    ["Marathi", 11164],
    ["Swahili", 10005],
    ["Pidgina", 9715],
    ["Gujarati", 9665],
    ["French", 9100],
    ["Punjabi", 8678],
    ["Bengali", 8226],
    ["Japanese", 7585],
    ["Azerbaijani", 7332],
    ["Serbian (Cyrillic)", 7317],
    ["Serbian (Latin)", 7263],
    ["Thai", 6928],
    ["Yoruba", 6316],
    ["Hausa", 6313],
    ["Oromo", 5738],
    ["Somali", 5636],
    ["Kirundi", 5558],
    ["Amharic", 5461],
    ["Nepali", 5286],
    ["Burmese", 5002],
    ["Uzbek", 4944],
    ["Tigrinya", 4827],
    ["Igbo", 4559],
    ["Korean", 4281],
    ["Sinhala", 3414],
    ["Kyrgyz", 2315],
    ["Scottish (Gaelic)", 1101],
]
