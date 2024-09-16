from transformers import MarianMTModel, MarianTokenizer

class MultilingualTranslator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def load_model(self, src_lang, tgt_lang):
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        self.models[(src_lang, tgt_lang)] = MarianMTModel.from_pretrained(model_name)
        self.tokenizers[(src_lang, tgt_lang)] = MarianTokenizer.from_pretrained(model_name)
    
    def translate(self, text, src_lang, tgt_lang):
        if (src_lang, tgt_lang) not in self.models:
            self.load_model(src_lang, tgt_lang)
        
        model = self.models[(src_lang, tgt_lang)]
        tokenizer = self.tokenizers[(src_lang, tgt_lang)]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

# Usage
translator = MultilingualTranslator()
print(translator.translate("Hello, how are you?", "en", "fr"))
print(translator.translate("Bonjour, comment allez-vous?", "fr", "de"))