import re
import os
from symspellpy.symspellpy import SymSpell, Verbosity
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

# ====================================
# 1. Utility: Load slang.txt ke dictionary
# ====================================
def load_slang_dict(path):
    slang_map = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if '`' in line:
                parts = line.strip().split('`')
                slang = parts[0].strip().lower()
                expansion = parts[1].split('|')[0].strip().lower()
                slang_map[slang] = expansion
    return slang_map

# ====================================
# 2. Utility: Bersihkan elongated
# ====================================
def normalize_elongated(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

# ====================================
# 3. Ganti slang berdasarkan kamus
# ====================================
def replace_slang(text, slang_dict):
    words = text.split()
    return " ".join([slang_dict.get(w.lower(), w) for w in words])

# ====================================
# 4. Koreksi typo dengan SymSpell
# ====================================
def correct_spelling_symspell(text, symspell):
    words = text.split()
    corrected = []
    for word in words:
        suggestions = symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected.append(suggestions[0].term)
        else:
            corrected.append(word)
    return " ".join(corrected)

# ====================================
# 5. Inisialisasi SymSpell
# ====================================
symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dict_path = "dict/frequency_dictionary_en_100000.txt"
if not symspell.load_dictionary(dict_path, term_index=0, count_index=1):
    print("❌ Gagal memuat dictionary.")
    exit()

# ====================================
# 6. Load slang.txt
# ====================================
slang_dict = load_slang_dict("dict/slang.txt")

# ====================================
# 7. Ekphrasis config
# ====================================
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'date', 'number'],
    annotate=set(),
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer().tokenize,
    dicts=[emoticons]
)

# ====================================
# 8. Contoh input
# ====================================
text = "any bug yet"

# ====================================
# 9. Pipeline Cleaning
# ====================================
processed = text_processor.pre_process_doc(text)
joined = " ".join(processed)
elong_fixed = normalize_elongated(joined)
slang_fixed = replace_slang(elong_fixed, slang_dict)
final_text = correct_spelling_symspell(slang_fixed, symspell)

# ====================================
# 10. Output
# ====================================
print("✅ Clean Output:\n", final_text)
