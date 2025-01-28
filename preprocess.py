import os
import sys
import re
from collections import defaultdict, Counter

def removeSGML(text):
    return re.sub(r'<[^>]+>', '', text)

def tokenizeText(text):
    text = re.sub(r"(\w+)'s", r"\1 's", text)
    text = re.sub(r"(\w+)'m", r"\1 am", text)
    text = re.sub(r"(\w+)'re", r"\1 are", text)
    text = re.sub(r"(\w+)'t", r"\1 not", text)
    text = re.sub(r"(\w+)'ll", r"\1 will", text)
    text = re.sub(r"(\w+)'ve", r"\1 have", text)
    text = re.sub(r"(\w+)'d", r"\1 would", text)
    
    tokens = re.findall(r'\b\w+\.?\w*\b', text)
    return tokens

def bpe(tokens, vocabSize):
    vocab = Counter()
    for token in tokens:
        vocab.update(list(token))
    
    merge_rules = []
    
    while len(vocab) < vocabSize:
        pair_freq = defaultdict(int)
        for token in tokens:
            chars = list(token)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freq[pair] += 1
        
        if not pair_freq:
            break
        most_freq_pair = max(pair_freq, key=pair_freq.get)
        
        new_token = ''.join(most_freq_pair)
        merge_rules.append((most_freq_pair, new_token))
        
        new_tokens = []
        for token in tokens:
            new_token_str = token.replace(''.join(most_freq_pair), new_token)
            new_tokens.append(new_token_str)
        tokens = new_tokens
        
        vocab.update([new_token])
    
    return tokens, merge_rules

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 preprocess.py <input_folder> <vocabSize>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    vocabSize = int(sys.argv[2])
    
    all_tokens = []
    for filename in os.listdir(input_folder):
        with open(os.path.join(input_folder, filename), 'r', encoding='ISO-8859-1') as file:
            text = file.read()
            text = removeSGML(text)
            tokens = tokenizeText(text)
            all_tokens.extend(tokens)
    
    bpe_tokens, merge_rules = bpe(all_tokens, vocabSize)
    
    with open('preprocess.output', 'w') as f:
        f.write(f"Tokens: {len(bpe_tokens)}\n")
        f.write(f"Merge rules: {len(merge_rules)}\n")
        f.write("The first 20 merge rules:\n")
        for rule in merge_rules[:20]:
            f.write(f"{rule[0]} -> {rule[1]}\n")
        
        token_freq = Counter(bpe_tokens)
        top_50 = token_freq.most_common(50)
        f.write("Top 50 tokens:\n")
        for token, freq in top_50:
            f.write(f"{token} {freq}\n")
    
    total_tokens = len(bpe_tokens)
    total_merge_rules = len(merge_rules)
    
    sorted_tokens = sorted(token_freq.items(), key=lambda x: -x[1])
    cumulative_freq = 0
    min_tokens = 0
    for token, freq in sorted_tokens:
        cumulative_freq += freq
        min_tokens += 1
        if cumulative_freq >= 0.25 * total_tokens:
            break
    
    with open('preprocess.answers', 'w') as f:
        f.write(f"Total number of BPE tokens: {total_tokens}\n")
        f.write(f"Total number of merge rules: {total_merge_rules}\n")
        f.write(f"Minimum number of unique BPE tokens accounting for 25% of total tokens: {min_tokens}\n")

if __name__ == "__main__":
    main()