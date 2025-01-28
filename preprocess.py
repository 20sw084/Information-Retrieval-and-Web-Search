import os
import re
import sys
from collections import Counter, defaultdict

# a. Function to remove SGML tags
def removeSGML(text):
    return re.sub(r"<[^>]+>", "", text)

# b. Function to tokenize text
def tokenizeText(text):
    # Replace contractions and handle possessives
    text = re.sub(r"\bI'm\b", "I am", text)
    text = re.sub(r"\b(\w+)'s\b", r"\1 's", text)
    text = re.sub(r"\b(\w+)'re\b", r"\1 are", text)
    text = re.sub(r"\b(\w+)'ve\b", r"\1 have", text)
    text = re.sub(r"\b(\w+)'ll\b", r"\1 will", text)
    text = re.sub(r"\b(\w+)'d\b", r"\1 had", text)
    text = re.sub(r"\b(\w+)n't\b", r"\1 not", text)

    # Keep dates and hyphenated words intact
    text = re.sub(r"(\d+/\d+/\d+)", r" \1 ", text)
    text = re.sub(r"(\w+-\w+)", r" \1 ", text)

    # Remove punctuation, but keep periods in abbreviations/numbers
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*", text)
    return tokens

# c. Function to perform Byte-Pair Encoding
def BPE(tokens, vocabSize):
    # Initial vocabulary of single characters
    vocab = Counter("".join(tokens))
    merge_rules = []
    
    # Convert tokens to subword sequences
    subwords = [list(token) + ["</w>"] for token in tokens]
    
    # Main loop: merge pairs until vocabSize is reached
    while len(vocab) < vocabSize:
        # Calculate pair frequencies
        pairs = Counter()
        for subword in subwords:
            for i in range(len(subword) - 1):
                pairs[(subword[i], subword[i + 1])] += 1
        
        # Find the most common pair
        if not pairs:
            break
        most_common = max(pairs, key=pairs.get)
        
        # Merge the most common pair in subwords
        merge_rules.append(most_common)
        new_vocab = []
        for subword in subwords:
            new_subword = []
            i = 0
            while i < len(subword):
                if i < len(subword) - 1 and (subword[i], subword[i + 1]) == most_common:
                    new_subword.append(subword[i] + subword[i + 1])
                    i += 2
                else:
                    new_subword.append(subword[i])
                    i += 1
            new_vocab.append(new_subword)
        subwords = new_vocab
    
        # Update vocabulary
        vocab = Counter([item for sublist in subwords for item in sublist])
    
    return subwords, merge_rules

# d. Main function
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 preprocess.py <folder> <vocabSize>")
        sys.exit(1)
    
    folder = sys.argv[1]
    vocabSize = int(sys.argv[2])
    
    # Step I: Read files from the folder
    tokens = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, encoding="ISO-8859-1") as f:
            text = f.read()
        
        # Step II: Apply removeSGML and tokenizeText
        text = removeSGML(text)
        tokens.extend(tokenizeText(text))
    
    # Step III: Apply BPE
    subwords, merge_rules = BPE(tokens, vocabSize)
    
    # Step IV: Generate output
    total_tokens = sum(len(subword) for subword in subwords)
    most_frequent_tokens = Counter([item for sublist in subwords for item in sublist]).most_common(50)
    
    # Write output to preprocess.output
    with open("preprocess.output", "w") as f:
        f.write(f"Tokens: {total_tokens}\n")
        f.write(f"Merge rules: {len(merge_rules)}\n")
        f.write("The first 20 merge rules:\n")
        for rule in merge_rules[:20]:
            f.write(f"({rule[0]}, {rule[1]}) -> {rule[0]}{rule[1]}\n")
        f.write("Top 50 tokens:\n")
        for token, freq in most_frequent_tokens:
            f.write(f"{token} [{freq}]\n")

if __name__ == "__main__":
    main()
