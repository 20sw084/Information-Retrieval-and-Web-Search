import os
import sys
from collections import defaultdict

# Function to train a bigram language model
def trainBigramLanguageModel(training_text):
    unigram_freq = defaultdict(int)
    bigram_freq = defaultdict(int)
    
    # Add <start> token at the beginning of each line
    lines = training_text.split('\n')
    for line in lines:
        if not line:
            continue
        # Prepend <start> token
        line = "<start>" + line
        # Calculate unigram frequencies
        for char in line:
            unigram_freq[char] += 1
        # Calculate bigram frequencies
        for i in range(len(line) - 1):
            bigram = line[i:i+2]
            bigram_freq[bigram] += 1
    
    return unigram_freq, bigram_freq

# Function to determine the language of a string
def identifyLanguage(test_text, languages, unigram_freqs, bigram_freqs):
    max_prob = -float('inf')
    best_language = None
    
    for i, language in enumerate(languages):
        # Get unigram and bigram frequencies for the current language
        unigram_freq = unigram_freqs[i]
        bigram_freq = bigram_freqs[i]
        
        # Calculate the total number of characters in the training data for this language
        total_chars = sum(unigram_freq.values())
        
        # Add-one smoothing: vocabulary size is the number of unique characters
        vocab_size = len(unigram_freq)
        
        # Initialize probability with log(1) = 0
        prob = 0.0
        
        # Prepend <start> token to the test text
        test_text_with_start = "<start>" + test_text
        
        # Calculate bigram probability with add-one smoothing
        for j in range(len(test_text_with_start) - 1):
            bigram = test_text_with_start[j:j+2]
            # Numerator: count of bigram + 1
            numerator = bigram_freq.get(bigram, 0) + 1
            # Denominator: count of first character in bigram + vocab_size
            denominator = unigram_freq.get(test_text_with_start[j], 0) + vocab_size
            # Avoid division by zero
            if denominator == 0:
                prob += -float('inf')
            else:
                prob += (numerator / denominator)
        
        # Update the best language if the current probability is higher
        if prob > max_prob:
            max_prob = prob
            best_language = language
    
    return best_language

# Main program
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 languageIdentification.py [path to training data folder] [test file]")
        sys.exit(1)
    
    training_folder = sys.argv[1]
    test_file = sys.argv[2]
    
    # Languages to consider
    languages = ["English", "French", "Italian"]
    
    # Train bigram models for each language
    unigram_freqs = []
    bigram_freqs = []
    for language in languages:
        training_file = os.path.join(training_folder, f"{language}.txt")
        with open(training_file, 'r', encoding='ISO-8859-1') as file:  # Use ISO-8859-1 encoding
            training_text = file.read()
            unigram_freq, bigram_freq = trainBigramLanguageModel(training_text)
            unigram_freqs.append(unigram_freq)
            bigram_freqs.append(bigram_freq)
    
    # Open the test file and identify the language for each line
    predictions = []
    with open(test_file, 'r', encoding='ISO-8859-1') as file:  # Use ISO-8859-1 encoding
        for line in file:
            line = line.strip()
            if not line:
                continue
            language = identifyLanguage(line, languages, unigram_freqs, bigram_freqs)
            predictions.append(language)
    
    # Write predictions to languageIdentification.output
    with open('languageIdentification.output', 'w') as f:
        for i, language in enumerate(predictions):
            f.write(f"{i+1} {language}\n")
    
    # Calculate accuracy using the solution file
    solution_file = os.path.join(os.path.dirname(test_file), "solution")
    correct = 0
    total = 0
    with open(solution_file, 'r', encoding='ISO-8859-1') as file:  # Use ISO-8859-1 encoding
        for i, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            true_language = line
            predicted_language = predictions[i]
            if true_language == predicted_language:
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100
    
    # Write accuracy to languageIdentification.answers
    with open('languageIdentification.answers', 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    main()