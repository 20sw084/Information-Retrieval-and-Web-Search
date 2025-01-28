import sys
import os
from collections import defaultdict
import math

def trainBigramLanguageModel(training_text):
    unigram_freq = defaultdict(int)
    bigram_freq = defaultdict(int)
    
    lines = training_text.splitlines()
    for line in lines:
        line = "<start>" + line  # Add start token
        for i in range(len(line)):
            unigram_freq[line[i]] += 1  # Count unigrams
            if i > 0:
                bigram_freq[(line[i-1], line[i])] += 1  # Count bigrams
    
    return unigram_freq, bigram_freq

def identifyLanguage(test_text, languages, unigram_dicts, bigram_dicts):
    max_prob = float('-inf')
    best_language = None
    vocab_size = len(set(char for d in unigram_dicts for char in d.keys()))  # Combined vocabulary
    
    for i, lang in enumerate(languages):
        unigram_freq = unigram_dicts[i]
        bigram_freq = bigram_dicts[i]
        
        # Calculate log-probability with add-one smoothing
        log_prob = 0
        test_text = "<start>" + test_text  # Add start token
        for j in range(1, len(test_text)):
            w1 = test_text[j-1]
            w2 = test_text[j]
            bigram_count = bigram_freq.get((w1, w2), 0) + 1
            unigram_count = unigram_freq.get(w1, 0) + vocab_size
            log_prob += math.log(bigram_count / unigram_count)
        
        if log_prob > max_prob:
            max_prob = log_prob
            best_language = lang
    
    return best_language

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 languageIdentification.py [training folder] [test file]")
        return
    
    training_folder = sys.argv[1]
    test_file = sys.argv[2]
    
    # Load training data
    languages = ['English', 'French', 'Italian']
    unigram_dicts = []
    bigram_dicts = []
    
    for lang in languages:
        with open(os.path.join(training_folder, lang), 'r', encoding='latin-1') as f:
            training_text = f.read()
        unigram, bigram = trainBigramLanguageModel(training_text)
        unigram_dicts.append(unigram)
        bigram_dicts.append(bigram)
    
    # Process test data
    output_lines = []
    with open(test_file, 'r', encoding='latin-1') as test_f:
        for i, line in enumerate(test_f, start=1):
            line = line.strip()
            predicted_language = identifyLanguage(line, languages, unigram_dicts, bigram_dicts)
            output_lines.append(f"{i} {predicted_language}")
    
    # Write results to output file
    with open("languageIdentification.output", "w", encoding='latin-1') as output_f:
        output_f.write("\n".join(output_lines))
    
    print("Output written to languageIdentification.output")
    # Example usage in main
    output_file = "languageIdentification.output"
    solution_file = "languageIdentification.data/solution"
    answers_file = "languageIdentification.answers"
    calculate_accuracy(output_file, solution_file, answers_file)

def calculate_accuracy(output_file, solution_file, answers_file):
    with open(output_file, "r") as out, open(solution_file, "r") as sol:
        output_lines = out.readlines()
        solution_lines = sol.readlines()

    correct = 0
    total = len(solution_lines)

    for out_line, sol_line in zip(output_lines, solution_lines):
        if out_line.strip().split()[1] == sol_line.strip().split()[1]:
            correct += 1

    accuracy = (correct / total) * 100

    # Write the accuracy to the answers file
    with open(answers_file, "w") as ans:
        ans.write(f"Accuracy of the language identifier: {accuracy:.2f}%\n")


if __name__ == "__main__":
    main()
