'''
=========================================================================
Darth Vader Sentiment Analysis

Simple sentiment analysis alogrithim that parses the text for the movie,
searches for the lines by Darth Vader, and then uses the NRC Lexicon to
provide a sentiment analysis of the words.
=========================================================================
'''
import re
# files will need to be in same directory or paths updated
nrc_lexicon = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
star_wars_4 = 'sw4.txt'
star_wars_5 = 'sw5.txt'
star_wars_6 = 'sw6.txt'

def create_lexicon(): # parses lexicon file and creates lookup table
    f = open(nrc_lexicon, 'r')
    lex = {}

    for line in f:
        curr = line.split("\t")
        if len(curr) == 3:
            if curr[0] not in lex.keys():
                if curr[2] == '1\n':
                    lex[curr[0]] = {curr[1]:1}
                else:
                    lex[curr[0]] = {}
            else:
                if curr[2] == '1\n':
                    lex[curr[0]].update({curr[1]:1})
    return lex

def parse_wars(lex, file): # parses starwars script and returns analysis
    score = {}
    f = open(file, 'r')
    for line in f:
        line = line.rstrip('\n')
        curr = line.split('\t')
        if curr[1] == "VADER": # analyze only Darth Vader lines
            analyze = re.sub(r"[^\w\s]", "", curr[2]) # removing punctuation
            analyze = analyze.split(" ")
            for word in analyze:
                word = word.lower() # lowercase
                if word in lex.keys():
                    score = scorer(word, score) # original
                elif word[:1] in lex.keys():
                    score = scorer(word[:1], score) # +s
                elif word[:2] in lex.keys():
                    score = scorer(word[:2], score) # +es/ed
    return dict(sorted(score.items()))

def scorer(word, score_dict): # updates score dictionary for new word
    vals = lex.get(word)
    for sentiment in vals.keys():
        if sentiment in score_dict.keys():
            score_dict[sentiment] += 1
        else:
            score_dict[sentiment] = 1
    return score_dict

def print_dict(dictionary): # print output as table
    print("-----------------------")
    print("{:<15}{:<5}".format("Sentiment","Score"))
    print("-----------------------")
    for key, val in dictionary.items():
       print("{:<15}{:<5}".format(key,val))
    print("-----------------------")

if __name__ == '__main__':  # run
    lex = create_lexicon()
    print("Star Wars 4:")
    print_dict(parse_wars(lex, star_wars_4))
    print("\n\nStar Wars 5:")
    print_dict(parse_wars(lex, star_wars_5))
    print("\n\nStar Wars 6:")
    print_dict(parse_wars(lex, star_wars_6))
