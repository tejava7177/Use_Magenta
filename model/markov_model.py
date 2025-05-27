from collections import defaultdict
import random

class MarkovChordModel:
    def __init__(self):
        self.transitions = defaultdict(list)

    def train(self, sequences):
        for seq in sequences:
            for i in range(len(seq)-1):
                self.transitions[seq[i]].append(seq[i+1])

    def generate(self, start, length=8):
        output = [start]
        for _ in range(length - 1):
            current = output[-1]
            next_chord = random.choice(self.transitions.get(current, [current]))
            output.append(next_chord)
        return output