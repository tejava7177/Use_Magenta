from music21 import converter, chord, harmony

def extract_chord_sequence(midi_file):
    score = converter.parse(midi_file)
    chords = score.chordify()

    chord_names = []
    for c in chords.recurse().getElementsByClass('Chord'):
        try:
            csym = harmony.chordSymbolFromChord(c)
            if csym.figure and "Chord Symbol Cannot Be Identified" not in csym.figure:
                chord_names.append(csym.figure)
        except:
            continue

    return chord_names