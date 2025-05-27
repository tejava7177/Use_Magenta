from music21 import converter

def extract_chord_sequence(midi_file):
    score = converter.parse(midi_file)
    chords = score.chordify()
    return [c.pitchedCommonName for c in chords.recurse().getElementsByClass('Chord')]