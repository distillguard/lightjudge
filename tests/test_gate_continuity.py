import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import gate_continuity as gate

class ContinuityDataTests(unittest.TestCase):
    def test_hellaswag_uses_unique_item_id_and_original_endings_order(self):
        raw = dict(ind=4, ctx='A person opens a door.', endings=['a','b','c','d'], label='3',
                   source_id='same-scene', activity_label='Opening doors', split_type='indomain')
        first = gate.normalize('hellaswag', raw, 'train')
        second = gate.normalize('hellaswag', {**raw, 'ind':5}, 'train')
        self.assertNotEqual(first['source_id'], second['source_id'])
        self.assertEqual(first['gold'], 'D')
        self.assertEqual([c['text'] for c in first['choices']], raw['endings'])
        self.assertEqual(first['scene_id'], second['scene_id'])
        other_scene = gate.normalize('hellaswag', {**raw, 'source_id':'different-scene'}, 'train')
        self.assertNotEqual(first['source_id'], other_scene['source_id'])
        self.assertIn('multiple-choice question', gate.screen_prompt(first))

    def test_hidden_or_invalid_hellaswag_labels_are_rejected(self):
        raw = dict(ind=4, ctx='context', endings=['a','b','c','d'], label='', source_id='scene',
                   activity_label='activity', split_type='indomain')
        for label in ('', '-1', '4', 'A'):
            with self.assertRaises(ValueError):
                gate.normalize('hellaswag', {**raw, 'label':label}, 'train')

    def test_openbook_noncanonical_source_label_maps_by_position(self):
        raw = {'id':'one', 'question_stem':'Question?', 'choices':{'label':['1','2','3','4'],
               'text':['first','second','third','fourth']}, 'answerKey':'2'}
        result = gate.normalize('openbookqa', raw, 'validation')
        self.assertEqual(result['gold'], 'B')
        self.assertEqual(result['choices'][1]['source_label'], '2')
        with self.assertRaises(ValueError):
            gate.normalize('openbookqa', {**raw, 'answerKey':'missing'}, 'validation')

    def test_nonfinite_diagnostics_are_rejected(self):
        for values in ([], [1,float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                gate.describe(values)

if __name__ == '__main__':
    unittest.main()
