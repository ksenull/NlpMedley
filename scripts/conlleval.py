# 'PADDING_LABEL', 'I-PER', 'E-MISC', 'OUT', 'I-MISC', 'S-MISC',
# 'B-PER', 'B-LOC', 'B-MISC', 'B-ORG', 'S-ORG', 'S-LOC', 'E-ORG', 'S-PER', 'E-LOC', 'I-LOC', 'I-ORG', 'E-PER'
from collections import defaultdict
import unittest

class ConllEval:

    def __init__(self, tag2idx, idx2tag):
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

    def split(self, chunk_tag):
        if chunk_tag == self.tag2idx['OUT']:
            return None, 'OUT'
        if chunk_tag == self.tag2idx['PADDING_LABEL']:
            return None, 'PAD'
        return self.idx2tag[chunk_tag].split('-', maxsplit=1)

    def count_chunks(self, true_seqs, pred_seqs, padding_label):
        """
        true_seqs: a list of true tags
        pred_seqs: a list of predicted tags
        padding_label: an idx of padding

        return:
        correct_chunks: a dict (counter),
                        key = chunk types,
                        value = number of correctly identified chunks per type
        true_chunks:    a dict, number of true chunks per type
        pred_chunks:    a dict, number of identified chunks per type
        """
        correct_chunks = defaultdict(int)
        true_chunks = defaultdict(int)
        pred_chunks = defaultdict(int)

        prev_true_tag, prev_pred_tag = self.tag2idx['OUT'], self.tag2idx['OUT']
        correct_chunk = None
        stack_of_true = []
        stack_of_pred = []

        for true_tag, pred_tag in zip(true_seqs, pred_seqs):

            if true_tag == padding_label:
                continue

            true_prefix, true_type = self.split(true_tag)
            pred_prefix, pred_type = self.split(pred_tag)

            if pred_type == 'PAD':
                pred_chunks[pred_type] += 1

            if true_prefix == 'B':
                stack_of_true.append(true_type)
            if pred_prefix == 'B':
                stack_of_pred.append(pred_type)

            is_finished_true = true_prefix == 'E' and stack_of_true[-1] == true_type
            is_finished_pred = pred_prefix == 'E' and len(stack_of_pred) > 0 and stack_of_pred[-1] == pred_type

            if is_finished_true:
                stack_of_true.pop(-1)
                true_chunks[true_type] += 1
            if is_finished_pred:
                stack_of_pred.pop(-1)
                pred_chunks[pred_type] += 1
            if is_finished_true and is_finished_pred and true_type == pred_type:
                correct_chunks[true_type] += 1

            if true_type == 'OUT':
                for type in stack_of_true:
                    true_chunks[type] += 1
                stack_of_true = []
                true_chunks[true_type] += 1
            if pred_type == 'OUT':
                for type in stack_of_pred:
                    pred_chunks[type] += 1
                stack_of_pred = []
                pred_chunks[pred_type] += 1
            if true_type == 'OUT' and pred_type == 'OUT':
                correct_chunks[true_type] += 1

            if true_prefix == 'S':
                true_chunks[true_type] += 1
            if pred_prefix == 'S':
                pred_chunks[pred_type] += 1
            if true_prefix == 'S' and pred_prefix == 'S' and true_type == pred_type:
                correct_chunks[true_type] += 1
        return correct_chunks, true_chunks, pred_chunks

    @staticmethod
    def get_result(correct_chunks, true_chunks, pred_chunks):
        tp, tn, fp, fn = 0, 0, 0, 0
        chunk_types = set([t for t, _ in true_chunks] + [t for t, _ in pred_chunks])
        sum_correct_chunks = sum([count for _, count in correct_chunks.items()])
        for chunk_type in chunk_types:
            tp += correct_chunks[chunk_type]
            tn += (sum_correct_chunks - correct_chunks[chunk_type])
            fp += (pred_chunks[chunk_type] - correct_chunks[chunk_type])
            fn += (true_chunks[chunk_type] - correct_chunks[chunk_type])
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * prec * recall / (prec + recall) if tp != 0 else 0
        return acc, prec, recall, f1


class TestEvalMethods(unittest.TestCase):
    def setUp(self):
        self.tag2idx = {'PADDING_LABEL': 0, 'S-ORG': 1, 'S-PER': 2, 'E-MISC': 3, 'E-ORG': 4, 'E-LOC': 5, 'I-LOC': 6,
                   'B-PER': 7, 'E-PER': 8, 'OUT': 9, 'B-ORG': 10, 'S-MISC': 11, 'B-MISC': 12, 'B-LOC': 13, 'I-ORG': 14,
                   'S-LOC': 15, 'I-MISC': 16, 'I-PER': 17}
        self.idx2tag = ['PADDING_LABEL', 'S-ORG', 'S-PER', 'E-MISC', 'E-ORG', 'E-LOC', 'I-LOC', 'B-PER', 'E-PER', 'OUT',
                   'B-ORG', 'S-MISC', 'B-MISC', 'B-LOC', 'I-ORG', 'S-LOC', 'I-MISC', 'I-PER']
        self.eval = ConllEval(self.tag2idx, self.idx2tag)

    def test_count_chunks1(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT'], self.tag2idx['PADDING_LABEL']]
        correct_answer = ([('PER', 1), ('OUT', 1)], [('PER', 1), ('OUT', 2)], [('PER', 1), ('OUT', 1), ('PAD', 1)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks2(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['OUT'], self.tag2idx['OUT'], self.tag2idx['PADDING_LABEL']]
        correct_answer = ([('OUT', 1)], [('PER', 1), ('OUT', 2)], [('PER', 1), ('OUT', 2), ('PAD', 1)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks3(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['I-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['I-PER'], self.tag2idx['E-PER'], self.tag2idx['PADDING_LABEL']]
        correct_answer = ([('PER', 1)], [('PER', 1), ('OUT', 1)], [('PER', 1), ('PAD', 1)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks4(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['I-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['I-PER'], self.tag2idx['OUT'], self.tag2idx['OUT']]
        correct_answer = ([('OUT', 1)], [('PER', 1), ('OUT', 1)], [('PER', 1), ('OUT', 2)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks5(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['I-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['OUT'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        correct_answer = ([('OUT', 1)], [('PER', 1), ('OUT', 1)], [('PER', 1), ('OUT', 2)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks6(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['S-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['S-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        correct_answer = ([('PER', 2), ('OUT', 1)], [('PER', 2), ('OUT', 1)], [('PER', 2), ('OUT', 1)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks7(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['S-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['S-PER'], self.tag2idx['OUT'], self.tag2idx['OUT']]
        correct_answer = ([('PER', 1), ('OUT', 1)], [('PER', 2), ('OUT', 1)], [('PER', 2), ('OUT', 2)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks8(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['S-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['OUT'], self.tag2idx['S-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        correct_answer = ([('PER', 1), ('OUT', 1)], [('PER', 2), ('OUT', 1)], [('OUT', 2), ('PER', 1)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

    def test_count_chanks9(self):
        true_seq = [self.tag2idx['B-PER'], self.tag2idx['S-PER'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        pred_seq = [self.tag2idx['B-PER'], self.tag2idx['OUT'], self.tag2idx['E-PER'], self.tag2idx['OUT']]
        correct_answer = ([('OUT', 1)], [('PER', 2), ('OUT', 1)], [('PER', 1), ('OUT', 2)])
        res = self.eval.count_chunks(true_seq, pred_seq, self.tag2idx['PADDING_LABEL'])
        res = tuple(list(chunks.items()) for chunks in res)
        # print(res)
        self.assertEqual(res, correct_answer)

if __name__ == '__main__':
    unittest.main()
