from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
import sys

class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        # self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)
        self.bestPerformance = {topn: [-1, {}] for topn in self.topN} 
        self.file_name = conf['model.name'] + '@' + conf[conf['model.name']] + '.txt'
        self.out_dir = self.output['-dir']

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measures = ranking_evaluation(self.data.test_set, rec_list, self.topN)

        for i, topN in enumerate(self.topN):    
            performance = {}
            for m in measures[i][1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)

            # Check and update the best performance
            if self.bestPerformance[topN][0] == -1 or self.bestPerformance[topN][1]['Recall'] < performance['Recall']:
                self.bestPerformance[topN][0] = epoch + 1
                self.bestPerformance[topN][1] = performance
                self.save()

        for i, topN in enumerate(self.topN):
            print('-' * 120)    
            print(f"Real-Time Ranking Performance (Top-{topN} Item Recommendation)")
            measure = [m.strip() for m in measures[i][1:]]
            print("*Current Performance*")
            print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
            bp = ''
            bp += 'Recall' + ':' + str(self.bestPerformance[topN][1]['Recall']) + '  |  '
            bp += 'NDCG' + ':' + str(self.bestPerformance[topN][1]['NDCG'])
            print('*Best Performance* ')
            print('Epoch:', str(self.bestPerformance[topN][0]) + ',', bp)
        print('-' * 120)
        
        return measures
    
    def evaluate(self, rec_list):
        print('The result has been output to ', abspath(self.out_dir), '.')
        measures = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.result = []
        for measure in measures:
            for indicator in measure:
                self.result.append(indicator)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(self.out_dir, self.file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))
        
    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list