from bleu.cluster.doCluster import *

# 对真实日志集进行处理
if __name__ == '__main__':
    doRealLogCluster('bleu/data/csv/Help Desk.csv', 'Case ID', 'Activity')
    doRealLogCluster('bleu/data/csv/Hospital Billing - Event Log.csv', 'case', 'event')
    doRealLogCluster('bleu/data/csv/Sepsis Cases - Event Log.csv', 'case', 'event')
    doRealLogCluster('bleu/data/csv/BPI Challenge 2017 - Offer log.csv', 'case', 'event')
    doRealLogCluster('bleu/data/csv/BPI_Challenge_2020 - DomesticDeclarations.csv', 'case', 'event')
