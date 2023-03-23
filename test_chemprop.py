import os
from chemprop.predict_one import predict_one

if __name__ == '__main__':
    print(os.getcwd())
    print(predict_one('models/all_db_checkpoints', [['CC']])) 
    print(predict_one('models/weights_lite', [['CC(CCCCCCCCO)c1cscc1-c1cnc2c(n1)-c1nccnc1CC2']]))
    print(predict_one('models/weights_lite', [['Cc1cncc(C)c1']]))
