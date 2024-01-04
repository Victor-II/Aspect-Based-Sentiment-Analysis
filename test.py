from dataset.dataset import ACSA_DS

if __name__ == '__main__':
    path = '/home/victor-ii/Desktop/Research_ABSA/Task_1/data/mams/mams_acsa/val.xml'
    ds = ACSA_DS(path, 70)
    print(ds[0])