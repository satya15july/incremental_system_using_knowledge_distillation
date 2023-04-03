select_classes_2_1 = ['apple', 'aquarium_fish']
select_classes_2_2 = ['baby', 'bear']


select_classes_1_10 = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle']
select_classes_10_16 = ['bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel']

select_classes_1_15 = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly']
select_classes_15_20 = ['camel', 'can', 'castle', 'caterpillar', 'cattle']
select_classes_10_20 = ['bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle']
select_classes_20_30 = ['chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur']
select_classes_30_40 = ['dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard']
select_classes_40_50 = ['lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain']
select_classes_50_60 = ['mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree']
select_classes_60_70 = ['plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket' ]
select_classes_70_80 = ['rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider']
select_classes_80_90 = ['squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table','tank', 'telephone', 'television', 'tiger', 'tractor']
select_classes_90_100 = ['train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

BATCH_SIZE = 128
EPOCH = 10000
SAVE_INTERVAL = 2000
RESUME = True

OLD_CLASSES = 10
NEW_CLASSES = 6
TOTAL_CLASSES = OLD_CLASSES + NEW_CLASSES
TEMPRATURE = 3

LOGGING = False

MODEL_PATH = 'models'



