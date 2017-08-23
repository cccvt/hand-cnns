from src.datasets.gteagazeplus import GTEAGazePlus
gtea = GTEAGazePlus()
# Repeats
all_seqs = gtea.all_seqs
repeats = {}
for person in all_seqs:
    repeats[person] =  gtea.get_repeat_classes(seqs=[person])
filtered = {}
for person, classes in repeats.items():
    filtered[person] = [label for label in classes if label in gtea.cvpr_labels]

inter = set(repeats['Ahmad'])
for person in ['Alireza', 'Carlos', 'Rahul']:
    inter = inter.intersection(set(repeats[person]))

delta = [label for label in inter if label not in gtea.cvpr_labels]
import pdb; pdb.set_trace()


