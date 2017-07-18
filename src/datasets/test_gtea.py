from src.datasets.gtea import GTEA


def test_filenames():
    gtea_dataset = GTEA(seqs=['S1'])
    assert len(gtea_dataset) == 9110


def test_all_filenames():
    gtea_s1 = GTEA(seqs=['S1'])
    gtea_s2 = GTEA(seqs=['S2'])
    gtea_s3 = GTEA(seqs=['S3'])
    gtea_s4 = GTEA(seqs=['S4'])
    gtea_all = GTEA(seqs=['S1', 'S2', 'S3', 'S4'])
    assert len(gtea_all) == len(gtea_s1) + len(gtea_s2) +\
        len(gtea_s3) + len(gtea_s4)

