import PIL
import pytest
import sys
sys.path.insert(0, '..')


from src import uciego

def test_loadimage():
    raw_img = uciego.load_image('../data/UCI-EGO/Seq1/fr101.jpg')
    assert type(raw_img) == PIL.Image.Image
