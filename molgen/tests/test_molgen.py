import os

from django.core.management import call_command
from django.test import TestCase
from django.utils.six import StringIO

from pgmols.models import Mol

MY_DIR = os.path.dirname(__file__)
FRAGMENTS_FILE = 'fragments_small.json'

PROJECT_NAME = "test_project"
CONFGEN_NAME = "pretend_confgen"

class CommandTest(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_build_dir(self):
        out = StringIO()
        self.assertEquals(0, Mol.objects.count())
        call_command('molgen',
                     PROJECT_NAME,
                     os.path.join(MY_DIR, FRAGMENTS_FILE),
                     '--recipe=test',
                     stdout=out)
        self.assertEquals(9, Mol.objects.count())

class SymmetricTerminalSpacerCoreTestCase(TestCase):
    def test_symmetric_terminal_spacer_core_recipe(self):
        fragments_json_path = os.path.join(
            MY_DIR, 'symmetric_terminal_spacer_core_fragments.json')
        out = StringIO()
        self.assertEquals(0, Mol.objects.count())
        call_command('molgen',
                     PROJECT_NAME,
                     fragments_json_path,
                     '--recipe=symmetric_terminal_spacer_core',
                     stdout=out)
        self.assertEquals(12, Mol.objects.count())

class SpiroRecipeTestCase(TestCase):
    def test_spiro_recipe(self):
        fragments_json_path = os.path.join(
            MY_DIR, 'spiro_fragments.json')
        out = StringIO()
        self.assertEquals(0, Mol.objects.count())
        call_command('molgen',
                     PROJECT_NAME,
                     fragments_json_path,
                     '--recipe=spiro',
                     stdout=out)
        self.assertEquals(16, Mol.objects.count())



