import sys
import os
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent / 'PPBO' / 'src'
sys.path.append(str(module_path.resolve()))

os.chdir(module_path)

from gp_model import GPModel
from ppbo_settings import PPBO_settings
from acquisition import next_query

os.chdir(current_path)