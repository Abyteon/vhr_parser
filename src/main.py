from vhr_parser import FileParser, DbcParser
from .config import msgid_filter

dbc_parser = DbcParser("path/to/dbc/file.dbc")

parser = FileParser("../data/input/", "../data/output/", dbc_parser)
parser.process_directory()
#
print("hello world")
