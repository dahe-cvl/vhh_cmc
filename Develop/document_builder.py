import os
import sphinx.cmd.make_mode as sphinx_build

#######################################################
# HINT:
# to run this script successfully tex compiler
# as well as perl must be installed on your system
#######################################################

OUT_DIR = "../ApiSphinxDocumentation/source/"  # enter path to source directory (including conf.py etc.)
build_output = os.path.join("../ApiSphinxDocumentation/", "build/") # enter path to build directory (output path)

# build HTML (same as `make html`)
build_html_args = ["html", OUT_DIR, build_output]
sphinx_build.run_make_mode(args=build_html_args)

# build PDF latex (same as `make latexpdf`)
build_pdf_args = ["latexpdf", OUT_DIR, build_output]
sphinx_build.run_make_mode(args=build_pdf_args)