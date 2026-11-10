"""Embed a trace (from make_trace.py) into the debugger page.

    python src/demo/build_demo.py [trace.json] [out.html]
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
trace = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "trace.json")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "credit_debugger.html")
data = open(trace).read().replace("</", "<\\/")  # keep the JSON from closing its <script>
page = open(os.path.join(here, "template.html")).read().replace("/*TRACE*/", data)
open(out, "w").write(page)
print(out, f"{os.path.getsize(out) / 1e6:.2f} MB")
