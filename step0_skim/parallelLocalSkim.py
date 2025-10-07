import argparse
import multiprocessing

ncpus = multiprocessing.cpu_count()
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Input root file or txt file containing the list of files to process",
)
parser.add_argument("-o", "--output", type=str, required=True, help="Output file name")
parser.add_argument(
    "--maxGenPU", type=int, default=70, help="Maximum number of PU interactions"
)
parser.add_argument(
    "--maxGenPt", type=int, default=100, help="Maximum number of PU interactions"
)
parser.add_argument(
    "--maxEvents", type=int, default=-1, help="Maximum number of events to process"
)
parser.add_argument(
    "--year", type=str, default="2023", help="Year (needed for setting the GT)"
)
parser.add_argument(
    "--data", action="store_true", help="If set, the input file is data"
)
parser.add_argument(
    "--storeUnmatched",
    action="store_true",
    help="Store also LP electrons not matched to GenEle (only for MC)",
)
parser.add_argument(
    "--ncpu",
    type=int,
    default=ncpus,
    help="Number of threads to use for processing (default: all available)",
)

args = parser.parse_args()

if args.input.endswith(".txt"):
    with open(args.input, "r") as f:
        files = f.read().splitlines()
    filelist = ["root://cms-xrd-global.cern.ch/" + f for f in files]

chunks = [
    ",".join(filelist[i*len(filelist)//args.ncpu : (i+1)*len(filelist)//args.ncpu if i < args.ncpu - 1 else len(filelist)])
    for i in range(args.ncpu)
]


command = f'cmsRun skimMini.py -i @@CHUNK@@ -o {args.output.replace(".root","_@@IDX@@.root")} --year "{args.year}" --maxEvents {args.maxEvents} --maxGenPU {args.maxGenPU} --maxGenPt {args.maxGenPt}'
if args.data:
    command += " --data"
if args.storeUnmatched:
    command += " --storeUnmatched"
for chunk_idx, chunk in enumerate(chunks):
    cmd = command.replace("@@CHUNK@@", chunk).replace("@@IDX@@", str(chunk_idx))
    print(f"Running chunk {chunk_idx+1}/{args.ncpu}: {cmd}")
    import os
    os.system(cmd+" &")
