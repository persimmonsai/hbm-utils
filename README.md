## HBM footprint and symbol generation tool

hbm-bump-gen.py generates a KiCad 6/7 footprint and symbol 
for HBM3 based on JEDEC [JESD238A](https://www.jedec.org/standards-documents/docs/jesd238a).

The HBM3 footprint has a very large number of micro bumps with some
pattern based repitition and some unusual layout. The hbm-bump-gen.py 
script was written to move mistakes from hand edited schematic
symbol and package footprint into python bugs. 

## Running the script

The generator has some dependencies detailed in the `requirements.txt` file. 

    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    python3 hbm-bump-gen.py