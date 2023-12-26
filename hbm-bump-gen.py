import absl
import re
from enum import Enum


class PinEncoding(object):

    def __init__(self: object):
        pass

class BijectivePinEncoding(PinEncoding):

    def __init__(self: object, alphabet: str):
        super.__init_subclass__()
        value: int = 1
        self.ordinals = dict()
        self.alphabet = alphabet

        for c in alphabet:
            self.ordinals[c] = value
            value += 1

    def encode(self: object, num: int) -> str:
        """
        Converts any positive integer to base20(letters only) with no 0th case.

        This is a bijective base 20 number system

        Args:
            num: The integer to convert.

        Returns:
            A string representing the integer in base 20.
        """
        result: str = ''

        while num > 0:
            x,y = divmod(num-1, len(self.alphabet))
            num = x
            result = self.alphabet[y] + result
        
        return result

    def decode(self: object, num: str) -> int:
        """
        Converts an upper case string to an integer representation
         
        Args:
            num: The string to convert.

        Returns:
            An integer representing the number in base 10
        """
        multiplier: int = 1
        result: int = 0

        while len(num) > 0:
            ordinal = self.ordinals[num[-1]]
            result = result + ordinal * multiplier
            num = num[0:-1]
            multiplier *= len(self.alphabet)
            
        return result     

# Identity class for linear values
class LinearPinEncoding(PinEncoding):

    def __init__(self):
        super.__init_subclass__()

    def encode(self, num: int) -> int:
        return num
    
    def decode(self, num: int) -> int:
        return num




class BumpType(Enum):
    UNDEFINED = 1,
    DEPOPULATED = 2,
    MECHANICAL = 3,
    SUPPLY = 4,
    SIGNAL = 5

class Bump(object):

    __name: str
    __net: str
    
    __type: BumpType

    def __init__(self: object, name : str, net : str):
        self.__name = name
        self.__net = net
        self.__type = BumpType.UNDEFINED

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value : str | None):
        self.__name = value

    @property
    def net(self) -> str | None:
        return self.__net
    
    @net.setter
    def net(self, value : str | None):
        self.__net = value

    @property 
    def type(self) -> BumpType:
        return self.__type
    
    @type.setter
    def type(self, value: BumpType):
        self.__type = value

class Die(object):

    name: str
    rows: list[str | int]
    cols: list[str | int]
    bump_map: dict[dict[Bump]]

    PE: PinEncoding

    def __init__(self: object, name: str, PE: PinEncoding, rows: list[str | int], cols: list[str | int]):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.bump_map = dict()
        self.PE = PE
        print(rows)

        for y in rows:
            self.bump_map[y] = dict()
            for x in cols:
                name = f"{y}{x}"
                self.bump_map[y][x] = Bump(name, None)

    def ApplyChannel(self: object, instance: str, row: str, col: int):

        channel_pattern = [   
            [None,'DQ[7]', None,'DQ[5]', None, 'RD[0]', None,    'DQ[3]', None,    'DQ[1]', None, 'ECC[0]' ],
            ['DBI[0]', None,'DQ[6]', None, 'DQ[4]', None,  'DPAR[0]', None,   'DQ[2]', None,   'DQ[0]', None   ],
            [],
            ['DBI[1]', None, 'DQ[14]', None, 'DQ[12]', None, 'WDQS[0]_t', None, 'DQ[10]', None, 'DQS', None],
            [None, 'DQ[15]', None, 'DQ[13]', None, 'WDQS[0]_c', None, 'DQ[11]', None, 'DQ[9]', None, 'ECC[1]'],
            [],
            [],
            ['DBI[2]', None, 'DQ[22]', None, 'DQ[20]', None, 'RDQS[0]_t', None, 'DQ[18]', None, 'DQ[16]', None],
            [None, 'DQ[23]', None, 'DQ[21]', None, 'RDWS[0]_c', None, 'DQ[19]', None, 'DQ[17]', None, 'SEV[0]'],
            [],
            [None, 'DQ[31]', None, 'DQ[29]', None, 'RD[1]', None, 'DQ[27]', None, 'DQ[25]', None, 'SEV[1]'],
            ['DBI[3]', None, 'DQ[30]', None, 'DQ[28]', None, 'DERR[0]', None, 'DQ[26]', None, 'DQ[24]', None],
            [],
            [],
            [None, 'ARFU', None, 'C[7]', None, 'C[5]', None, 'C[4]', None, 'C[2]', None, 'C[0]'],
            ['RA', None, 'APAR', None, 'C[6]', None, 'CK_t', None, 'C[3]', None, 'C[1]', None ],
            [None, 'R[9]', None, 'R[7]', None, 'CK_c', None, 'R[4]', None, 'R[3]', None, 'R[1]'],
            ['AERR', None, 'R[8]', None, 'R[6]', None, 'R[5]', None, 'R[0]', None, 'R[2]', None],
            [],
            [],
            [None, 'DQ[39]', None, 'DQ[37]', None, 'RD[2]', None, 'DQ[35]', None, 'DQ[33]', None, 'ECC[2]'],
            ['DBI[4]', None, 'DQ[38]', None, 'DQ[36]', None, 'DPAR[1]', None, 'DQ[34]', None, 'DQ[32]', None],
            [],
            ['DBI[5]', None, 'DQ[46]', None, 'DQ[44]', None, 'WDQSel_t', None, 'DQ[42]', None, 'DQ[40]', None],
            [None, 'DQ[47]', None, 'DQ[45]', None, 'WDQSel_c', None, 'DQ[43]', None, 'DQ[41]', None, 'ECC[3]'],
            [],
            [],
            ['DBI[6]', None, 'DQ[54]', None, 'DQ[52]', None, 'RDQSel_t', None, 'DQ[50]', None, 'DQ[48]', None],
            [None, 'DQ[55]', None, 'DQ[53]', None, 'RDQSel_c', None, 'DQ[51]', None, 'DQ[49]', None, 'SEV[2]'],
            [],
            [None, 'DQ[63]', None, 'DQ[61]', None, 'RD[3]', None, 'DQ[59]', None, 'DQ[57]', None, 'SEV[3]'],
            ['DBI[7]', None, 'DQ[62]', None, 'DQ[60]', None, 'DERR[1]', None, 'DQ[58]', None, 'DQ[56]', None]
        ]

        for row_list in channel_pattern:
            c = col
            for pin in row_list:
                if pin:
                    self.bump_map[row][c] = pin
                    print(f"{row} {c} {pin}")
                c += 1

            
            row = PE.encode(PE.decode(row) + 1)


                    


class Location(object):

    rows: str
    cols: str

    def __init__(self: object):
        pass
    
if __name__ == "__main__":

    # JESD238A 
    # Rows = [A,HA]
    # Cols = [1,148]

    PE = BijectivePinEncoding(alphabet='ABCDEFGHJKLMNPRTUVWY')

    N_COLS = (148-1) + 1
    N_ROWS = (PE.decode('HA'))

    HBM = Die(name = "HBM3 JESD238A",
              PE = PE, 
              rows = [PE.encode(x) for x in range(1,PE.decode('HA')+1) ],
              cols = range(1,149)
              )

    # Mechanical bumps
    set_mech  = [HBM.bump_map[PE.encode(y)][1]   for y in range(1,N_ROWS+1,2)] 
    set_mech += [HBM.bump_map[PE.encode(y)][2]   for y in range(2,N_ROWS+1,2)] 
    set_mech += [HBM.bump_map[PE.encode(y)][147] for y in range(1,N_ROWS+1,2)] 
    set_mech += [HBM.bump_map[PE.encode(y)][148] for y in range(2,N_ROWS+1,2)] 

    # Set depopulated

    # 
    set_depop = [HBM.bump_map[PE.encode(y)][x] for y in range(1,N_ROWS+1,2) for x in [6, 12, 18, 39, 45, 51, 57, 63, 69, 75, 81, 87]]
 
    # 
    set_depop += [HBM.bump_map[PE.encode(y)][x] for y in range(2,N_ROWS+1,2) for x in [3, 9, 15, 42, 48, 54, 60, 66, 72, 78, 84, 90]]

    set_depop += [HBM.bump_map[PE.encode(y)][x] for y in range(1,N_ROWS+1,2) for x in range(21,37,2)]
    set_depop += [HBM.bump_map[PE.encode(y)][x] for y in range(2,N_ROWS+1,2) for x in range(22,37,2)]

    set_depop += [HBM.bump_map[PE.encode(y)][x] for y in range(PE.decode('CU'),PE.decode('DE')+1,2) for x in range(51,89+1,2)]
    set_depop += [HBM.bump_map[PE.encode(y)][x] for y in range(PE.decode('CV'),PE.decode('DD')+1,2) for x in range(52,90+1,2)]

    for b in set_mech:
        b.type = BumpType.MECHANICAL

    for dp in set_depop:
        dp.type = BumpType.DEPOPULATED

    HBM.ApplyChannel("a", "AA", 100)

