"""Physical constants and conversion factors

As a general convention, all classes will assume that the units are in SI. Nonetheless, conversions will be applied when importing the input file
"""
from collections.abc import Callable
from pprint import pformat

# ------------------------------
# CONSTANTS IN SI
# ------------------------------
kB = 1.380662e-23 #[J/K]
h = 6.626176e-34 #[J.s]
R = 8.31446261815324 #[J/(K.mol)]
NA = 6.02214076e23 #[1/mol]

# Help / Cues for users
CONSTANTS : dict = {
    "kB" : "Boltzmann Constant in [J/K]",
    "h" : "Planck Constant in [J.s]",
    "R" : "Universal gas constant in [J/(K.mol)]",
    "NA" : "Avogadro's constant in [1/mol]"
}

def help_constants(constant: str|None = None):
    """Returns info about specific constant in the code, or about all the constants included"""
    if constant is None:
        return CONSTANTS
    else:
        supported_constants = ', '.join(CONSTANTS.keys())
        return CONSTANTS.get(constant, f"constant {constant} not supported. Supported values: {supported_constants}")


# ------------------------------
# CONVERSIONS
# ------------------------------
ConversionDict = dict[str, Callable[[float], float]]

# Help / Cues for users
SUPPORTED_UNITS : dict[tuple[str,list[str]]] = {
    "energy" : ("J", ["eV","cal"]),
    "mass" : ("kg", ["amu"]),
    "length" : ("m", ["A"]),
    "pressure" : ("Pa", ["bar", "atm"]),
}

def help_units():
    """Returns a string with information about units, base units used for calculations, and supported units for conversions"""
    s = "SUPPORTED UNITS:"
    for k, v in SUPPORTED_UNITS.items():
        s += f"\n\n{k}:\n"
        s += f"Base unit: {v[0]}\n"
        s += "Supported units for conversions: "
        s += ", ".join(v[1])
    return s


# Energy (base unit here will be J)
to_J : ConversionDict = {
    "eV" : lambda x: x * 1.60218e-19,
    "cal" : lambda x: x * 4.184,
}

from_J : ConversionDict = {
    "eV" : lambda x: x * 6.242e18,
    "cal" : lambda x: x * 0.239006,
}


# Mass (base unit will be kg)
to_kg : ConversionDict = {
    "amu" : lambda x: x * 1.66053886e-27,
}

from_kg : ConversionDict = {
    "amu" : lambda x: x * 6.022e26,
}


# Length (base unit will be m)
to_m : ConversionDict = {
    "A" : lambda x: x * 1e-10,
}

from_m : ConversionDict = {
    "A" : lambda x: x * 1e10,
}


# Pressure (base unit will be Pa)
to_Pa : ConversionDict = {
    "bar" : lambda x: x * 1.0e5,
    "atm" : lambda x: x * 1.01325e5
}

from_Pa : ConversionDict = {
    "bar" : lambda x: x * 1.0e-5,
    "atm" : lambda x: x / 1.01325e5
}


# Testing
if __name__ == "__main__":
    test_constants = False
    test_conversion = False

    # Constants
    if test_constants:
        print(help_constants("kB"))
        print(help_constants())
        print(help_constants("MadeUp"))

    # Conversion factors
    if test_conversion:
        print(to_J["eV"](5))
        print(from_m["A"](0.0001))
        print(help_units())