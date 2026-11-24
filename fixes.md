# Fixes for PyCatKin
I will write here all the things I see can be improved in the current package without tons of work (this is effectively a backlog)

>NOTE: I should ask Opencode to help me understand what basic attributes and methods each class should have (based on how other classes call them and use them)

---
# constants
1. I should have a more elegant approach to unit conversion. Right now, I have variables (au2eV) that I import with "*". A better approach would be to have a dataclass or a dictionary (I just use get to access the conversions at runtime)

---
# Classes
**GENERAL COMMENTS**
1. Avoid populating attributes when calling methods, there should be a cleaner way
2. Have verbose text strings written separatedly and populate them as templates (cleaner code)

## State
1. Have a more elegant init function and an initializer from a pickle object (though it would be better to deprecate this option for safety reasons)
2. The class has many methods that modify attributes and that create new ones ad-hoc and non-reliably. I think this should be fixed. We should have a series of methods (preferably static) and call them during init. It is also not clear what the State object should have. Perhaps having an abstract class would be better. The main methods with this behavior are get_vibrations and get_atoms
3. There are many "save" functions as well. Perhaps I can also group them into a single call
4. There are many "calc" functions as well. I can have an ad-hoc calculator or just leave them there
5. I have also made a scaling state. I need to modify this as well

## Reaction
