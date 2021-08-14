

def checkplate(plate):
    if len(plate) != 8:
        return False

    if plate[2].isdigit():
        return False

    ac=0
    for i in plate:
        if not i.isdigit():
            ac = ac + 1
    if ac != 1:
        return False

    return True